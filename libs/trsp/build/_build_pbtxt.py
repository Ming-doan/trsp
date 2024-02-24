'''
Triton Server Support for building model repository.
----
Author: Quang-Minh Doan (Vietnam)
Github: https://github.com/Ming-doan/trsp
----
MIT License

Copyright (c) 2024 Quang-Minh Doan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import copy
import shutil
import onnx
from ..utils._abstract import (
    TritonEnum,
    TritonConfig,
    ModelConfig,
    FormatedInputOutputTensors,
    FormatedTensors,
    FormatedTritonConfig,
    EnsembleSchedulingConfig,
    EnsembleSchedulingStep
)
from ..utils._utils import (
    get_absolute_path,
    dictionary_to_string,
    get_backend_string,
    get_dtype_string,
    get_kind_instance,
    get_file_instruction_string,
    get_triton_python_model_config_string
)
from ..utils._constants import (
    INFO_PREFIX,
    SUCCESS_PREFIX,
    BUILD_DIR
)


class BuildProtoBufTxt:
    '''
    Build ProtoBufTxt Class.
    Use to build Triton Server model repository and its configuration files.
    '''

    def __init__(self, data: TritonConfig):
        self.__data = data
        self.__file_name = "config"
        self.__model_repository = get_absolute_path(
            f"{BUILD_DIR}/{self.__data['model_repository']}")

    def __create_folders(self, name: str, model_config: ModelConfig) -> str:
        '''
        Create model folders and subfolders for each version.
        '''
        # Create model directory
        model_path = os.path.join(
            self.__model_repository, name)
        os.makedirs(model_path, exist_ok=True)

        # Create version directories. Ignore if engine is ensemble
        if model_config["engine"] != "ensemble":
            for version in model_config["versions"]:
                version_path = os.path.join(
                    model_path, str(version["version"]))
                os.makedirs(version_path, exist_ok=True)
        # Create at least ensemble version directory
        else:
            version_path = os.path.join(model_path, "1")
            os.makedirs(version_path, exist_ok=True)

        # Return model path
        return model_path

    def __format_config(self, name: str, model_config: ModelConfig) -> FormatedTritonConfig:
        '''
        Format from yaml format to Triton Server config.pbtxt format.
        '''
        # Initialize config
        config: FormatedTritonConfig = {
            "name": name,
        }

        # Add backend
        if model_config["engine"] == "ensemble":
            config["platform"] = "ensemble"
        else:
            config["backend"] = get_backend_string(model_config["engine"])

        # Add max_batch_size
        config["max_batch_size"] = model_config["max_batch_size"]

        # Add inputs and outputs
        for i, inp in enumerate(model_config[f"{name}_input"]):
            config[f"input_{i+1}"] = [inp]
        for i, out in enumerate(model_config[f"{name}_output"]):
            config[f"output_{i+1}"] = [out]

        # Add ensemble scheduling if engine is ensemble
        if model_config["engine"] == "ensemble":
            config["ensemble_scheduling"] = model_config[f"{name}_ensemble_scheduling"]

        # Add dynamic batching if enabled
        if "dynamic_batching" in model_config:
            if model_config["dynamic_batching"]:
                config["dynamic_batching"] = {}
                if "max_queue_delay_microseconds" in model_config:
                    config["dynamic_batching"]["max_queue_delay_microseconds"] = model_config["max_queue_delay_microseconds"]

        # Add instance_group if enabled
        if "instance_group" in model_config:
            config["instance_group"] = []
            for group in model_config["instance_group"]:
                instance_group = {
                    "kind": get_kind_instance(group["kind"])
                }
                if group["kind"] == "gpu":
                    if "count" in group:
                        instance_group["count"] = group["count"]
                if "gpus" in group:
                    instance_group["gpus"] = TritonEnum(group["gpus"])
                config["instance_group"].append(instance_group)

        return config

    def __format_onnx(self, path: str, model_config: ModelConfig) -> FormatedInputOutputTensors:
        '''
        Process ONNX model and generate input and output configs.
        '''
        def __get_onnx_shape(input_layer) -> list[int]:
            '''
            Get ONNX input shape.
            If dynamic batching is enabled, remove batch dimension.
            '''
            # Get tensor shape
            tensor_shape: list[int] = [
                dim.dim_value for dim in input_layer.type.tensor_type.shape.dim]

            # Remove batch dimension if dynamic batching is enabled
            if "dynamic_batching" in model_config:
                if model_config["dynamic_batching"]:
                    return tensor_shape[1:]

            return tensor_shape

        # Initialize configs ---------------------------------------------------
        configs: FormatedInputOutputTensors = {
            "input": [],
            "output": []
        }

        # Process each version -------------------------------------------------
        for version in model_config["versions"]:
            # Load ONNX model
            model_path = get_absolute_path(version["path"])
            onnx_model = onnx.load(model_path)

            # Modify input shape and output if dynamic batching is enabled
            if "dynamic_batching" in model_config:
                if model_config["dynamic_batching"]:
                    # Modify input dimensions param of ONNX model
                    for i, input_layer in enumerate(onnx_model.graph.input):
                        input_layer.type.tensor_type.shape.dim[
                            0].dim_param = f"{input_layer.name}_dynamic_axes_{i+1}"

                    # Modify output dimensions param of ONNX model
                    for i, output_layer in enumerate(onnx_model.graph.output):
                        output_layer.type.tensor_type.shape.dim[
                            0].dim_param = f"{output_layer.name}_dynamic_axes_{i+1}"

            # Get model data type
            model_data_type = model_config["dtype"] if "dtype" in model_config else "float32"

            # Define mapping values for TritonEnum
            # If input or output of the model is a string, replace 0 with -1
            mapping_values = {
                "0": "-1"
            }

            # Add input configs
            for input_layer in onnx_model.graph.input:
                input_config: FormatedTensors = {
                    "name": input_layer.name,
                    "data_type": get_dtype_string(model_data_type),
                    "dims": TritonEnum(__get_onnx_shape(input_layer), mapping_values=mapping_values)
                }
                configs["input"].append(input_config)
            # Add output configs
            for output_layer in onnx_model.graph.output:
                output_config: FormatedTensors = {
                    "name": output_layer.name,
                    "data_type": get_dtype_string(model_data_type),
                    "dims": TritonEnum(__get_onnx_shape(output_layer), mapping_values=mapping_values)
                }
                configs["output"].append(output_config)

            # Save ONNX model
            model_save_path = os.path.join(
                path, str(version["version"]), "model.onnx"
            )
            onnx.save(onnx_model, model_save_path)

        # Return input and output configs
        return configs

    def __format_python(self, path: str, model_name: str, model_config: ModelConfig) -> FormatedInputOutputTensors:
        '''
        Process Python model and generate input and output configs.
        '''
        # Initialize configs ---------------------------------------------------
        configs: FormatedInputOutputTensors = {
            "input": [],
            "output": []
        }

        # Add input configs
        for i, inp in enumerate(model_config["tensor"]["input"]):
            input_config: FormatedTensors = {
                "name": f"{model_name}_input_{i+1}",
                "data_type": get_dtype_string(inp["dtype"]),
                "dims": TritonEnum(inp["dims"])
            }
            configs["input"].append(input_config)
        # Add output configs
        for i, out in enumerate(model_config["tensor"]["output"]):
            output_config: FormatedTensors = {
                "name": f"{model_name}_output_{i+1}",
                "data_type": get_dtype_string(out["dtype"]),
                "dims": TritonEnum(out["dims"])
            }
            configs["output"].append(output_config)

        # Write python model file ----------------------------------------------

        # Process each version
        for version in model_config["versions"]:
            # Get absolute path
            model_path = get_absolute_path(version["module"]["path"])
            model_directory = os.path.join(
                path, str(version["version"])
            )

            # Copy python model file to model directory
            shutil.copy(model_path, model_directory)

            # Create model.py file
            model_save_path = os.path.join(
                model_directory, "model.py"
            )

            # Write model.py file
            with open(model_save_path, "w") as f:
                f.write(get_triton_python_model_config_string(
                    model_name, version["module"], configs))

        # Return input and output configs
        return configs

    def __format_ensemble(self, model_name: str, triton_config: TritonConfig) -> tuple[FormatedInputOutputTensors, EnsembleSchedulingConfig]:
        '''
        Process Ensemble model and generate input and output configs.
        '''
        models = triton_config["models"]

        # Initialize configs ---------------------------------------------------
        configs: FormatedInputOutputTensors = {
            "input": [],
            "output": []
        }
        schedule_configs: EnsembleSchedulingConfig = {
            "step": []
        }

        # Assign input equal to the first model input ---------------------------
        first_model_name = models[model_name]["steps"][0]["model"]

        # Raise error if first model input is not found.
        # Which mean the ensemble model is placed before the step model.
        if f"{first_model_name}_input" not in models[first_model_name]:
            raise ValueError(
                f"Model {first_model_name} input not found. Ensemble model {model_name} must be placed after the {first_model_name} model.")

        configs["input"] = copy.deepcopy(models[
            first_model_name][f"{first_model_name}_input"])

        # Modify inputs name
        for i, inp in enumerate(configs["input"]):
            inp["name"] = f"{model_name}_input_{i+1}"

        # Assign output equal to the last model output --------------------------
        last_model_name = models[model_name]["steps"][-1]["model"]
        configs["output"] = copy.deepcopy(models[
            last_model_name][f"{last_model_name}_output"])

        # Modify outputs name
        for i, out in enumerate(configs["output"]):
            out["name"] = f"{model_name}_output_{i+1}"

        # Add scheduling -------------------------------------------------------
        steps_names = []
        # Process each step
        for step_idx, step in enumerate(models[model_name]["steps"]):
            # Append new step name list
            steps_names.append([])

            # Get step version
            step_version = step["version"]
            if step_version == "latest":
                step_version = -1
            elif isinstance(step_version, int):
                step_version = step_version
            else:
                raise ValueError(
                    f"Model version {step_version} is not valid. Must be an integer or 'latest'.")

            # Initialize step config
            step_config: EnsembleSchedulingStep = {
                "model_name": step["model"],
                "model_version": step_version,
            }

            # Add input map
            for i, inp in enumerate(models[f"{step['model']}"][f"{step['model']}_input"]):
                # Get value.
                # If i is 0, value is input name of the ensemble model.
                # Else, value is output_map of previous step model.
                if step_idx == 0:
                    value = f"{model_name}_input_{i+1}"
                else:
                    try:
                        value = steps_names[step_idx-1][i]
                    except IndexError:
                        value = f"{step['model']}_input_map_{i+1}"

                step_config[f"input_map_{i+1}"] = {
                    "key": inp["name"],
                    "value": value
                }

            # Add output map
            for i, out in enumerate(models[f"{step['model']}"][f"{step['model']}_output"]):
                # Get value.
                # If i is last index, value is output name of the ensemble model.
                # Else, create value map for the next input.
                if step_idx == len(models[model_name]["steps"]) - 1:
                    value = f"{model_name}_output_{i+1}"
                else:
                    value = f"{step['model']}_output_map_{i+1}"
                    # Append to step name list
                    steps_names[step_idx].append(value)

                step_config[f"output_map_{i+1}"] = {
                    "key": out["name"],
                    "value": value
                }

            schedule_configs["step"].append(step_config)

        # Clean up un-used output_map value in input_map value ------------------
        # Initialize used input names
        used_input_names = []
        for inp in configs["input"]:
            used_input_names.append(inp["name"])
        for out in configs["output"]:
            used_input_names.append(out["name"])

        # Iterate over input_map value to append to used_input_names
        for step in schedule_configs["step"]:
            # Check all keys contain input_map
            for key, input_map in step.items():
                if "input_map" in key:
                    used_input_names.append(input_map["value"])

        # Remove un-used output_map value in input_map value
        for step in schedule_configs["step"]:
            # Check all keys contain output_map
            _steps = copy.deepcopy(step)
            for key, output_map in _steps.items():
                if "output_map" in key:
                    if output_map["value"] not in used_input_names:
                        del step[key]

        return configs, schedule_configs

    def __generate_pbtxt_string(self, data: FormatedTritonConfig) -> str:
        '''
        Generate config.pbtxt data to string.
        '''
        dict_string = dictionary_to_string(data)
        return get_file_instruction_string(data) + dict_string

    def __write_pbtxt(self, path: str, file_string: str):
        '''
        Write config.pbtxt file to model directory.
        '''
        file_path = os.path.join(path, f"{self.__file_name}.pbtxt")
        with open(file_path, "w") as f:
            f.write(file_string)

    def build(self):
        '''
        Build model repository and its configuration files.
        '''
        # Print info -----------------------------------------------------------
        print(INFO_PREFIX + "Building model repository...")

        # Create model_repository directory if not exists
        os.makedirs(self.__model_repository, exist_ok=True)

        # Process each model ---------------------------------------------------
        for name, model_config in self.__data["models"].items():
            # Create model directory
            model_path = self.__create_folders(name, model_config)

            # Create ONNX model file, if engine is onnx
            if model_config["engine"] == "onnx":
                input_output_configs = self.__format_onnx(
                    model_path, model_config)

            # Create Python model file, if engine is python
            elif model_config["engine"] == "python":
                input_output_configs = self.__format_python(
                    model_path, name, model_config)

            # Create Ensemble model file, if engine is ensemble
            elif model_config["engine"] == "ensemble":
                input_output_configs, scheduling_configs = self.__format_ensemble(
                    name, self.__data)

            # Raise error if engine is not supported
            else:
                raise ValueError(
                    f"Engine {model_config['engine']} is not supported.")

            # Add input and output configs to model config
            model_config[f"{name}_input"] = input_output_configs["input"]
            model_config[f"{name}_output"] = input_output_configs["output"]

            # If model is ensemble, add scheduling config
            if model_config["engine"] == "ensemble":
                model_config[f"{name}_ensemble_scheduling"] = scheduling_configs

            # Create main config data
            config = self.__format_config(name, model_config)

            # Generate and write config.pbtxt file
            proto_string = self.__generate_pbtxt_string(config)
            self.__write_pbtxt(model_path, proto_string)

        # Print success --------------------------------------------------------
        print(SUCCESS_PREFIX +
              f"Build completed. Model repository: {self.__model_repository}")
