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
from ._abstract import TritonEnum, PythonModuleConfig, FormatedTritonConfig, FormatedInputOutputTensors
from ._constants import TRITON_PRESEVED_KEYWORDS


def get_absolute_path(path: str) -> str:
    '''
    Get absolute path from relative path.
    '''
    paths = path.split("/")
    # Check if path is absolute
    if paths[0] == "":
        return path
    # Check if path is relative
    if paths[0] == ".":
        return os.path.join(os.getcwd(), *paths[1:])
    # Check if path is relative to home directory
    if paths[0] == "~":
        return os.path.join(os.path.expanduser("~"), *paths[1:])
    # Check if path is relative to current directory
    if paths[0] == "..":
        back_count = paths.count("..")
        current_directory = os.getcwd()
        for _ in range(back_count):
            current_directory = current_directory[0:current_directory.rfind(
                "\\")]
        return os.path.join(current_directory, *paths[back_count:])
    return os.path.join(os.getcwd(), path)


def dictionary_to_string(dictionary: FormatedTritonConfig, indent: int = 0, tab: int = 2) -> str:
    '''
    Convert dictionary to pretty string.
    '''
    def __get_special_key(key: str) -> str:
        if "input_map" in key:
            return "input_map"
        if "output_map" in key:
            return "output_map"
        if "input" in key:
            return "input"
        if "output" in key:
            return "output"
        return key

    string = ""
    for key, value in dictionary.items():
        if isinstance(value, str):
            string += f"{' '*indent}{__get_special_key(key)}: \"{value}\"\n"
        if isinstance(value, int):
            string += f"{' '*indent}{__get_special_key(key)}: {value}\n"
        if isinstance(value, TritonEnum):
            string += f"{' '*indent}{__get_special_key(key)}: {value}\n"
        if isinstance(value, list):
            string += f"{' '*indent}{__get_special_key(key)} [\n"
            for i, item in enumerate(value):
                if isinstance(item, str):
                    string += f"{' '*(indent+tab)}\"{item}\"\n"
                if isinstance(item, int):
                    string += f"{' '*(indent+tab)}{item}\n"
                if isinstance(item, TritonEnum):
                    string += f"{' '*(indent+tab)}{item}\n"
                if isinstance(item, dict):
                    string += f"{' '*(indent+tab)}{{\n"
                    string += dictionary_to_string(item, indent+tab*2, tab)
                    string += f"{' '*(indent+tab)}}}{',' if i != len(value) - 1 else ''}\n"
            string += f"{' '*indent}]\n"
        if isinstance(value, dict):
            string += f"{' '*indent}{__get_special_key(key)} {{\n"
            string += dictionary_to_string(value, indent+tab, tab)
            string += f"{' '*indent}}}\n"
    return string


def get_backend_string(engine: str) -> str:
    '''
    Get backend string for Triton Server config.pbtxt file.
    '''
    if engine == "onnx":
        return "onnxruntime"
    if engine == "python":
        return "python"
    return ""


def get_kind_instance(kind: str) -> str:
    '''
    Get kind instance string for Triton Server config.pbtxt file.
    '''
    if kind == "gpu":
        return TritonEnum("KIND_GPU")
    return TritonEnum("KIND_CPU")


def get_dtype_string(dtype: str) -> str:
    '''
    Get data type string for Triton Server config.pbtxt file.
    '''
    if dtype == "float32":
        return TritonEnum("TYPE_FP32")
    if dtype == "float64":
        return TritonEnum("TYPE_FP64")
    if dtype == "int32":
        return TritonEnum("TYPE_INT32")
    if dtype == "int64":
        return TritonEnum("TYPE_INT64")
    if dtype == "uint8":
        return TritonEnum("TYPE_UINT8")
    if dtype == "uint16":
        return TritonEnum("TYPE_UINT16")
    if dtype == "uint32":
        return TritonEnum("TYPE_UINT32")
    if dtype == "uint64":
        return TritonEnum("TYPE_UINT64")
    if dtype == "int8":
        return TritonEnum("TYPE_INT8")
    if dtype == "int16":
        return TritonEnum("TYPE_INT16")
    if dtype == "bool":
        return TritonEnum("TYPE_BOOL")
    if dtype == "string":
        return TritonEnum("TYPE_STRING")
    raise ValueError(f"Unsupported data type: {dtype}")


def get_python_filename(path: str) -> str:
    '''
    Get python filename from path.
    '''
    filename = os.path.basename(path).split(".")[0]
    if filename in TRITON_PRESEVED_KEYWORDS:
        raise ValueError(
            f"Filename {filename} is a preserved keyword. Preserved keywords: {TRITON_PRESEVED_KEYWORDS}")
    return filename


def get_imports_from_modules_data(data: PythonModuleConfig) -> str:
    '''
    Get import from modules data.
    '''
    imports = []
    imports.append(data["execute"])
    if "initialize" in data:
        imports.append(data["initialize"])
    if "finalize" in data:
        imports.append(data["finalize"])

    # Check if function name not in preserved keywords
    for import_name in imports:
        if import_name in TRITON_PRESEVED_KEYWORDS:
            raise ValueError(
                f"Function name {import_name} is a preserved keyword. Preserved keywords: {TRITON_PRESEVED_KEYWORDS}")

    return f"from .{get_python_filename(data['path'])} import {', '.join(imports)}"


def get_python_initialize_function(func_name: str) -> str:
    '''
    Get python initialize function.
    If func_name is None, return ...
    '''
    if func_name:
        return f"self.params = {func_name}(args)"
    return "self.params = None"


def get_python_finalize_function(func_name: str) -> str:
    '''
    Get python finalize function.
    If func_name is None, return ...
    '''
    if func_name:
        return f"{func_name}(self.params)"
    return "..."


def get_tensor_inputs_name(tensor_config: FormatedInputOutputTensors) -> list[str]:
    '''
    Get tensor inputs name.
    '''
    tensor_inputs_name = []
    for tensor in tensor_config["input"]:
        tensor_inputs_name.append(tensor["name"])
    return tensor_inputs_name


def get_tensor_outputs_name(tensor_config: FormatedInputOutputTensors) -> list[str]:
    '''
    Get tensor outputs name.
    '''
    tensor_outputs_name = []
    for tensor in tensor_config["output"]:
        tensor_outputs_name.append(tensor["name"])
    return tensor_outputs_name


# Data is a dictionary. Contain entire model config of Triton pbtxt.
def get_file_instruction_string(data: FormatedTritonConfig): return f'''# Welcome to the Triton Server Protobuf Text Format (PBtxt) file.
# This is auto generated file by `trsp` module. Developed by Ming-doan.
# Models: {data['name']}.
# Engine: {data['backend'] if "backend" in data else data["platform"]}.
# ------------------------------

'''


def get_triton_python_model_config_string(name: str, data: PythonModuleConfig, tensor_config: FormatedInputOutputTensors): return f'''# Model Configuration for Triton Server Python Model.
# Auto generated by `trsp` module. Developed by Ming-doan.
# Model: {name}.
# Engine: python.
# ------------------------------

import triton_python_backend_utils as pb_utils
{get_imports_from_modules_data(data)}

class TritonPythonModel:
    def initialize(self, args):
        {get_python_initialize_function(data['initialize'] if 'initialize' in data else None)}

    def execute(self, requests):
        tensor_inputs_name = {str(get_tensor_inputs_name(tensor_config))}
        tensor_outputs_name = {str(get_tensor_outputs_name(tensor_config))}

        responses = []
        for request in requests:
            # Get input tensors
            input_tensors = []
            for name in tensor_inputs_name:
                input_tensors.append(
                    pb_utils.get_input_tensor_by_name(request, name).as_numpy()
                )

            # Transfer tensors to execute function
            outputs = {data['execute']}(self.params, input_tensors)

            # Create output tensors
            output_tensors = []
            for name, output in zip(tensor_outputs_name, outputs):
                output_tensors.append(pb_utils.Tensor(name, output))

            # Create response
            response = pb_utils.InferenceResponse(output_tensors)

            # Append response
            responses.append(response)
        return responses

    def finalize(self):
        {get_python_finalize_function(data['finalize'] if 'finalize' in data else None)}
'''
