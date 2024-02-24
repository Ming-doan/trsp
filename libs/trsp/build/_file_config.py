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

import yaml
from ..utils._abstract import TritonConfig


class FileConfig:
    '''
    File Configuration Class.
    Used to load and validate configuration file.
    '''

    def __init__(self, file_path: str):
        self.__file_path = file_path
        self.__config = self.__load_config()

    def __load_config(self) -> TritonConfig:
        '''
        Load configuration file and validate its fields.
        '''
        # Load configuration file ------------------------------------------------
        with open(self.__file_path, "r") as f:
            configs: TritonConfig = yaml.safe_load(f)

        # Validate configuration file ---------------------------------------------
        # Check if fields are present
        assert "model_repository" in configs, "Model repository not found in configuration file."
        assert "models" in configs, "Models not found in configuration file."

        # Check if models field is valid.
        for model in configs["models"]:
            model_config = configs["models"][model]

            assert "engine" in model_config, f"Model `engine` not found in configuration models: {model}."
            assert "max_batch_size" in model_config, f"Model `max_batch_size` not found in configuration models: {model}."

            # Check if versions field is present, except for ensemble
            if model_config["engine"] != "ensemble":
                assert "versions" in model_config, f"Model `versions` not found in configuration models: {model}."

            # Check if versions field is valid, except for ensemble
            if model_config["engine"] != "ensemble":
                for version in model_config["versions"]:
                    assert "version" in version, f"Model `version` not found in configuration versions: {model}."

                    # If engine is onnx, check if onnx field is valid
                    if model_config["engine"] == "onnx":
                        assert "path" in version, f"Model `path` not found in configuration versions: {model}."

                    # If engine is python, check if python field is valid
                    if model_config["engine"] == "python":
                        assert "module" in version, f"Model `module` not found in configuration versions: {model}."
                        assert "path" in version[
                            "module"], f"Model `path` not found in configuration module: {model}."
                        assert "execute" in version[
                            "module"], f"Model `execute` not found in configuration module: {model}."

            # If engine is ensemble, check if ensemble field is valid
            if model_config["engine"] == "ensemble":
                assert "steps" in model_config, f"Model `steps` not found in configuration models: {model}."

                for step in model_config["steps"]:
                    assert "model" in step, f"Model `model` not found in configuration steps: {model}."
                    assert "version" in step, f"Model `version` not found in configuration steps: {model}."

            # If instance_group is present, check if it is valid
            if "instance_group" in model_config:
                for group in model_config["instance_group"]:
                    assert "kind" in group, f"Model `kind` not found in configuration instance_group: {model}."

            # For python backend, check if python field is valid
            if model_config["engine"] == "python":
                assert "tensor" in model_config, f"Model `tensor` not found in configuration versions: {model}."

                model_tensor = model_config["tensor"]

                # For each input, check if it is valid
                assert "input" in model_tensor, f"Model `input` not found in configuration versions: {model}."
                for inp in model_tensor["input"]:
                    assert "dims" in inp, f"Model `dims` not found in configuration input: {model}."
                    assert "dtype" in inp, f"Model `dtype` not found in configuration input: {model}."

                # For each output, check if it is valid
                assert "output" in model_tensor, f"Model `output` not found in configuration versions: {model}."
                for out in model_tensor["output"]:
                    assert "dims" in out, f"Model `dims` not found in configuration output: {model}."
                    assert "dtype" in out, f"Model `dtype` not found in configuration output: {model}."

        # Check for application field
        if "app" in configs:
            assert "path" in configs["app"], "Application `path` not found in configuration file."

        # All fields are valid ---------------------------------------------------
        return configs

    def get_config(self) -> TritonConfig:
        '''
        Get configuration dictionary.
        '''
        return self.__config
