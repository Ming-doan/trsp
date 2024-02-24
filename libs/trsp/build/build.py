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

import argparse
import shutil
from ._file_config import FileConfig
from ._build_pbtxt import BuildProtoBufTxt
from ..utils._abstract import TritonConfig
from ..utils._utils import get_absolute_path
from ..utils._constants import ERROR_PREFIX, WARNING_PREFIX, BUILD_DIR
from ..utils._docker import get_docker_template


# Define argument parser
parser = argparse.ArgumentParser(
    description='Triton Server Build Model Module.')


def main():
    '''
    Main function for Triton Server Build Model Module.
    '''
    # Add arguments -----------------------------------------------------------
    # Model repository path. Eg: /path/to/model/repository
    parser.add_argument('--model-repository', type=str,
                        help='Path to the model repository')

    # Model model path. Eg: /path/to/model/model.onnx
    parser.add_argument('--model-path', type=str,
                        help='Path to the model model')

    # Model name. Name for triton services. Eg: model
    parser.add_argument('--model-name', type=str,
                        help='Name of the model to be built')

    # Model version. Eg: 1
    parser.add_argument('--model-version', type=int,
                        help='Version of the model to be built')

    # Maximum batch size of Triton server. Eg: 0 (default: 0, no batching)
    parser.add_argument('--max-batch-size', type=int,
                        help='Maximum batch size of Triton server')

    # Enable dynamic batching of Triton server. Eg: False (default: False)
    parser.add_argument('--dynamic-batching', type=bool,
                        help='Enable dynamic batching of Triton server')

    # Model configuration file path. Eg: /path/to/model/config.yaml
    parser.add_argument('-f', type=str,
                        help='Path to the model configuration file. If provided, other arguments will be ignored.')

    # Rebuild model repository. Eg: False (default: False)
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild model repository. If provided, model repository will be rebuilt.')

    # Parse arguments --------------------------------------------------------
    args = parser.parse_args()

    # Load configuration file if provided ------------------------------------
    if args.f:
        # Show warning if other arguments are provided.
        if args.model_repository or args.model_name or args.model_path or args.model_version or args.max_batch_size or args.dynamic_batching:
            print(WARNING_PREFIX +
                  "Configuration file provided. Other arguments will be ignored.")

        # Load configuration file
        config = FileConfig(args.f).get_config()

    # Use provided arguments -------------------------------------------------
    else:
        # Default configuration
        args.model_version = args.model_version or 1
        args.max_batch_size = args.max_batch_size or 0
        args.dynamic_batching = args.dynamic_batching or False

        # Check for missing arguments
        if not args.model_repository:
            print(
                ERROR_PREFIX + "Model repository path is required. Use --model-repository to specify.")
            return
        if not args.model_path:
            print(ERROR_PREFIX + "Model path is required. Use --model-path to specify.")
            return
        if not args.model_name:
            print(ERROR_PREFIX + "Model name is required. Use --model-name to specify.")
            return

        # Initialize configuration
        config: TritonConfig = {
            "model_repository": args.model_repository,
            "models": {}
        }

        # Add models configuration
        config["models"][args.model_name] = {
            "engine": "onnx",
            "dtype": "float32",
            "max_batch_size": args.max_batch_size,
            "dynamic_batching": args.dynamic_batching,
            "versions": [
                {
                    "version": args.model_version,
                    "path": args.model_path
                }
            ]
        }

    # Rebuild model repository if provided ------------------------------------
    if args.rebuild:
        shutil.rmtree(get_absolute_path(
            f"{BUILD_DIR}/{config['model_repository']}"))

    # Write to model repository
    try:
        BuildProtoBufTxt(config).build()
    except Exception as e:
        print(ERROR_PREFIX + str(e))

    # Create docker file
    if "requirements" not in config:
        config["requirements"] = []
    try:
        with open(get_absolute_path(BUILD_DIR + "/Dockerfile"), "w") as f:
            f.write(get_docker_template(config["requirements"]))
    except Exception as e:
        print(ERROR_PREFIX + str(e))


# Run main function if module is run directly
if __name__ == '__main__':
    main()
