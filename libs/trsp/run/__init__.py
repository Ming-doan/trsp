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
import argparse
import subprocess
from ..utils._constants import BUILD_DIR, ERROR_PREFIX


parser = argparse.ArgumentParser(
    description='Triton Server Running Module.')


def main():
    build_directory = os.path.join(os.getcwd(), BUILD_DIR)
    # Add arguments -----------------------------------------------------------
    # Port to be used by docker container. Eg: 7860
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to be used by docker container')

    # Parse arguments --------------------------------------------------------
    args = parser.parse_args()

    # Build Docker image and run container ------------------------------------

    # Get model repository path
    folders = os.listdir(build_directory)
    model_repository = None
    for folder in folders:
        if os.path.isdir(os.path.join(build_directory, folder)):
            model_repository = folder
            break

    # Looking for Dockerfile in build directory
    if not os.path.exists(os.path.join(build_directory, 'Dockerfile')):
        print(ERROR_PREFIX + 'Dockerfile not found in build directory')
        return

    # Build docker image
    try:
        subprocess.run(
            ['docker', 'build', '-t', 'triton-server', build_directory])
    except Exception as e:
        print(ERROR_PREFIX + 'Failed to build docker image. ' + str(e))
        return

    # Run docker container
    try:
        subprocess.run(['docker', 'run', '-d', '--gpus=all', '-p',
                       f'{args.port}:{args.port}', '-v', f'{os.path.join(os.getcwd(), BUILD_DIR, model_repository)}:/models', 'triton-server'])
    except Exception as e:
        print(ERROR_PREFIX + 'Failed to run docker container. ' + str(e))
        return


# Run function
if __name__ == '__main__':
    main()
