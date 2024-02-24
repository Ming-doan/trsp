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

# Chroma logger prefix
ERROR_PREFIX = "\033[91mERROR: \033[0m"
SUCCESS_PREFIX = "\033[92mSUCCESS: \033[0m"
WARNING_PREFIX = "\033[93mWARNING: \033[0m"
INFO_PREFIX = "\033[94mINFO: \033[0m"


# Constants
TRITON_PRESEVED_KEYWORDS = [
    "model", "config", "triton_python_backend_utils", "pb_utils", "TritonPythonModel"]
BUILD_DIR = "build"
