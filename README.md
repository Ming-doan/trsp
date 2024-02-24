# Triton Server Support for building Model repository

![coverage](https://img.shields.io/badge/coverage-60%-yellow)
![version](https://img.shields.io/badge/version-0.0.1_alpha-blue)
![licence](https://img.shields.io/badge/licence-MIT-red)

> This package help building model repository of Triton Server with more easy `.yaml` file.

⚠️ **TO USE THIS LIBRARY, UNDERSTAND TRITON INFERENCE SERVER FIRST. [TRITON INFERENCE SERVER TUTORIAL HERE](https://github.com/triton-inference-server/tutorials).**

## 👋 Installation

Install `trsp` package in Python.

```bash
pip install trsp
```

## ⚡ Quick command

- Build model repository with config file.

```bash
trsp-build -f /path/to/config.yaml
```

- Launch Triton Server with Docker.

```bash
trsp-run
```

## 📃 Configuration file with `.yaml` or `.yml`

To start, let's create a `config.yaml` file.

### ONNX Model.

Suppose we have an onnx model named `mymodel.onnx`. Here is folder structure:

```
config.yaml
mymodel.onnx
```

Define model config to `config.yaml`.

```yaml
model_repository: name_of_repository

models:
  my_model: # Write your own model name
    engine: onnx
    max_batch_size: 0
    versions:
      - version: 1
        path: mymodel.onnx
```

This config will create a model repository formated as Triton Inference Server requirements. It's look-like:

```
build/
  name_of_repository/
    my_model/
      1/
        model.onnx
      config.pbtxt
```

### Python Model.

To create a python model, create a python file `my_logic.py` to define core logic as bellow:

```
config.yaml
my_logic.py
```

```python
import numpy

def my_initialize(args: dict):
    # The args here is a dictionary contains config provided by Triton.
    ...
    return {}

def my_logic(args, inputs: list[np.ndarray]):
    # The args here is any that you return from the above function.
    # The inputs here is a list of numpy array that contain your input data.
    ...
    return (processed_data,) # Return a tuple of processed things.
```

Write a configuration.

```yaml
model_repository: name_of_repository

models:
  my_python_model:
    engine: python
    max_batch_size: 0
    versions:
      version: 1
      module:
        path: my_logic.py
        initialize: my_initialize
        execute: my_logic
    # You must define input and output shape and data type of the python model.
    tensor:
      input:
        - dims: [1, 2, 3, 4]
          dtype: float32
      output:
        - dims: [1, 2, 3, 4]
          dtype: float32

# List out the library that you use for your logic for trsp install it when run.
requirements:
  - numpy
```

### Ensemble Model.

Create ensemble model in Triton Server. Define in configuration as below:

```yaml
model_repository: name_of_repository

models:
  my_ensemble_model:
    engine: ensemble
    max_batch_size: 0
    steps:
      - model: my_model
        version: latest
      - model: my_python_model
        version: latest
```

## 😊 Contributors

- Quang-Minh Doan - [Ming-doan](https://github.com/Ming-doan)
