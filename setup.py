from setuptools import find_packages, setup
from libs.trsp import __version__

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="trsp",
    version=__version__,
    description="Triton Server Support for building Model repository",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ming-doan",
    author_email="quangminh57dng@gmail.com",
    url="https://github.com/Ming-doan/trsp",
    package_dir={"": "libs"},
    packages=find_packages(where="libs"),
    license="MIT",
    install_requires=[
        "pyyaml",
        "onnx",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "trsp-build = trsp.build:main",
            "trsp-run = trsp.run:main",
        ]
    },
)
