from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="hf_bleeding_edge",
    version="0.0.1",
    install_requires=[
        "torch",
        "transformers",
        "typing",
    ],
)