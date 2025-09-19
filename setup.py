# setup.py
from setuptools import setup, find_packages

setup(
    name="torchcriterion",
    version="0.1.0",
    description="A modular PyTorch loss function library...",
    author="TransformerTitan",
    packages=find_packages(),
    install_requires=["torch>=2.0,<3.0"],
    python_requires=">=3.7",
)
