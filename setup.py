from setuptools import setup, find_packages

setup(
    name="neuroflux",
    version="1.0.0",
    description="Self-Optimizing, Fault-Tolerant Architecture for Scalable AI",
    author="NeuroFlux Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "einops>=0.6.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "datasets>=2.12.0",
        "torch_xla>=2.0",
        "google-cloud-storage>=2.0.0",
        "sortedcontainers>=2.4.0",
        "plotly>=5.13.0",
        "pandas>=1.5.0",
        "seaborn>=0.12.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "psutil>=5.9.0",
        "gputil>=1.4.0"
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest-benchmark",
        ],
        "tpu": [
            "torch_xla",
            "google-cloud-storage"
        ]
    }
)
