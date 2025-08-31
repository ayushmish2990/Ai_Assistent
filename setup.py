from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coding-ai-model",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An advanced AI coding assistant with code generation and debugging capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coding-ai-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "data": [
            "nltk>=3.8.1",
            "pygments>=2.14.0",
            "tree-sitter>=0.20.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "code-ai-train=scripts.training.train_full_pipeline:main",
            "code-ai-serve=scripts.deployment.serve_api:main",
        ],
    },
)
