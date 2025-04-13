from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="risk-models",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid quantum-classical risk modeling library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/risk-models",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-risk-model=risk_models.scripts.train_model:main",
            "predict-risk=risk_models.scripts.predict:main",
        ],
    },
) 