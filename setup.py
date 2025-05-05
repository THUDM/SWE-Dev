import setuptools
from pathlib import Path

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

long_description = Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="swedev",
    version="0.1.0",
    author="Haoran Wang",
    author_email="ubecwang@gmail.com", 
    description="Software Engineering Agents with Training and Inference Scaling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UbeCc/SWE-Dev",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
) 