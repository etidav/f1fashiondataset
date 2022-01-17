import setuptools

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt", "r") as f:
    requirements_dev = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="f1fashiondataset",
    version="0.0.1",
    description="Tool for reproducing the HERMES paper benchmark results on the F1 fashion dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/etidav/f1fashiondataset",
    author="Etienne DAVID",
    author_email="etienne.david12@gmail.com",
    license="MIT",
    packages=["f1fashiondataset"],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={"dev": requirements_dev}
)
