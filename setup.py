import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quanta_neural_networks",  # Replace with your own username
    version="1.0",
    author="Varun Sundar",
    author_email="vsundar4@wisc.edu",
    description="Transforming photons directly to downstream computer vision objectives",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com:wisionlab/quanta_neural_networks.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)