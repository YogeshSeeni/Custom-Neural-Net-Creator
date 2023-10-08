from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='custom_neural_net_creator',
    version='1.2',
    license='MIT',
    author="Yogesh Seenichamy",
    author_email='yogeshseeni60@gmail.com',
    description="A Neural Network Module to create Custom Dense Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/YogeshSeeni/NeuralNetworkModule',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
    ],
)