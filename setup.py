from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

project_urls = {
    "Homepage": 'https://github.com/YogeshSeeni/NeuralNetworkModule',
    "Documentation": "https://custom-neural-net-creator.readthedocs.io/en/latest/"
}

setup(
    name='custom_neural_net_creator',
    version='1.4',
    license='MIT',
    author="Yogesh Seenichamy",
    author_email='yogeshseeni60@gmail.com',
    description="A Neural Network Module to create Custom Dense Neural Networks",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    project_urls=project_urls,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
    ]
)