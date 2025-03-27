# setup.py

from setuptools import setup, find_packages

setup(
    name="cgp_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'copy',   
        'random'    
    ],
    description="Library for Cartesian Genetic Programming",
    author="Ondrej Kováč",
    author_email="xkovac57@vutbr.cz",
)