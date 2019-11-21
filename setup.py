from setuptools import setup

reqs = []
with open("requirements.txt") as f:
    reqs.append(f.readline())

setup(
    name='edf-loader',
    version='1.0',
    description='Python library to load channels in EDF files as dictionaries, with support for train/test set splitting.',
    author='Hannes Kr. Hannesson',
    author_email='hanneskr4@gmail.com',
    install_requires=reqs,  # external packages as dependencies
    scripts=[
        'Loader.py',
    ],
    py_modules=["Loader"]
)
