from setuptools import setup, find_packages

setup(
    name="trocr-SWE",
    version="0.1",
    packages=find_packages(),
    entry_points={
            'console_scripts': [
                'trocr-SWE=main:main',
            ],
        },
)