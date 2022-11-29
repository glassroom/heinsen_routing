# coding: utf-8
from setuptools import setup

setup(name='heinsen_routing',
    version='1.0.3',
    description='Implementation of the routing algorithm proposed by Franz A. Heinsen, 2019 and 2022 variants.',
    url='https://github.com/glassroom/heinsen_routing',
    author='Franz A. Heinsen',
    author_email='franz@glassroom.com',
    license='MIT',
    packages=['heinsen_routing'],
    install_requires='torch',
    zip_safe=False)
