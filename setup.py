#!/usr/bin/env python
"""
# Author: Kui XU
# Created Time : Mon 3 Jul 2017 09:42:31 PM CST
# File Name: setup.py
# Description:
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='prismnet',
      version='0.1.1',
      description='PrismNet',
      packages=find_packages(),

      author='Kui XU',
      author_email='kuixu.cs@gmail.com',
      url='https://github.com/kuixu/PrismNet',
      install_requires=requirements,
      python_requires='>3.6.0',

      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: CentOS :: Linux',
     ],
     )
