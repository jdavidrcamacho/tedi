#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='tedi',
      version='0.1',
      description='Implementation of Gaussian and t-Student processes regression',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/tedi',
      packages=['tedi'],
      install_requires=[
        'numpy',
        'scipy'
      ],
     )
