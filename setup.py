"""
    Setup file for instalation
"""
from setuptools import setup

setup(name='tedi',
      version='1.4',
      description='Implementation of Gaussian and Student-t processes regression',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/tedi',
      packages=['tedi'],
      install_requires=[
        'numpy',
        'scipy',
      ],
     )
