#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import unittest

setuptools.setup(name='endotip',
    version='0.1.0',
    description="""Python module to localize the tooltip of surgical tools in endoscopic 
        images.""",
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT License',
    url='https://github.com/luiscarlosgph/endotip',
    packages=['endotip'],
    package_dir={'endotip' : 'src'}, 
    install_requires = [
        'numpy', 
        'opencv-python', 
        'scikit-image', 
        'sknw',
        'sklearn',
        'endoseg',
    ],
    #test_suite = 'tests',
)
