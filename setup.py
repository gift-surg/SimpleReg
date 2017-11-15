###
# \file setup.py
#
# Install with symlink: 'pip install -e .'
# Changes to the source file will be immediately available to other users
# of the package
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

from setuptools import setup

long_description = "This package contains the wrapping of FLIRT and NiftyReg "
"to Python."

setup(name='SimpleReg',
      version='0.1rc1',
      description='Wrapped registration tools including FLIRT and NiftyReg',
      long_description=long_description,
      url='https://github.com/gift-surg/SimpleReg',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['simplereg'],
      install_requires=[
          "pysitk==0.1",
          "numpy>=1.13.1",
          "SimpleITK>=1.0.1",
          "nipype>=0.13.1",
      ],
      dependency_links=[
          "git+https://github.com/gift-surg/PySiTK.git@v0.1#egg=PySiTK-0.1"
      ],
      zip_safe=False,
      keywords='development registration',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Healthcare Industry',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: BSD License',

          'Topic :: Software Development :: Build Tools',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ],

      )
