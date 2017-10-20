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
# \see https://python-packaging.readthedocs.io/en/latest/minimal.html
# \see https://python-packaging-user-guide.readthedocs.io/tutorials/distributing-packages/


from setuptools import setup

long_description = "This package contains the wrapping of FLIRT and NiftyReg "
"to Python."

setup(name='SimpleReg',
      version='0.1.dev1',
      description='Wrapped registration tools including FLIRT and NiftyReg',
      long_description=long_description,
      url='https://cmiclab.cs.ucl.ac.uk/gift-surg/SimpleReg',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['simplereg'],
      install_requires=[
          "pysitk",
          "numpy>=1.13.1",
          "SimpleITK>=1.0.1",
          "nipype>=0.13.1",
          "nibabel>=2.2.0",
          "rdflib>=4.2.2",
          "nipy>=0.4.1",
          "dipy>=0.12.0",
      ],
      zip_safe=False,
      keywords='development registration',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ],

      )
