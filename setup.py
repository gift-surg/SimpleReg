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

setup(name='RegistrationTools',
      version='0.1.dev1',
      description='Wrapped registration tools including FLIRT and NiftyReg',
      long_description=long_description,
      url='https://cmiclab.cs.ucl.ac.uk/mebner/RegistrationTools',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='MIT',
      packages=['registrationtools'],
      install_requires=['pythonhelper'],
      zip_safe=False,
      keywords='development registration',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
           'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          # 'Programming Language :: Python :: 2',
          # 'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          # 'Programming Language :: Python :: 3',
          # 'Programming Language :: Python :: 3.2',
          # 'Programming Language :: Python :: 3.3',
          # 'Programming Language :: Python :: 3.4',
      ],

      )
