#!/usr/bin/python

##
# \file run_tests.py
# \brief      main-file to run specified unit tests
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#


# Import libraries
import unittest
import sys
import os

# Import modules for unit testing
# from niftyreg_test import *
# from flirt_test import *
from simple_itk_registration_test import *
from wrap_itk_registration_test import *

if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()
