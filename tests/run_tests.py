#!/usr/bin/python

##
# \file run_tests.py
# \brief      main-file to run specified unit tests
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#


import unittest

# Import modules for unit testing
# from niftyreg_test import *
# from flirt_test import *
# from simple_itk_registration_test import *
# from wrap_itk_registration_test import *
# from point_based_registration_test import *
# from landmark_registration_test import *
from utilities_test import *
# from application_test import *

if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()
