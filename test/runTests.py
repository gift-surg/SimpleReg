#!/usr/bin/python

##
# \file runTests.py
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
from TestNiftyReg import *
from TestFLIRT import *

if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()
