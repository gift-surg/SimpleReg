# \file SimpleItkRegistrationBase.py
# \brief Base class for registration methods
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017

# Import libraries
import pysitk.PythonHelper as ph
import pysitk.SimpleITKHelper as sitkh

from abc import ABCMeta, abstractmethod


##
# Abstract class for registration methods
# \date       2017-08-08 17:27:44+0100
#
class RegistrationBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._registration_transform_sitk = None
        self._computational_time = ph.get_zero_time()

    def run(self):

        time_start = ph.start_timing()

        # Execute registration method
        self._run()

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)
