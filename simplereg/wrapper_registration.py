# \file WrapperRegistration.py
# \brief Class to provide basis to wrap command line registration tools
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017

# Import libraries
from abc import ABCMeta, abstractmethod

from simplereg.simple_itk_registration_base \
    import SimpleItkRegistrationBase


##
# Abstract class to wrap registration methods from SimpleITK objects
# \date       2017-08-08 17:27:44+0100
#
class WrapperRegistration(SimpleItkRegistrationBase):
    __metaclass__ = ABCMeta

    ##
    # Store information which are considered as basic for all registration
    # tools
    # \date       2017-08-08 17:28:02+0100
    #
    # \param      self              The object
    # \param      fixed_sitk        Fixed image as sitk.Image object
    # \param      moving_sitk       Moving image as sitk.Image object
    # \param      fixed_sitk_mask   Fixed image mask as sitk.Image object
    # \param      moving_sitk_mask  Moving image mask as sitk.Image object
    # \param      options           Options to add for command line tool;
    #                               string
    #
    def __init__(self,
                 fixed_sitk,
                 moving_sitk,
                 fixed_sitk_mask,
                 moving_sitk_mask,
                 options,
                 verbose,
                 ):

        SimpleItkRegistrationBase.__init__(self,
                                           fixed_sitk=fixed_sitk,
                                           moving_sitk=moving_sitk,
                                           fixed_sitk_mask=fixed_sitk_mask,
                                           moving_sitk_mask=moving_sitk_mask,
                                           )
        self._options = options
        self._verbose = verbose

    ##
    # Sets the options of the registration method.
    # \date       2017-08-08 17:26:45+0100
    #
    # \param      self     The object
    # \param      options  The options as string
    #
    def set_options(self, options):
        self._options = options

    ##
    # Gets the options.
    # \date       2017-08-08 17:27:07+0100
    #
    # \param      self  The object
    #
    # \return     The options as string
    #
    def get_options(self):
        return self._options

    ##
    # Sets the verbose.
    # \date       2017-11-08 18:26:59+0000
    #
    # \param      self    The object
    # \param      vebose  Turn on/off verbose. Boolean.
    #
    def set_verbose(self, vebose):
        self._verbose = verbose

    ##
    # Gets the verbose.
    # \date       2017-11-08 18:27:16+0000
    #
    # \param      self  The object
    #
    # \return     Verbose as boolean
    #
    def get_verbose(self):
        return self._verbose
