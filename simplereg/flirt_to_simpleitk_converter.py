# \file flirt_to_simpleitk_converter.py
#  \brief Class to convert between FLIRT and SimpleITK representations
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2018


import os
import sys
import numpy as np
import SimpleITK as sitk
import nipype.interfaces.c3

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Class to convert between FLIRT and SimpleITK representations
# \date       2018-06-10 22:20:41-0600
#
class FlirtToSimpleItkConverter(object):

    ##
    # Convert FLIRT to SimpleITK transform
    # \date       2018-06-10 16:09:56-0600
    #
    # \param      path_to_flirt_mat       Path to FLIRT matrix
    # \param      path_to_fixed           Path to fixed image used by FLIRT (-ref)
    # \param      path_to_moving          Path to moving image used by FLIRT (-src)
    # \param      path_to_sitk_transform  Path to output SimpleITK transform
    # \param      verbose                 Turn on/off verbose output
    #
    @staticmethod
    def convert_flirt_to_sitk_transform(
            path_to_flirt_mat,
            path_to_fixed,
            path_to_moving,
            path_to_sitk_transform,
            verbose=0,
    ):

        ph.create_directory(os.path.dirname(path_to_sitk_transform))

        c3d = nipype.interfaces.c3.C3dAffineTool()
        c3d.inputs.reference_file = path_to_fixed
        c3d.inputs.source_file = path_to_moving
        c3d.inputs.transform_file = path_to_flirt_mat
        c3d.inputs.fsl2ras = True
        c3d.inputs.itk_transform = path_to_sitk_transform

        if verbose:
            ph.print_execution(c3d.cmdline)
        c3d.run()

    ##
    # Convert SimpleITK to FLIRT transform
    #
    # Remark: Conversion to FLIRT only provides 4 decimal places
    # \date       2018-06-10 16:09:56-0600
    #
    # \param      path_to_sitk_transform  Path to SimpleITK transform
    # \param      path_to_fixed           Path to fixed image used for registration
    # \param      path_to_moving          Path to moving image used for reg.
    # \param      path_to_flirt_mat       Path to output FLIRT matrix
    # \param      verbose                 Turn on/off verbose output
    #
    @staticmethod
    def convert_sitk_to_flirt_transform(
            path_to_sitk_transform,
            path_to_fixed,
            path_to_moving,
            path_to_flirt_mat,
            verbose=0,
    ):

        ph.create_directory(os.path.dirname(path_to_flirt_mat))

        c3d = nipype.interfaces.c3.C3dAffineTool()
        c3d.inputs.reference_file = path_to_fixed
        c3d.inputs.source_file = path_to_moving

        # position of -ras2fsl matters!!
        c3d.inputs.args = "-itk %s -ras2fsl -o %s" % (
            path_to_sitk_transform, path_to_flirt_mat)
        if verbose:
            ph.print_execution(c3d.cmdline)
        c3d.run()
