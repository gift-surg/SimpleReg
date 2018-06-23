##
# \file resampler.py
# \brief      Class to perform resampling operations
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       June 2018
#


import os
import itk
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import simplereg.data_reader as dr
import simplereg.data_writer as dw
import simplereg.utilities as utils
from simplereg.niftyreg_to_simpleitk_converter import \
    NiftyRegToSimpleItkConverter as nreg2sitk

from simplereg.definitions import ALLOWED_INTERPOLATORS


class Resampler(object):

    def __init__(self,
                 path_to_fixed,
                 path_to_moving,
                 path_to_transform,
                 interpolator="Linear",
                 spacing=None,
                 padding=0,
                 add_to_grid=0,
                 verbose=0,
                 ):

        self._path_to_fixed = path_to_fixed
        self._path_to_moving = path_to_moving
        self._path_to_transform = path_to_transform
        self._interpolator = interpolator
        self._spacing = spacing
        self._padding = padding
        self._add_to_grid = add_to_grid
        self._verbose = verbose

        self._warped_moving_sitk = None
        self._warped_moving_itk = None

    def write_image(self, path_to_output):
        if self._warped_moving_sitk is not None:
            dw.DataWriter.write_image(
                self._warped_moving_sitk, path_to_output)

    def run(self):

        interpolator_sitk = self._convert_interpolator(self._interpolator)

        # read input
        fixed_sitk = dr.DataReader.read_image(self._path_to_fixed)
        moving_sitk = dr.DataReader.read_image(self._path_to_moving)
        if self._path_to_transform is not None:
            transform_sitk = dr.DataReader.read_transform(
                self._path_to_transform)
        else:
            transform_sitk = getattr(
                sitk, "Euler%dDTransform" % fixed_sitk.GetDimension())()

        # resample image
        size, origin, spacing, direction = self.get_space_resampling_properties(
            image_sitk=fixed_sitk,
            spacing=self._spacing,
            add_to_grid=self._add_to_grid,
            add_to_grid_unit="mm")
        self._warped_moving_sitk = sitk.Resample(
            moving_sitk,
            size,
            transform_sitk,
            interpolator_sitk,
            origin,
            spacing,
            direction,
            float(self._padding),
            fixed_sitk.GetPixelIDValue(),
        )

    @staticmethod
    def _convert_interpolator(interpolator):
        if interpolator.isdigit():
            if int(interpolator) == 0:
                interpolator_sitk = sitk.sitkNearestNeighbor
            elif int(interpolator) == 1:
                interpolator_sitk = sitk.sitkLinear
            else:
                raise IOError("Interpolator order not known")
        else:
            if interpolator in ALLOWED_INTERPOLATORS:
                interpolator_sitk = getattr(
                    sitk, "sitk%s" % interpolator)
            else:
                raise IOError("Interpolator not known.")
        return interpolator_sitk

    ##
    # Gets the space resampling properties given an image an optional
    # spacing/grid adjustment desires.
    # \date       2018-05-03 13:04:05-0600
    #
    # \param      image_sitk        Image as sitk.Image object
    # \param      spacing           Spacing for resampling space. If scalar,
    #                               isotropic resampling grid is assumed
    # \param      add_to_grid       Additional grid extension/reduction in each
    #                               direction of each axis. If scalar, changes
    #                               are applied uniformly to grid
    # \param      add_to_grid_unit  Changes to grid size to be understood in
    #                               either millimeter ("mm") or voxel units
    #
    # \return     The space resampling properties: size_out, origin_out,
    #             spacing_out, direction_out
    #
    @staticmethod
    def get_space_resampling_properties(
        image_sitk,
        spacing=None,
        add_to_grid=None,
        add_to_grid_unit="mm",
    ):

        if not isinstance(image_sitk, sitk.Image):
            raise IOError("Image must be of type sitk.Image")

        dim = image_sitk.GetDimension()

        # Read input image information:
        spacing_in = np.array(image_sitk.GetSpacing())
        size_in = np.array(image_sitk.GetSize()).astype(int)
        origin_out = np.array(image_sitk.GetOrigin())
        direction_out = np.array(image_sitk.GetDirection())

        # Check given spacing information for grid resampling
        if spacing is not None:
            spacing_out = np.atleast_1d(spacing).astype(np.float64)
            if spacing_out.size != 1 and spacing_out.size != dim:
                raise IOError(
                    "spacing information for resampling must either match the "
                    "image dimension or is 1D (isotropic resampling)")
            if spacing_out.size == 1:
                spacing_out = np.ones(dim) * spacing_out[0]
        else:
            spacing_out = spacing_in

        # Check given (optional) add_to_grid information
        if add_to_grid is not None:
            add_to_grid = np.atleast_1d(add_to_grid)
            if add_to_grid.size != 1 and add_to_grid.size != dim:
                raise IOError(
                    "add_to_grid must either match the image "
                    "dimension or is 1D (uniform change)")
            if add_to_grid.size == 1:
                add_to_grid = np.ones(dim) * add_to_grid[0]

            # Get scaling for offset and grid change in (continuous) voxels
            if add_to_grid_unit == "mm":
                scale = add_to_grid
                add_to_grid_vox = add_to_grid / spacing_out
            else:
                scale = add_to_grid / spacing_out
                add_to_grid_vox = add_to_grid

            # Offset origin to account for change in grid size
            offset = np.eye(dim)
            for i in range(dim):
                # For Python3: Get standard integers; otherwise error
                index = [int(j) for j in offset[:, i]]
                offset[:, i] = image_sitk.TransformIndexToPhysicalPoint(index)
                offset[:, i] -= origin_out
                offset[:, i] /= np.linalg.norm(offset[:, i])
            origin_out -= np.sum(offset, axis=1) * scale
        else:
            add_to_grid_vox = 0

        size_out = size_in * spacing_in / spacing_out + 2 * add_to_grid_vox
        size_out = np.round(size_out).astype(int)

        # For Python3: sitk.Resample in Python3 does not like np.int types!
        size_out = [int(i) for i in size_out]

        return size_out, origin_out, spacing_out, direction_out

    ##
    # Gets the resampled image sitk.
    # \date       2018-05-03 13:09:27-0600
    #
    # \param      image_sitk        The image sitk
    # \param      spacing           The spacing
    # \param      interpolator      The interpolator
    # \param      padding           The padding
    # \param      add_to_grid       The add to grid
    # \param      add_to_grid_unit  The add to grid unit
    #
    # \return     The resampled image sitk.
    #
    @staticmethod
    def get_resampled_image_sitk(
            image_sitk,
            spacing=None,
            interpolator=sitk.sitkLinear,
            padding=0,
            add_to_grid=None,
            add_to_grid_unit="mm"):

        size, origin, spacing, direction = self.get_space_resampling_properties(
            image_sitk=image_sitk, spacing=spacing, add_to_grid=add_to_grid,
            add_to_grid_unit=add_to_grid_unit)

        resampled_image_sitk = sitk.Resample(
            image_sitk,
            size,
            getattr(sitk, "Euler%dDTransform" % image_sitk.GetDimension())(),
            interpolator,
            origin,
            spacing,
            direction,
            padding,
            image_sitk.GetPixelIDValue()
        )

        return resampled_image_sitk
