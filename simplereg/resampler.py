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
        else:
            dw.DataWriter.write_image(
                self._warped_moving_itk, path_to_output)

    def run(self):
        # Possible to use _run_itk for all interpolators. However, loading of
        # itk library takes noticeably longer. Hence, only use it when required
        if self._interpolator in ["OrientedGaussian"]:
            if self._path_to_transform is not None:
                # This could be implemented for rigid transformations.
                # For affine, or even displacement fields, it is not quite
                # clear how a PSF-transformed option shall look like.
                raise ValueError(
                    "OrientedGaussian interpolation does not allow a "
                    "transformation during resampling.")
            self._run_itk()
        else:
            self._run_sitk()

    def _run_itk(self):
        # read input
        fixed_itk = dr.DataReader.read_image(self._path_to_fixed, as_itk=1)
        moving_itk = dr.DataReader.read_image(self._path_to_moving, as_itk=1)

        # get image resampling information
        size, origin, spacing, direction = self.get_space_resampling_properties(
            image_sitk=fixed_itk,
            spacing=self._spacing,
            add_to_grid=self._add_to_grid,
            add_to_grid_unit="mm")

        if self._path_to_transform is not None:
            transform_itk = dr.DataReader.read_transform(
                self._path_to_transform, as_itk=1)
        else:
            transform_itk = getattr(
                itk, "Euler%dDTransform" % fixed_itk.GetImageDimension()).New()

        interpolator_itk = self._convert_interpolator_itk(
            fixed_itk,
            moving_itk,
            spacing,
            self._interpolator,
            alpha_cut=3,
        )

        # resample image
        resampler_itk = itk.ResampleImageFilter[
            type(moving_itk), type(fixed_itk)].New()
        resampler_itk.SetInput(moving_itk)
        resampler_itk.SetSize(size)
        resampler_itk.SetTransform(transform_itk)
        resampler_itk.SetInterpolator(interpolator_itk)
        resampler_itk.SetOutputOrigin(origin)
        resampler_itk.SetOutputSpacing(spacing)
        resampler_itk.SetOutputDirection(fixed_itk.GetDirection())
        resampler_itk.SetDefaultPixelValue(self._padding)
        resampler_itk.UpdateLargestPossibleRegion()
        resampler_itk.Update()
        self._warped_moving_itk = resampler_itk.GetOutput()
        self._warped_moving_itk.DisconnectPipeline()

    def _run_sitk(self):
        # read input
        fixed_sitk = dr.DataReader.read_image(self._path_to_fixed)
        moving_sitk = dr.DataReader.read_image(self._path_to_moving)

        # get image resampling information
        size, origin, spacing, direction = self.get_space_resampling_properties(
            image_sitk=fixed_sitk,
            spacing=self._spacing,
            add_to_grid=self._add_to_grid,
            add_to_grid_unit="mm")

        if self._path_to_transform is not None:
            transform_sitk = dr.DataReader.read_transform(
                self._path_to_transform)
        else:
            transform_sitk = getattr(
                sitk, "Euler%dDTransform" % fixed_sitk.GetDimension())()

        # resample image
        self._warped_moving_sitk = sitk.Resample(
            moving_sitk,
            size,
            transform_sitk,
            self._convert_interpolator_sitk(self._interpolator),
            origin,
            spacing,
            direction,
            float(self._padding),
            fixed_sitk.GetPixelIDValue(),
        )

    @staticmethod
    def _convert_interpolator_sitk(interpolator):
        if interpolator.isdigit():
            if int(interpolator) == 0:
                interpolator = "NearestNeighbor"
            elif int(interpolator) == 1:
                interpolator = "Linear"
            else:
                raise ValueError(
                    "Interpolator order not known. Allowed options are: 0, 1")
        if interpolator not in ALLOWED_INTERPOLATORS:
            raise ValueError(
                "Interpolator not known. Allowed options are: %s" % (
                    ", ".join(ALLOWED_INTERPOLATORS)))

        return getattr(sitk, "sitk%s" % interpolator)

    def _convert_interpolator_itk(
        self,
        fixed_itk,
        moving_itk,
        spacing,
        interpolator,
        alpha_cut,
        pixel_type=itk.D,
    ):
        if interpolator.isdigit():
            if int(interpolator) == 0:
                interpolator = "NearestNeighbor"
            elif int(interpolator) == 1:
                interpolator = "Linear"
            else:
                raise ValueError(
                    "Interpolator order not known. Allowed options are: 0, 1")
        if interpolator not in ALLOWED_INTERPOLATORS:
            raise ValueError(
                "Interpolator not known. Allowed options are: %s" % (
                    ", ".join(ALLOWED_INTERPOLATORS)))

        if interpolator == "OrientedGaussian":
            cov = self._get_oriented_psf_covariance(
                fixed_itk, moving_itk, spacing)
            interpolator_itk = itk.OrientedGaussianInterpolateImageFunction[
                type(fixed_itk), pixel_type].New()
            interpolator_itk.SetCovariance(cov.flatten())
            interpolator_itk.SetAlpha(alpha_cut)
        else:
            interpolator_itk = getattr(
                itk, "%sInterpolateImageFunction" % interpolator)[
                type(fixed_itk), pixel_type].New()
        return interpolator_itk

    def _get_oriented_psf_covariance(self, fixed_itk, moving_itk, spacing):

        # Fixed axis-aligned covariance matrix representing the PSF
        cov = self._get_psf_covariance(spacing)

        # Express fixed axis-aligned PSF in moving space coordinates
        fixed_direction = sitkh.get_sitk_from_itk_direction(
            fixed_itk.GetDirection())
        moving_direction = sitkh.get_sitk_from_itk_direction(
            moving_itk.GetDirection())
        U = self._get_rotation_matrix(fixed_direction, moving_direction)
        cov = U.dot(cov).dot(U.transpose())

        return cov

    ##
    # Compute (axis aligned) covariance matrix from spacing. The PSF is
    # modelled as Gaussian with
    #  *- FWHM = 1.2*in-plane-resolution (in-plane)
    #  *- FWHM = slice thickness (through-plane)
    # \date       2017-11-01 16:16:36+0000
    #
    # \param      spacing  3D array containing in-plane and through-plane
    #                      dimensions
    #
    # \return     (axis aligned) covariance matrix representing PSF modelled
    #             Gaussian as 3x3 np.array
    #
    @staticmethod
    def _get_psf_covariance(spacing):
        sigma2 = np.zeros_like(np.array(spacing))

        # Compute Gaussian to approximate in-plane PSF:
        sigma2[0:2] = (1.2 * spacing[0:2])**2 / (8 * np.log(2))

        # Compute Gaussian to approximate through-plane PSF:
        if sigma2.size == 3:
            sigma2[2] = spacing[2]**2 / (8 * np.log(2))

        return np.diag(sigma2)

    ##
    # Gets the relative rotation matrix to express fixed-axis aligned
    # covariance matrix in coordinates of moving image
    # \date       2016-10-14 16:37:57+0100
    #
    # \param      fixed_direction   fixed image direction
    # \param      moving_direction  moving image direction
    #
    # \return     The relative rotation matrix as 3x3 numpy array
    #
    @staticmethod
    def _get_rotation_matrix(fixed_direction, moving_direction):
        dim = np.sqrt(np.array(fixed_direction).size).astype(np.uint8)
        fixed_direction = np.array(fixed_direction).reshape(dim, dim)
        moving_direction = np.array(moving_direction).reshape(dim, dim)

        return moving_direction.transpose().dot(fixed_direction)

    ##
    # Gets the space resampling properties given an image an optional
    # spacing/grid adjustment desires.
    # \date       2018-05-03 13:04:05-0600
    #
    # \param      image_sitk        Image as sitk.Image or itk.Image object
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

        if not isinstance(image_sitk, (sitk.Image, itk.Image.D3, itk.Image.SS3)):
            raise IOError("Image must be of type sitk.Image or itk.Image")

        # Read input image information:
        spacing_in = np.array(image_sitk.GetSpacing())
        origin_out = np.array(image_sitk.GetOrigin())
        if isinstance(image_sitk, sitk.Image):
            size_in = np.array(image_sitk.GetSize()).astype(int)
            direction_out = np.array(image_sitk.GetDirection())
        else:
            size_in = np.array(image_sitk.GetBufferedRegion().GetSize())
            direction_out = np.array(
                sitkh.get_sitk_from_itk_direction(image_sitk.GetDirection()))
        dim = len(origin_out)

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

        size, origin, spacing, direction = Resampler.get_space_resampling_properties(
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
