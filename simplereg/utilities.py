##
# \file utilities.py
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2018
#

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import nipype.interfaces.c3

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import DIR_TMP

##
# Compute fiducial registration error (FRE) between fixed and warped moving
# landmarks
# \date       2018-04-21 22:24:10-0600
#
# \param      reference_nda  Reference landmarks as (N x dim) numpy array where
#                            dim is either 2 or 3
# \param      estimate_nda   Estimate landmarks as (N x dim) numpy array where
#                            dim is either 2 or 3
#
# \return     FRE as scalar value
#


def fiducial_registration_error(reference_nda, estimate_nda):
    if not isinstance(reference_nda, np.ndarray):
        raise IOError("Fixed points must be of type np.array")

    if not isinstance(estimate_nda, np.ndarray):
        raise IOError("Moving points must be of type np.array")

    if reference_nda.shape[1] != 2 and reference_nda.shape[1] != 3:
        raise IOError("Fixed points must be of dimension N x 2 or N x 3")

    if estimate_nda.shape[1] != 2 and estimate_nda.shape[1] != 3:
        raise IOError(
            "Warped moving points must be of dimension N x 2 or N x 3")

    if reference_nda.shape != estimate_nda.shape:
        raise IOError(
            "Dimensions of fixed and warped moving points must be equal")

    N = float(reference_nda.shape[0])
    FRE = np.square(np.sum(np.square(reference_nda - estimate_nda)) / N)

    return FRE


##
# Convert a NiftyReg-RegAladin affine transformation matrix into a SimpleITK
# transform
# \date       2018-04-25 21:40:05-0600
#
# \param      matrix  affine matrix as np.ndarray
#
# \return     Affine transformation of type sitk.AffineTransform
#
def convert_regaladin_to_sitk_transform(matrix):

    if not isinstance(matrix, np.ndarray):
        raise IOError("matrix must be a np.ndarray")

    if matrix.shape != (4, 4):
        raise IOError("matrix array must be of shape (4, 4)")

    if np.sum(np.abs(matrix[-1, :] - np.array([0, 0, 0, 1]))):
        raise IOError("last row of matrix must be [0, 0, 0, 1]")

    # retrieve dimension
    nda_2D = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    if np.sum(np.abs(matrix[2:, 0:] - nda_2D)) < 1e-6:
        dim = 2
    else:
        dim = 3

    # retrieve (dim x dim) matrix and translation
    A = matrix[0:dim, 0:dim]
    t = matrix[0:dim, -1]

    # Convert to SimpleITK physical coordinate system
    R = np.eye(dim)
    R[0, 0] = -1
    R[1, 1] = -1
    A = R.dot(A).dot(R)
    t = R.dot(t)

    # Convert to affine transform
    # (Note, it is not possible to extract a EulerxDTransform even for
    # rigid reg_aladin trafos: 'Error: Attempt to set a Non-Orthogonal matrix';
    # QR decomposition could work, but initial tests showed that np.linalg.qr
    # and scipy.linalg.qr (can) return a matrix R with negative diagonal
    # values)
    transform_sitk = sitk.AffineTransform(A.flatten(), t)

    return transform_sitk


##
# Convert SimpleITK Transform into a NiftyReg-RegAladin affine matrix
# \date       2018-04-25 21:41:08-0600
#
# \param      transform_sitk  transformation as sitk.Transform
#
# \return     NiftyReg-RegAladin transformation as (4 x 4)-np.array
#
def convert_sitk_to_regaladin_transform(transform_sitk):

    if not isinstance(transform_sitk, sitk.Transform):
        raise IOError("Input must be a sitk.Transform")

    dim = transform_sitk.GetDimension()
    A_sitk = np.array(transform_sitk.GetMatrix()).reshape(dim, dim)
    t_sitk = np.array(transform_sitk.GetTranslation())

    # Convert to physical coordinate system
    A = np.eye(4)
    R = np.eye(dim)

    R[0, 0] = -1
    R[1, 1] = -1
    A[0:dim, 0:dim] = R.dot(A_sitk).dot(R)
    A[0:dim, 3] = R.dot(t_sitk)

    return A


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


##
# Gets the space resampling properties given an image an optional spacing/grid
# adjustment desires.
# \date       2018-05-03 13:04:05-0600
#
# \param      image_sitk        Image as sitk.Image object
# \param      spacing           Spacing for resampling space. If scalar,
#                               isotropic resampling grid is assumed
# \param      add_to_grid       Additional grid extension/reduction in each
#                               direction of each axis. If scalar, changes are
#                               applied uniformly to grid
# \param      add_to_grid_unit  Changes to grid size to be understood in either
#                               millimeter ("mm") or voxel units
#
# \return     The space resampling properties: size_out, origin_out,
#             spacing_out, direction_out
#
def get_space_resampling_properties(image_sitk,
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
def get_resampled_image_sitk(
        image_sitk,
        spacing=None,
        interpolator=sitk.sitkLinear,
        padding=0,
        add_to_grid=None,
        add_to_grid_unit="mm"):

    size, origin, spacing, direction = get_space_resampling_properties(
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


##
# Update image header of sitk.Image
# \date       2018-06-09 13:42:20-0600
#
# \param      image_sitk      sitk.Image object
# \param      transform_sitk  sitk.Transform
#
# \return     sitk.Image with updated image header
#
def update_image_header(image_sitk, transform_sitk):
    transformed_image_sitk = sitkh.get_transformed_sitk_image(
        image_sitk, transform_sitk)
    return transformed_image_sitk


##
# Split multi-label mask into 4D (or 5D) image where each time point
# corresponds to an independent mask label
# \date       2018-06-09 13:51:34-0600
#
# \param      path_to_labels  Path to multi-label mask
# \param      dimension       Dimension of output mask. Either 4 or 5.
# \param      path_to_output  Path to 4D/5D output multi-label mask
#
def split_labels(path_to_labels, dimension, path_to_output):
    if dimension == 4:
        labels_nib = nib.load(path_to_labels)
        nda = labels_nib.get_data().astype(np.uint8)
    else:
        labels_sitk = sitk.ReadImage(path_to_labels)
        nda = sitk.GetArrayFromImage(labels_sitk).astype(np.uint8)

    # split labels into separate components
    n_labels = nda.max()
    shape = nda.shape + (n_labels, )
    nda_4d = np.zeros((shape), dtype=np.uint8)
    for label in range(n_labels):
        indices = np.where(nda == label + 1)
        indices += (label * np.ones(len(indices[0]), dtype=np.uint8),)
        nda_4d[indices] = 1

    if dimension == 4:
        labels_4d_nib = nib.Nifti1Image(
            nda_4d, affine=labels_nib.affine, header=labels_nib.header)
        labels_4d_nib.set_data_dtype(np.uint8)
        ph.create_directory(os.path.dirname(path_to_output))
        nib.save(labels_4d_nib, path_to_output)
    else:
        labels_5d_sitk = sitk.GetImageFromArray(nda_4d)
        labels_5d_sitk.SetOrigin(labels_sitk.GetOrigin())
        labels_5d_sitk.SetSpacing(labels_sitk.GetSpacing())
        labels_5d_sitk.SetDirection(labels_sitk.GetDirection())
        sitkh.write_nifti_image_sitk(labels_5d_sitk, path_to_output)


def convert_sitk_to_nib_image(image_sitk):

    # Read
    # nda = sitk.GetArrayFromImage(image_sitk)
    # nda = np.swapaxes(nda, axis1=0, axis2=image_sitk.GetDimension() - 1)

    # nda_nib = np.zeros(shape_nib, dtype=nda_sitk.dtype)

    # DIR_TMP = "/tmp/"
    path_to_file = os.path.join(DIR_TMP, "tmp.nii.gz")
    sitkh.write_nifti_image_sitk(image_sitk, path_to_file)
    # sitk.WriteImage(image_sitk, path_to_file)

    image_nib = nib.load(path_to_file)
    return image_nib


def get_niftyreg_jacobian_determinant_from_displacement_sitk(
        displacement_sitk):

    path_to_disp = os.path.join(DIR_TMP, "disp.nii.gz")
    path_to_jac = os.path.join(DIR_TMP, "disp_jac.nii.gz")

    displacement_nib = convert_sitk_to_nib_image(displacement_sitk)
    displacement_nib.header['intent_p1'] = 1
    nib.save(displacement_nib, path_to_disp)

    # sitkh.write_nifti_image_sitk(displacement_sitk, path_to_disp)
    # cmd_args = ["fslmodhd"]
    # cmd_args.append(path_to_disp)
    # cmd_args.append("intent_p1 1")
    # ph.execute_command(" ".join(cmd_args))

    cmd_args = ["reg_jacobian"]
    cmd_args.append("-trans %s" % path_to_disp)
    cmd_args.append("-jac %s" % path_to_jac)
    ph.execute_command(" ".join(cmd_args))

    det_jac_sitk = sitk.ReadImage(path_to_jac)
    return det_jac_sitk
