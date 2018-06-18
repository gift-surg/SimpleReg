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


def compose_transforms(transform_outer, transform_inner):

    if not isinstance(transform_outer, sitk.DisplacementFieldTransform) \
            and not isinstance(transform_outer, sitk.Transform):
        raise IOError("Outer transform must be of type sitk.Transform or "
                      "sitk.DisplacementFieldTransform")
    if not isinstance(transform_inner, sitk.DisplacementFieldTransform) \
            and not isinstance(transform_inner, sitk.Transform):
        raise IOError("Inner transform must be of type sitk.Transform or "
                      "sitk.DisplacementFieldTransform")

    # Compose affine transforms
    if isinstance(transform_outer, sitk.Transform) \
            and isinstance(transform_inner, sitk.Transform):
        transform = compose_affine_transforms(transform_outer, transform_inner)

    # Compose displacement fields if at least one transform is a disp field.
    else:
        # Convert sitk.Transform to displacement field if necessary
        if isinstance(transform_outer, sitk.Transform):
            displacement_sitk = sitk.TransformToDisplacementField(
                transform_outer)
            transform_outer = sitk.DisplacementFieldTransform(
                sitk.Image(displacement_sitk))
        if isinstance(transform_inner, sitk.Transform):
            displacement_sitk = sitk.TransformToDisplacementField(
                transform_inner)
            transform_inner = sitk.DisplacementFieldTransform(
                sitk.Image(displacement_sitk))

        transform = compose_displacement_field_transforms(
            transform_outer, transform_inner)

    return transform


def compose_displacement_field_transforms(transform_outer, transform_inner):
    if not isinstance(transform_outer, sitk.DisplacementFieldTransform) \
            or not isinstance(transform_inner, sitk.DisplacementFieldTransform):
        raise IOError("Transforms must be of type "
                      "sitk.TransDisplacementFieldTransformform")

    raise RuntimeError(
        "Composition of displacement fields not implemented yet")

    # Throws error
    # transform_outer.AddTransform(transform_inner)


def compose_affine_transforms(transform_outer, transform_inner):
    if not isinstance(transform_outer, sitk.Transform) \
            or not isinstance(transform_inner, sitk.Transform):
        raise IOError("Transforms must be of type sitk.Transform")

    dim = transform_inner.GetDimension()
    if dim != transform_outer.GetDimension():
        raise IOError("Transform dimensions must match")

    A_inner = np.asarray(transform_inner.GetMatrix()).reshape(dim, dim)
    c_inner = np.asarray(transform_inner.GetCenter())
    t_inner = np.asarray(transform_inner.GetTranslation())

    A_outer = np.asarray(transform_outer.GetMatrix()).reshape(dim, dim)
    c_outer = np.asarray(transform_outer.GetCenter())
    t_outer = np.asarray(transform_outer.GetTranslation())

    A_composite = A_outer.dot(A_inner)
    c_composite = c_inner
    t_composite = A_outer.dot(
        t_inner + c_inner - c_outer) + t_outer + c_outer - c_inner

    if transform_outer.GetName() == transform_inner.GetName():
        if transform_inner.GetName() == "AffineTransform":
            transform = sitk.AffineTransform(dim)
        else:
            transform = getattr(sitk, transform_inner.GetName())()
    else:
        transform = sitk.AffineTransform(dim)

    transform.SetMatrix(A_composite.flatten())
    transform.SetTranslation(t_composite)
    transform.SetCenter(c_composite)

    return transform
