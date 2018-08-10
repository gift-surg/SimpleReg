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
import scipy.ndimage.morphology

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import simplereg.data_writer as dw
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


##
# Convert a label to its boundaries using binary erosion
# \date       2018-07-02 15:42:01-0600
#
# \param      path_to_labels  Path to multi-label mask
# \param      path_to_output  Path to output multi-label boundary mask
# \param      iterations      Number of binary erosion operations
#
def convert_label_to_boundary(path_to_labels, path_to_output, iterations=1):
    labels_sitk = sitk.ReadImage(path_to_labels)
    nda_labels = sitk.GetArrayFromImage(labels_sitk)

    if nda_labels.dtype != 'uint8' and nda_labels.dtype != 'uint16':
        raise ValueError(
            "Label data array must be of type integer. "
            "If you are sure that the provided image is the correct label "
            "you can convert the data type using "
            "simplereg_transform -d path-to-label uint8 path-to-label_out")

    nda_labels_boundary = np.zeros_like(nda_labels)

    for i in range(nda_labels.max()):
        label = i + 1
        nda_mask = np.zeros_like(nda_labels)
        nda_mask[np.where(nda_labels == label)] = 1
        nda_mask_boundary = nda_mask - \
            scipy.ndimage.morphology.binary_erosion(
                nda_mask, iterations=iterations)
        nda_labels_boundary += label * nda_mask_boundary

    labels_boundary_sitk = sitk.GetImageFromArray(nda_labels_boundary)
    labels_boundary_sitk.CopyInformation(labels_sitk)

    dw.DataWriter.write_image(labels_boundary_sitk, path_to_output)


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
