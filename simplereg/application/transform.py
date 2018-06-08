#!/usr/bin/env python

import os
import argparse
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Apply SimpleITK transform to image or landmarks
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Apply SimpleITK transform to image or landmarks.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to image (.nii.gz) or landmarks (.txt)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output image (.nii.gz) or transformed landmarks (.txt)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-t", "--transform",
        help="Path to (SimpleITK) transformation (.txt) or displacement "
        "field (.nii.gz)",
        type=str,
        required=0,
    )
    parser.add_argument(
        "-tinv", "--transform-inv",
        help="Path to inverse (SimpleITK) transformation to be applied",
        type=str,
        required=0,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    args = parser.parse_args()

    if args.transform is None and args.transform_inv is None:
        raise IOError("Either --transform or --transform-inv must be set")

    type_transform = ph.strip_filename_extension(args.transform)[1]
    type_filename = ph.strip_filename_extension(args.filename)[1]

    if args.transform is not None:
        is_inverse = False
        path_to_transform = args.transform
    else:
        is_inverse = True
        path_to_transform = args.transform_inv

    if type_transform in ["nii", "nii.gz"]:
        is_displacement = True
    else:
        is_displacement = False

    if type_filename in ["nii", "nii.gz"]:
        is_landmarks = False
    else:
        is_landmarks = True

    if is_displacement and not is_landmarks:
        raise IOError(
            "Use simplereg_resample to get warped image from displacement"
            "field.")

    if is_displacement:
        displacement_sitk = sitk.ReadImage(args.transform)
        transform_sitk = sitk.DisplacementFieldTransform(
            sitk.Image(displacement_sitk))
        if is_inverse:
            displacement_sitk = transform_sitk.GetInverseDisplacementField()
            transform_sitk = sitk.DisplacementFieldTransform(
                sitk.Image(displacement_sitk))
    else:
        transform_sitk = sitkh.read_transform_sitk(
            path_to_transform, inverse=is_inverse)

    if is_landmarks:
        landmarks_nda = np.loadtxt(args.filename)
        transformed_landmarks_nda = np.zeros_like(landmarks_nda)
        for i in range(landmarks_nda.shape[0]):
            transformed_landmarks_nda[i, :] = transform_sitk.TransformPoint(
                landmarks_nda[i, :])
        ph.write_array_to_file(
            args.output,
            transformed_landmarks_nda,
            access_mode="w",
            verbose=args.verbose)

    else:
        image_sitk = sitk.ReadImage(args.filename)

        # transform image
        transformed_image_sitk = sitkh.get_transformed_sitk_image(
            image_sitk, transform_sitk)
        sitkh.write_nifti_image_sitk(
            transformed_image_sitk, args.output, verbose=args.verbose)

    return 0


if __name__ == '__main__':
    main()
