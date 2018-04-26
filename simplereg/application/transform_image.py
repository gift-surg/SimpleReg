#!/usr/bin/env python

import os
import argparse
import SimpleITK as sitk

import pysitk.simple_itk_helper as sitkh


##
# Apply SimpleITK transform to image
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Apply SimpleITK transform to image.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "--filename",
        help="Path to filename",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--output",
        help="Path to output",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--transform",
        help="Path to (SimpleITK) transformation to be applied",
        type=str,
        required=0,
    )
    parser.add_argument(
        "--transform-inv",
        help="Path to inverse (SimpleITK) transformation to be applied",
        type=str,
        required=0,
    )
    parser.add_argument(
        "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    args = parser.parse_args()

    if args.transform is None and args.transform_inv is None:
        raise IOError("Either --transform or --transform-inv must be set")

    if args.transform is not None:
        is_inverse = False
    else:
        is_inverse = True

    # read input
    image_sitk = sitk.ReadImage(args.filename)
    transform_sitk = sitkh.read_transform_sitk(
        args.transform, inverse=is_inverse)

    # transform image
    transformed_image_sitk = sitkh.get_transformed_sitk_image(
        image_sitk, transform_sitk)
    sitkh.write_nifti_image_sitk(
        transformed_image_sitk, args.output, verbose=args.verbose)

    return 0


if __name__ == '__main__':
    main()
