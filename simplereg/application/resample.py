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
        description="Resample image.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "--moving",
        help="Path to moving image",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--fixed",
        help="Path to fixed image",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--output",
        help="Path to resampled image",
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
        "--interpolator",
        help="Interpolator for image resampling",
        type=str,
        required=0,
        default="linear",
    )
    parser.add_argument(
        "--padding",
        help="Padding value",
        type=int,
        required=0,
        default=0,
    )
    parser.add_argument(
        "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    args = parser.parse_args()

    # read input
    fixed_sitk = sitk.ReadImage(args.fixed)
    moving_sitk = sitk.ReadImage(args.moving)
    if args.transform:
        transform_sitk = sitkh.read_transform_sitk(args.transform)
    else:
        transform_sitk = getattr(
            sitk, "Euler%dDTransform" % fixed_sitk.GetDimension())()

    # resample image
    warped_moving_sitk = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        transform_sitk,
        getattr(sitk, "sitk%s" % args.interpolator.title()),
        args.padding,
        fixed_sitk.GetPixelIDValue(),
    )

    sitkh.write_nifti_image_sitk(
        warped_moving_sitk, args.output, verbose=args.verbose)

    return 0


if __name__ == '__main__':
    main()
