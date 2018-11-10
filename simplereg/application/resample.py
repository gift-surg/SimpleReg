#!/usr/bin/env python

import argparse

import pysitk.python_helper as ph

import simplereg.resampler
from simplereg.definitions import ALLOWED_INTERPOLATORS


##
# Resample image
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
        "-m", "--moving",
        help="Path to moving image",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-f", "--fixed",
        help="Path to fixed image. "
        "Can be 'same' if fixed image space is identical to the moving image "
        "space.",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to resampled image",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-t", "--transform",
        help="Path to (SimpleITK) transformation (.txt) or displacement "
        "field (.nii.gz) to be applied",
        type=str,
        required=0,
    )
    parser.add_argument(
        "-i", "--interpolator",
        help="Interpolator for image resampling. Can be either name (%s) "
        "or order (0, 1)" % (
            ", ".join(ALLOWED_INTERPOLATORS)),
        type=str,
        required=0,
        default="Linear",
        # default="BSpline",  # might cause problems for some images
    )
    parser.add_argument(
        "-p", "--padding",
        help="Padding value",
        type=int,
        required=0,
        default=0,
    )
    parser.add_argument(
        "-s", "--spacing",
        help="Set spacing for resampling grid in fixed image space",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-atg", "--add-to-grid",
        help="Additional grid extension/reduction in each direction of each "
        "axis in millimeter. If scalar, changes are applied uniformly to grid",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    args = parser.parse_args()

    if args.fixed == "same":
        args.fixed = args.moving

    resampler = simplereg.resampler.Resampler(
        path_to_fixed=args.fixed,
        path_to_moving=args.moving,
        path_to_transform=args.transform,
        interpolator=args.interpolator,
        spacing=args.spacing,
        padding=args.padding,
        add_to_grid=args.add_to_grid,
        verbose=args.verbose,
    )
    resampler.run()
    resampler.write_image(args.output)

    if args.verbose:
        ph.show_niftis([
            args.output,
            args.fixed,
        ])

    return 0


if __name__ == '__main__':
    main()
