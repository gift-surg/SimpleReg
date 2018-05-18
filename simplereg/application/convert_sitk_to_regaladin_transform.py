#!/usr/bin/env python

import os
import argparse
import SimpleITK as sitk

import pysitk.python_helper as ph

import simplereg.utilities as utils


##
# Apply Convert SimpleITK transformation to a RegAladin (NiftyReg) one
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Convert SimpleITK transformation to a "
        "RegAladin (NiftyReg) one.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to SimpleITK transformation (.txt)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output transformation (.txt)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    args = parser.parse_args()


    transform_sitk = sitk.AffineTransform(sitk.ReadTransform(args.filename))
    matrix = utils.convert_sitk_to_regaladin_transform(transform_sitk)
    ph.write_array_to_file(args.output, matrix, access_mode="w")


    return 0


if __name__ == '__main__':
    main()
