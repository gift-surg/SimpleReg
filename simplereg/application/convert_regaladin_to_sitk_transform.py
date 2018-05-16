#!/usr/bin/env python

import os
import argparse
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph

import simplereg.utilities as utils


##
# Apply Convert RegAladin (NiftyReg) transformation to a SimpleITK one
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Convert RegAladin (NiftyReg) transformation to a "
        "SimpleITK one.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to RegAladin (NiftyReg) transformation (.txt)",
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


    matrix = np.loadtxt(args.filename)
    transform_sitk = utils.convert_regaladin_to_sitk_transform(matrix)
    ph.create_directory(os.path.dirname(args.output))
    sitk.WriteTransform(transform_sitk, args.output)

    return 0


if __name__ == '__main__':
    main()
