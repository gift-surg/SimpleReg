#!/usr/bin/env python

import os
import argparse
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Apply Invert SimpleITK transformation
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Invert SimpleITK transformation.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to (SimpleITK) transformation (.txt)",
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

    transform_sitk = sitkh.read_transform_sitk(args.filename, inverse=True)
    ph.create_directory(os.path.dirname(args.output))
    sitk.WriteTransform(transform_sitk, args.output)

    return 0


if __name__ == '__main__':
    main()
