#!/usr/bin/env python

import os
import argparse

import pysitk.python_helper as ph


def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Create landmark mask or embed them in an image.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to image (.nii.gz)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to image with annotated landmarks (.nii.gz)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-l", "--landmarks",
        help="Path to landmarks (.txt)",
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

    


    return 0


if __name__ == '__main__':
    main()
