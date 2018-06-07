#!/usr/bin/env python

import os
import argparse

import pysitk.python_helper as ph

import simplereg.landmark_estimator as le


def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Estimate landmarks from image mask. "
        "Centroids define the landmark position within each labeled region.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to fiducial image mask",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to obtained landmarks (.txt)",
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
    parser.add_argument(
        "--save-to-image",
        help="Save obtained landmarks to image (.nii.gz)",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    landmark_estimator = le.LandmarkEstimator(
        path_to_image_mask=args.filename,
        verbose=args.verbose,
    )
    landmark_estimator.run()
    landmarks = landmark_estimator.get_landmarks()
    ph.write_array_to_file(
        args.output, landmarks, access_mode="w", verbose=args.verbose)

    if args.save_to_image is not None:
        landmark_estimator.save_landmarks_to_image(args.save_to_image)

    return 0


if __name__ == '__main__':
    main()
