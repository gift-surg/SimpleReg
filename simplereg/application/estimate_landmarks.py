#!/usr/bin/env python

import os
import argparse

import pysitk.python_helper as ph

import simplereg.landmark_estimator as le


def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Estimate landmarks from image mask. "
        "K-Means algorithm is used to estimate centroids from "
        "labeled regions. ",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "--filename",
        help="Path to fiducial image mask",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--output",
        help="Path to obtained landmarks (.txt)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    parser.add_argument(
        "--clusters",
        help="Number of clusters, i.e. number of expected landmarks.",
        type=int,
        required=0,
        default=3,
    )
    parser.add_argument(
        "--save-to-image",
        help="Save obtained landmarks to image",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ph.print_subtitle("Estimate landmarks")
    landmark_estimator = le.LandmarkEstimator(
        path_to_image_mask=args.filename,
        n_clusters=args.clusters,
        verbose=args.verbose,
    )
    landmark_estimator.run()
    landmarks = landmark_estimator.get_landmarks()
    ph.write_array_to_file(args.output, landmarks, access_mode="w")

    if args.save_to_image:
        path_to_image = "%s.nii.gz" % \
            ph.strip_filename_extension(args.output)[0]
        landmark_estimator.save_landmarks_to_image(path_to_image)

    return 0


if __name__ == '__main__':
    main()
