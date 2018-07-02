#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import simplereg.data_reader as dr
import simplereg.data_writer as dw
import simplereg.utilities as utils
import simplereg.landmark_estimator as le
import simplereg.landmark_visualizer as lv
from simplereg.niftyreg_to_simpleitk_converter import \
    NiftyRegToSimpleItkConverter as nreg2sitk
from simplereg.flirt_to_simpleitk_converter import \
    FlirtToSimpleItkConverter as flirt2sitk
from simplereg.landmark_visualizer import IMPLEMENTED_MARKERS


##
# Tools for various transformation and conversion tasks
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    # Read input
    parser = argparse.ArgumentParser(
        description="Tools for various transformation and conversion tasks.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-i", "--image-header",
        help="Update image header given a (Simple)ITK transform",
        nargs=3,
        metavar=("IMAGE", "TRANSFORM", "OUTPUT_IMAGE"),
        default=None,
    )
    parser.add_argument(
        "-c", "--compose",
        help="Compose two (Simple)ITK transformations T2 and T1 to output "
        "transform T3(x) = T2(T1)(x)",
        nargs=3,
        metavar=("TRANSFORM_2", "TRANSFORM_1", "OUTPUT_TRANSFORM"),
        default=None,
    )
    parser.add_argument(
        "-d", "--datatype",
        help="Transform datatype of image (or displacement field) data array. "
        "Numpy datatypes are recognized, e.g. float32, float64 and uint8.",
        nargs=3,
        metavar=("IMAGE", "DATATYPE", "OUTPUT_IMAGE"),
        default=None,
    )
    parser.add_argument(
        "-inv", "--invert-transform",
        help="Invert a (Simple)ITK transform",
        nargs=2,
        metavar=("TRANSFORM", "OUTPUT_TRANSFORM"),
        default=None,
    )
    parser.add_argument(
        "-l", "--landmark",
        help="Apply (Simple)ITK transform (or displacement field) to landmarks. "
        "Landmarks are encoded in a text file with one landmark position in "
        "mm per line j: "
        "<key_j_x> <key_j_y> (<key_j_z>)",
        nargs=3,
        metavar=("LANDMARKS", "TRANSFORM", "OUTPUT_LANDMARKS"),
        default=None,
    )
    parser.add_argument(
        "-sitk2nreg", "--sitk-to-nreg",
        help="Convert (Simple)ITK to NiftyReg transform representation. "
        "This includes both affine transformations and displacement fields.",
        metavar=("SIMPLEITK", "OUTPUT_NIFTYREG"),
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-nreg2sitk", "--nreg-to-sitk",
        help="Convert NiftyReg to (Simple)ITK transform representation. "
        "This includes conversions from both "
        "affine transformations (reg_aladin) "
        "and displacement fields (reg_f3d).",
        metavar=("NIFTYREG", "OUTPUT_SIMPLEITK"),
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-flirt2sitk", "--flirt-to-sitk",
        help="Convert FLIRT to (Simple)ITK transform representation "
        "(3D only). "
        "Fixed and moving images need to be provided. ",
        metavar=("FLIRT", "FIXED", "MOVING", "OUTPUT_SIMPLEITK"),
        type=str,
        nargs=4,
        default=None,
    )
    parser.add_argument(
        "-sitk2flirt", "--sitk-to-flirt",
        help="Convert (Simple)ITK to FLIRT transform representation "
        "(3D only). "
        "Fixed and moving images need to be provided.",
        metavar=("SIMPLEITK", "FIXED", "MOVING", "OUTPUT_FLIRT"),
        type=str,
        nargs=4,
        default=None,
    )
    parser.add_argument(
        "-sitk2nii", "--swap-sitk-nii",
        help="Swap representation of points/landmarks between (Simple)ITK and "
        "NIfTI coordinate systems (x maps_to -x, y maps_to -y in ITK). "
        "In particular, NiftyReg uses NIfTI representation.",
        metavar=("SITK/NII", "OUTPUT_NII/SITK"),
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-split", "--split-labels",
        help="Split multi-label mask into 4D (or 5D) image where each "
        "time point corresponds to an independent mask label",
        metavar=("LABELS", "DIM", "OUTPUT_LABELS"),
        type=str,
        nargs=3,
        default=None,
    )
    parser.add_argument(
        "-label2land", "--label-to-landmark",
        help="Compute landmarks representing the centroids of each label. "
        "If a binary mask is provided centroids are computed for each "
        "connected region.",
        metavar=("LABEL", "OUTPUT_LANDMARKS"),
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-land2label", "--landmark-to-label",
        help="Convert landmark coordinates to image label where "
        "each landmark corresponds to a different label. "
        "An image needs to be provided to define the image space.",
        metavar=("LANDMARKS", "IMAGE", "OUTPUT_LABEL"),
        type=str,
        nargs=3,
        default=None,
    )
    parser.add_argument(
        "-land2image", "--landmark-to-image",
        help="Embed landmarks into image",
        metavar=("LANDMARKS", "IMAGE", "OUTPUT_IMAGE"),
        type=str,
        nargs=3,
        default=None,
    )
    parser.add_argument(
        "-label2bound", "--label-to-boundary",
        help="Convert labels to boundaries using binary erosion",
        metavar=("LABEL", "OUTPUT_BOUNDARY"),
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "-m", "--marker",
        help="Marker used to visualize landmark positions. "
        "Only used for --landmark-to-label and --landmark-to-image. "
        "Allowed markers are: %s" % ", ".join(IMPLEMENTED_MARKERS),
        type=str,
        default="cross",
    )
    parser.add_argument(
        "-r", "--radius",
        help="Radius of marker used to visualize landmark positions. "
        "Only used for --landmark-to-label and --landmark-to-image. "
        "If marker is 'dot', this setting is ignored.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Turn on/off verbose output",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    if args.image_header is not None:
        image_sitk = dr.DataReader.read_image(args.image_header[0])
        transform_sitk = dr.DataReader.read_transform(args.image_header[1])
        transformed_image_sitk = utils.update_image_header(
            image_sitk, transform_sitk)
        dw.DataWriter.write_image(
            transformed_image_sitk, args.image_header[2], args.verbose)

    if args.compose is not None:
        transform1_sitk = dr.DataReader.read_transform(args.compose[1])
        transform2_sitk = dr.DataReader.read_transform(args.compose[0])
        transform_sitk = utils.compose_transforms(
            transform2_sitk, transform1_sitk)
        dw.DataWriter.write_transform(
            transform_sitk, args.compose[2], args.verbose)

    if args.datatype is not None:
        image_nib = dr.DataReader.read_transform(
            args.datatype[0], nii_as_nib=1)
        image_nib.header.set_data_dtype(getattr(np, args.datatype[1]))
        dw.DataWriter.write_transform(image_nib, args.datatype[2])

    if args.invert_transform is not None:
        transform_inv_sitk = dr.DataReader.read_transform(
            args.invert_transform[0], inverse=1)
        dw.DataWriter.write_transform(
            transform_inv_sitk, args.invert_transform[1], args.verbose)

    if args.landmark is not None:
        landmarks_nda = dr.DataReader.read_landmarks(args.landmark[0])
        transform_sitk = dr.DataReader.read_transform(args.landmark[1])
        for i in range(landmarks_nda.shape[0]):
            landmarks_nda[i, :] = transform_sitk.TransformPoint(
                landmarks_nda[i, :])
        dw.DataWriter.write_landmarks(
            landmarks_nda, args.landmark[2], args.verbose)

    if args.sitk_to_nreg is not None:
        nreg2sitk.convert_sitk_to_nreg_transform(
            args.sitk_to_nreg[0],
            args.sitk_to_nreg[1],
            args.verbose,
        )

    if args.nreg_to_sitk is not None:
        nreg2sitk.convert_nreg_to_sitk_transform(
            args.nreg_to_sitk[0],
            args.nreg_to_sitk[1],
            args.verbose,
        )

    if args.flirt_to_sitk is not None:
        flirt2sitk.convert_flirt_to_sitk_transform(
            args.flirt_to_sitk[0],
            args.flirt_to_sitk[1],
            args.flirt_to_sitk[2],
            args.flirt_to_sitk[3],
        )

    if args.sitk_to_flirt is not None:
        flirt2sitk.convert_sitk_to_flirt_transform(
            args.sitk_to_flirt[0],
            args.sitk_to_flirt[1],
            args.sitk_to_flirt[2],
            args.sitk_to_flirt[3],
        )

    if args.swap_sitk_nii is not None:
        landmarks_nda = dr.DataReader.read_landmarks(args.swap_sitk_nii[0])
        landmarks_nda[:, 0:2] *= -1
        dw.DataWriter.write_landmarks(
            landmarks_nda, args.swap_sitk_nii[1], args.verbose)

    if args.split_labels is not None:
        dim = int(args.split_labels[1])
        if dim != 4 and dim != 5:
            raise IOError("Output dimension can only be either 4 or 5")
        utils.split_labels(args.split_labels[0], dim, args.split_labels[2])

    if args.label_to_landmark is not None:
        landmark_estimator = le.LandmarkEstimator(
            path_to_image_label=args.label_to_landmark[0],
            verbose=args.verbose,
        )
        landmark_estimator.run()
        landmarks_nda = landmark_estimator.get_landmarks()
        dw.DataWriter.write_landmarks(
            landmarks_nda, args.label_to_landmark[1], args.verbose)

    if args.landmark_to_label:
        landmarks_nda = dr.DataReader.read_landmarks(args.landmark_to_label[0])
        image_sitk = dr.DataReader.read_image(args.landmark_to_label[1])
        landmark_visualizer = lv.LandmarkVisualizer(
            landmarks_nda=landmarks_nda,
            direction=image_sitk.GetDirection(),
            origin=image_sitk.GetOrigin(),
            spacing=image_sitk.GetSpacing(),
            size=image_sitk.GetSize()
        )
        landmark_visualizer.build_landmark_image_sitk(
            marker=args.marker, radius=args.radius)
        mask_sitk = landmark_visualizer.get_image_sitk()
        dw.DataWriter.write_image(mask_sitk, args.landmark_to_label[2])

    if args.landmark_to_image:
        landmarks_nda = dr.DataReader.read_landmarks(args.landmark_to_image[0])
        image_sitk = dr.DataReader.read_image(args.landmark_to_image[1])
        landmark_visualizer = lv.LandmarkVisualizer(
            landmarks_nda=landmarks_nda,
            direction=image_sitk.GetDirection(),
            origin=image_sitk.GetOrigin(),
            spacing=image_sitk.GetSpacing(),
            size=image_sitk.GetSize()
        )
        landmark_visualizer.build_landmark_image_sitk(
            marker=args.marker, radius=args.radius)
        image_sitk = landmark_visualizer.annotate_landmarks_on_image_sitk(
            image_sitk)
        dw.DataWriter.write_image(image_sitk, args.landmark_to_image[2])

    if args.label_to_boundary:
        utils.convert_label_to_boundary(
            args.label_to_boundary[0], args.label_to_boundary[1])

    return 0


if __name__ == '__main__':
    main()
