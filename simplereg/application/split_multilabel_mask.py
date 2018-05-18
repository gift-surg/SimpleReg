#!/usr/bin/env python

import os
import argparse
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import simplereg.utilities as utils


##
# split multi-label mask into 4d image where each timepoint corresponds to
# independent mask
# \date       2018-04-25 11:50:08-0600
#
# \return     exit code
#
def main():

    parser = argparse.ArgumentParser(
        description="Split multi-label mask into 4D image where each "
        "timepoint corresponds to independent mask.",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to multi-label mask",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output 4D image",
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

    labels_sitk = sitk.ReadImage(args.filename)

    nda = sitk.GetArrayFromImage(labels_sitk).astype(np.uint8int8)
    n_labels = int(np.max(nda))

    # split labels into separate components
    shape = nda.shape + (n_labels, )
    nda_4d = np.zeros((shape), dtype=np.uint8)
    for label in range(n_labels):
        indices = np.where(nda == label + 1)
        indices += (label * np.ones(len(indices[0]), dtype=np.uint8),)
        nda_4d[indices] = 1

    # Create sitk.Image object
    labels_4d_sitk = sitk.GetImageFromArray(nda_4d)
    labels_4d_sitk.SetOrigin(labels_sitk.GetOrigin())
    labels_4d_sitk.SetSpacing(labels_sitk.GetSpacing())
    labels_4d_sitk.SetDirection(labels_sitk.GetDirection())

    sitkh.write_nifti_image_sitk(labels_4d_sitk, args.output)

    if args.verbose:
        ph.show_niftis([args.filename, args.output])

    return 0


if __name__ == '__main__':
    main()
