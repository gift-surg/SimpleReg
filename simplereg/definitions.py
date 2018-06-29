import os

from pysitk.definitions import DIR_TMP

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_DATA = os.path.join(DIR_ROOT, "data")
DIR_TEST = os.path.join(DIR_DATA, "tests")

# OMP threads used for NiftyReg by default
OMP = 8

ALLOWED_IMAGES = ["nii.gz", "nii"]
ALLOWED_TRANSFORMS = ["txt"]
ALLOWED_TRANSFORMS_DISPLACEMENTS = ["nii.gz", "nii"]
ALLOWED_LANDMARKS = ["txt"]
ALLOWED_INTERPOLATORS = [
    "Linear",
    "NearestNeighbor",
    "BSpline",
    "OrientedGaussian",
]
