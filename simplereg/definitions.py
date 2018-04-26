import os

from pysitk.definitions import DIR_TMP

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_DATA = os.path.join(DIR_ROOT, "data")
DIR_TEST = os.path.join(DIR_DATA, "test")

# OMP threads used for NiftyReg by default
OMP = 8
