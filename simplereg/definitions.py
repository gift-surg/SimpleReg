import os
import sys

from pysitk.definitions import DIR_TMP

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_TEST = os.path.join(DIR_ROOT, "data")

REG_ALADIN_EXE = "reg_aladin"
REG_F3D_EXE = "reg_f3d"
REG_RESAMPLE_EXE = "reg_resample"
REG_TRANSFORM_EXE = "reg_transform"
