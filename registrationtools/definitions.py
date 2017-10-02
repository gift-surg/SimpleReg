import os
import sys

from pythonhelper.definitions import DIR_TMP
# from pythonhelper.definitions import ITKSNAP_EXE
# from pythonhelper.definitions import FSLVIEW_EXE
# from pythonhelper.definitions import NIFTYVIEW_EXE

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_TEST = os.path.join(DIR_ROOT, "data")

FLIRT_EXE = "flirt"
C3D_AFFINE_TOOL_EXE = "c3d_affine_tool"  # for FLIRT parameter conversion

REG_ALADIN_EXE = "reg_aladin"
REG_F3D_EXE = "reg_f3d"
REG_RESAMPLE_EXE = "reg_resample"
REG_TRANSFORM_EXE = "reg_transform"
