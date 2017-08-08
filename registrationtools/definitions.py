import os
import sys

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_TMP = "/tmp/"

FLIRT_EXE = "flirt"
REG_ALADIN_EXE = "reg_aladin"
REG_F3D_EXE = "reg_f3d"
C3D_AFFINE_TOOL_EXE = "c3d_affine_tool"  # for FLIRT parameter conversion
