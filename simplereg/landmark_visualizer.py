##
# \file landmark_visualizer.py
# \brief      Class to create image mask from landmark coordinates. Landmarks
#             can also be embedded in image.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       June 2018
#

import os
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import skimage.measure

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

IMPLEMENTED_MARKERS = ["dot", "cross", "sphere", "hollow_sphere"]


##
# Class to create image mask from landmark coordinates. Landmarks can also be
# embedded in image.
# \date       2018-06-08 16:54:56-0600
#
class LandmarkVisualizer(object):

    def __init__(self, landmarks_nda, direction, origin, spacing, size):
        self._landmarks_nda = landmarks_nda
        self._direction = direction
        self._origin = origin
        self._spacing = np.array(spacing)
        self._size = size

        self._landmark_image_sitk = None

        self._get_marker = {
            "hollow_sphere": self._get_marker_hollow_sphere,
            "sphere": self._get_marker_sphere,
            "cross": self._get_marker_cross,
        }

    def set_landmarks_nda(self, landmarks_nda):
        self._landmarks_nda = landmarks_nda

    def get_image_sitk(self):
        return sitk.Image(self._landmark_image_sitk)

    def get_image_nda(self):
        return np.array(self._landmark_image_nda)

    def build_landmark_image_sitk(self, marker="cross", radius=2):

        if marker not in IMPLEMENTED_MARKERS:
            raise ValueError("Marker not recognized. "
                             "Allowed options are: %s" % ", ".join(
                                 IMPLEMENTED_MARKERS))

        nda = np.zeros(self._size[::-1], dtype=np.uint8)

        foo = sitk.GetImageFromArray(nda)
        foo.SetSpacing(self._spacing)
        foo.SetDirection(self._direction)
        foo.SetOrigin(self._origin)

        for i in range(self._landmarks_nda.shape[0]):
            landmark = self._landmarks_nda[i, :]
            index = foo.TransformPhysicalPointToIndex(landmark)[::-1]

            if True in np.isnan(landmark):
                continue

            if marker == "dot":
                nda[index] = i + 1

            else:
                marker_ = self._get_marker[marker](
                    radius=radius,
                    spacing=self._spacing,
                )
                nda = self._apply_marker(nda, index, marker_, value=i + 1)

        self._landmark_image_sitk = sitk.GetImageFromArray(nda)
        self._landmark_image_sitk.SetSpacing(self._spacing)
        self._landmark_image_sitk.SetDirection(self._direction)
        self._landmark_image_sitk.SetOrigin(self._origin)

        self._landmark_image_nda = nda

    def annotate_landmarks_on_image_sitk(self, image_sitk, value=0):
        image_nda = sitk.GetArrayFromImage(image_sitk)
        indices = np.where(self._landmark_image_nda > 0)
        image_nda[indices] = value

        image_landmarks_sitk = sitk.GetImageFromArray(image_nda)
        image_landmarks_sitk.CopyInformation(image_sitk)

        return sitk.Image(image_landmarks_sitk)

    @staticmethod
    def _get_marker_hollow_sphere(radius, spacing=np.ones(3), thickness=0.5):
        a = radius + np.ceil(thickness)
        x = np.linspace(-a, a, 2 * a + 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        marker = np.zeros_like(xx)
        values = \
            (xx / spacing[0])**2 + \
            (yy / spacing[1])**2 + \
            (zz / spacing[2])**2
        marker[values <= (radius + thickness)**2] = 1
        marker[values < (radius - thickness)**2] = 0
        return marker

    @staticmethod
    def _get_marker_sphere(radius, spacing=np.ones(3), thickness=0):
        a = radius + np.ceil(thickness)
        x = np.linspace(-a, a, 2 * a + 1)
        xx, yy, zz = np.meshgrid(x, x, x)
        marker = np.zeros_like(xx)
        values = \
            (xx / spacing[0])**2 + \
            (yy / spacing[1])**2 + \
            (zz / spacing[2])**2
        marker[values <= (radius + thickness)**2] = 1
        return marker

    @staticmethod
    def _get_marker_cross(radius, spacing=np.ones(3), thickness=None):
        a = [int(np.round(radius / s)) for s in spacing[::-1]]
        x = np.arange(2 * a[0] + 1)
        y = np.arange(2 * a[1] + 1)
        z = np.arange(2 * a[2] + 1)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        marker = np.zeros_like(xx)
        marker[:, a[1], a[2]] = 1
        marker[a[0], :, a[2]] = 1
        marker[a[0], a[1], :] = 1
        return marker

    @staticmethod
    def _apply_marker(nda, index, marker, value):
        a = int((marker.shape[0] - 1) / 2)
        index = np.array(index)

        indices = np.array(np.where(marker == 1)) - a
        indices = tuple(indices + index[..., np.newaxis])
        try:
            nda[indices] = value
        except:
            pass

        return nda
