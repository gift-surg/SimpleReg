##
# \file landmark_estimator.py
# \brief      Class to estimate landmarks from fiducial segmentations
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2018
#

import os
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import skimage.measure

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Class to estimate landmarks from fiducial segmentations
# \date       2018-04-24 15:42:18-0600
#
class LandmarkEstimator(object):

    def __init__(self, path_to_image_label, verbose=0):
        self._path_to_image_label = path_to_image_label
        self._verbose = verbose

        self._landmarks_image_space = None
        self._landmarks_voxel_space = None

    def get_landmarks(self):
        return np.array(self._landmarks_image_space)

    def save_landmarks_to_image(self, path_to_file):

        if self._landmarks_image_space is None:
            raise RuntimeError("Execute 'run' first to estimate landmarks")

        ph.print_info("Save landmarks to image '%s' ... " % path_to_file,
                      newline=False)

        # read original image
        image_label_sitk = sitk.ReadImage(self._path_to_image_label)
        image_label_nda = sitk.GetArrayFromImage(image_label_sitk) * 0

        # convert to integer voxels
        image_label_nda = self._get_array_with_landmarks(
            image_label_sitk.GetSize()[::-1], self._landmarks_voxel_space)
        # landmarks_voxel_space = self._landmarks_voxel_space.astype('int')

        # for i in range(landmarks_voxel_space.shape[0]):
        #     image_label_nda[landmarks_voxel_space[i, 2],
        #                    landmarks_voxel_space[i, 1],
        #                    landmarks_voxel_space[i, 0]] = 1

        image_landmarks_sitk = sitk.GetImageFromArray(image_label_nda)
        image_landmarks_sitk.CopyInformation(image_label_sitk)

        sitkh.write_nifti_image_sitk(image_landmarks_sitk, path_to_file)
        print("done")

        # show landmark estimate
        if self._verbose:
            # find bounding box for "zoomed in" visualization
            ran_x, ran_y, ran_z = self._get_bounding_box(image_label_nda)

            # get zoomed-in image mask
            image_label_nda_show = image_label_nda[
                ran_x[0]: ran_x[1], ran_y[0]: ran_y[1], ran_z[0]: ran_z[1]]
            landmarks_nda = self._get_array_with_landmarks(
                image_label_nda.shape, self._landmarks_voxel_space)
            show_mask_sitk = sitk.GetImageFromArray(image_label_nda_show)

            # get zoomed-in landmark estimate (dilated for visualization)
            landmarks_nda_show = landmarks_nda[
                ran_x[0]: ran_x[1], ran_y[0]: ran_y[1], ran_z[0]: ran_z[1]]
            landmarks_nda_show += scipy.ndimage.morphology.binary_dilation(
                landmarks_nda_show, iterations=10)
            show_landmarks_sitk = sitk.GetImageFromArray(landmarks_nda_show)

            sitkh.show_sitk_image(
                show_mask_sitk, segmentation=show_landmarks_sitk,
                label=os.path.basename(
                    ph.strip_filename_extension(self._path_to_image_label)[0]))

    def run(self):

        # convert image to data array
        image_label_sitk = sitk.ReadImage(self._path_to_image_label)
        image_label_nda = sitk.GetArrayFromImage(
            image_label_sitk).astype(np.uint8)

        # if binary mask separate into connected regions
        if image_label_nda.max() == 1:
            labels_nda = skimage.measure.label(image_label_nda)
        else:
            labels_nda = image_label_nda

        n_landmarks = labels_nda.max()

        # get landmark coordinates in (continuous) voxel space
        self._landmarks_voxel_space = np.zeros(
            (n_landmarks, image_label_sitk.GetDimension()))
        for i in range(n_landmarks):
            points = np.array(np.where(labels_nda == i + 1))

            # if label not found, set associated landmark coordinates to NaNs
            if points.size == 0:
                self._landmarks_voxel_space[i, :] = np.nan

            else:
                self._landmarks_voxel_space[i, :] = np.mean(points, axis=1)

        # sitk -> nda stores as z, y, x
        self._landmarks_voxel_space = self._landmarks_voxel_space[:, ::-1]

        # get landmark coordinates in image space
        self._landmarks_image_space = np.zeros_like(
            self._landmarks_voxel_space)
        for i in range(n_landmarks):
            self._landmarks_image_space[i, :] = \
                image_label_sitk.TransformContinuousIndexToPhysicalPoint(
                self._landmarks_voxel_space[i, :])

        if self._verbose:
            ph.print_info(
                "Landmarks in voxel space (integer; first index is 0): ")
            print(self._landmarks_voxel_space.astype(int))

            ph.print_info("Landmarks in image space: ")
            print(self._landmarks_image_space)

    ##
    # Return rectangular region surrounding masked region.
    # \date       2018-04-25 15:20:49-0600
    #
    # \param      nda     The nda
    # \param      offset  The offset
    #
    # \return     triple of z-,y-,z-intervals in voxel space
    #
    @staticmethod
    def _get_bounding_box(nda, offset=5):

        # Return in case no masked pixel available
        if np.sum(abs(nda)) == 0:
            return None, None, None

        # Get shape defining the dimension in each direction
        shape = nda.shape

        # Compute sum of pixels of each slice along specified directions
        sum_xy = np.sum(nda, axis=(0, 1))  # sum within x-y-plane
        sum_xz = np.sum(nda, axis=(0, 2))  # sum within x-z-plane
        sum_yz = np.sum(nda, axis=(1, 2))  # sum within y-z-plane

        # Find masked regions (non-zero sum!)
        range_x = np.zeros(2)
        range_y = np.zeros(2)
        range_z = np.zeros(2)

        # Non-zero elements of numpy array nda defining x_range
        ran = np.nonzero(sum_yz)[0]
        range_x[0] = np.max([0, ran[0] - offset])
        range_x[1] = np.min([shape[0], ran[-1] + 1 + offset])

        # Non-zero elements of numpy array nda defining y_range
        ran = np.nonzero(sum_xz)[0]
        range_y[0] = np.max([0, ran[0] - offset])
        range_y[1] = np.min([shape[1], ran[-1] + 1 + offset])

        # Non-zero elements of numpy array nda defining z_range
        ran = np.nonzero(sum_xy)[0]
        range_z[0] = np.max([0, ran[0] - offset])
        range_z[1] = np.min([shape[2], ran[-1] + 1 + offset])

        return range_x.astype(int), range_y.astype(int), range_z.astype(int)

    @staticmethod
    def _get_array_with_landmarks(nda_shape, landmarks_voxel_space):

        # convert to integer voxels
        landmarks_voxel_space = landmarks_voxel_space.astype('int')

        # fill array
        nda = np.zeros(nda_shape, dtype=np.int)
        for i in range(landmarks_voxel_space.shape[0]):
            nda[landmarks_voxel_space[i, 2],
                landmarks_voxel_space[i, 1],
                landmarks_voxel_space[i, 0]] = 1

        return nda
