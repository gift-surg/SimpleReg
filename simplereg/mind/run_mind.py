import os
import argparse
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="MIND test run",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "--filename",
        help="Path to 3D NIfTI image",
        type=str,
        required=1,
    )
    parser.add_argument(
        "--output",
        help="Path to 3D NIfTI vector image holding the MIND features",
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
    args = parser.parse_args()
    start_time = ph.start_timing()

    name = ph.strip_filename_extension(os.path.basename(args.filename))[0]
    image_sitk = sitk.ReadImage(args.filename)

    image_nda = sitk.GetArrayFromImage(image_sitk)

    if image_nda.ndim != 3:
        raise IOError("Image must be 3D")

    print("image_nda.shape = %s " % str(image_nda.shape))
    ph.print_info("Time image_nda: %s" % ph.stop_timing(start_time))

    # https://stackoverflow.com/questions/10997254/converting-numpy-arrays-to-matlab-and-vice-versa
    import matlab.engine
    ph.print_info("Time import matlab.engine: %s" % ph.stop_timing(start_time))

    eng = matlab.engine.start_matlab()
    ph.print_info("Time matlab.engine.start_matlab(): %s" %
                  ph.stop_timing(start_time))

    image_mat = matlab.double(image_nda.tolist())
    ph.print_info("Time image_mat: %s" % ph.stop_timing(start_time))

    # eng.addpath('/home/mebner/Dropbox/UCL/Software/SimpleReg/simplereg/mind/')
    mind_mat = eng.MIND_descriptor(image_mat)
    ph.print_info("Time mind_mat: %s" % ph.stop_timing(start_time))

    # https://www.mathworks.com/matlabcentral/answers/327455-convert-matlab-engine-outputs-to-numpy-arrays
    # mind_nda = np.asarray(mind_mat) # incredibly slow
    mind_nda = np.array(mind_mat._data).reshape(
        mind_mat.size[::-1]).transpose()  # fast
    ph.print_info("Time mind_nda/total: %s" % ph.stop_timing(start_time))

    # images_sitk = [None] * mind_nda.shape[-1]
    # paths_to_mind = [None] * mind_nda.shape[-1]

    mind_sitk = sitk.GetImageFromArray(mind_nda)
    mind_sitk.SetOrigin(image_sitk.GetOrigin())
    mind_sitk.SetSpacing(image_sitk.GetSpacing())
    mind_sitk.SetDirection(image_sitk.GetDirection())

    sitkh.write_nifti_image_sitk(mind_sitk, args.output)

    # # Write extracted MIND features
    # for i in range(mind_nda.shape[-1]):
    #     images_sitk[i] = sitk.GetImageFromArray(
    #         np.squeeze(mind_nda[:, :, :, i]))
    #     images_sitk[i].CopyInformation(image_sitk)
    #     paths_to_mind[i] = os.path.join(
    #         args.dir_output, "%s_MIND-%d.nii.gz" % (name, i + 1))
    #     sitkh.write_nifti_image_sitk(images_sitk[i], paths_to_mind[i])

    # Visualize image and extracted MIND features
    if args.verbose:
        ph.show_niftis([args.filename, args.output])
