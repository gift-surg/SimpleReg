import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.definitions import DIR_DATA

if __name__ == '__main__':
    start_time = ph.start_timing()

    path_to_nii = os.path.join(DIR_DATA,'3D_Brain_Source.nii.gz')

    image_sitk = sitk.ReadImage(path_to_nii)
    image_nda = sitk.GetArrayFromImage(image_sitk)
    print("image_nda.shape = %s " % str(image_nda.shape))
    ph.print_info("Time image_nda: %s" % ph.stop_timing(start_time))

    # https://stackoverflow.com/questions/10997254/converting-numpy-arrays-to-matlab-and-vice-versa
    import matlab.engine
    ph.print_info("Time import matlab.engine: %s" % ph.stop_timing(start_time))
    
    eng = matlab.engine.start_matlab()
    ph.print_info("Time matlab.engine.start_matlab(): %s" % ph.stop_timing(start_time))
  
    image_mat = matlab.double(image_nda.tolist())
    ph.print_info("Time image_mat: %s" % ph.stop_timing(start_time))
  
    # eng.addpath('/home/mebner/Dropbox/UCL/Software/SimpleReg/simplereg/mind/')
    mind_mat = eng.MIND_descriptor(image_mat)
    ph.print_info("Time mind_mat: %s" % ph.stop_timing(start_time))

    # https://www.mathworks.com/matlabcentral/answers/327455-convert-matlab-engine-outputs-to-numpy-arrays
    # mind_nda = np.asarray(mind_mat) # incredibly slow
    mind_nda = np.array(mind_mat._data).reshape(mind_mat.size[::-1]).transpose() # fast
    ph.print_info("Time mind_nda/total: %s" % ph.stop_timing(start_time))

    images_sitk = [None] * mind_nda.shape[-1]
    for i in range(mind_nda.shape[-1]):
        images_sitk[i] = sitk.GetImageFromArray(mind_nda[:, :, :, i])
        images_sitk[i].CopyInformation(image_sitk)
    images_sitk.insert(0, image_sitk)

    sitkh.show_sitk_image(images_sitk)
