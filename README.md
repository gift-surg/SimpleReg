# RegistrationTools 

This software package provides the interface to a collection of registration tools developed in support of various research-focused toolkits within the [GIFT-Surg](http://www.gift-surg.ac.uk/) project.

Incorporated registration methods are
* [SimpleITK](http://www.simpleitk.org/)
* [(Wrap)ITK](https://itk.org/)
* [NiftyReg](http://cmictig.cs.ucl.ac.uk/component/content/article/software/niftyreg?Itemid=145)
* [FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

If you have any questions or comments (or find bugs), please drop me an email to `michael.ebner.14@ucl.ac.uk`.

## Installation

Required dependencies can be installed using `pip` by running
* `pip install -r requirements.txt`
* `pip install -e .`

In addition, you will need to install `itk` for Python. In case you want to make use of the [Volumetric MRI Reconstruction from Motion Corrupted 2D Slices](https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction) tool or any of its dependencies, please install the ITK version as described there. Otherwise, simply run
* `pip install itk`

In order to run the provided unit tests, please execute
* `python test/runTests.py`

## License
This framework is licensed under the [MIT license ![MIT](https://raw.githubusercontent.com/legacy-icons/license-icons/master/dist/32x32/mit.png)](http://opensource.org/licenses/MIT)
