# RegistrationTools 

This software package provides the interface to a collection of registration tools developed in support of various research-focused toolkits within the [GIFT-Surg][giftsurg] project.

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

## Licensing and Copyright
Copyright (c) 2017, [University College London][ucl].
This framework is made available as free open-source software under the [BSD-3-Clause License][bsd]. Other licenses may apply for dependencies.

[citation]: https://www.sciencedirect.com/science/article/pii/S1053811917308042
[mebner]: http://cmictig.cs.ucl.ac.uk/people/phd-students/michael-ebner
[tig]: http://cmictig.cs.ucl.ac.uk
[bsd]: https://opensource.org/licenses/BSD-3-Clause
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[guarantors]: https://guarantorsofbrain.org/
[ucl]: http://www.ucl.ac.uk
[uclh]: http://www.uclh.nhs.uk
[epsrc]: http://www.epsrc.ac.uk
[wellcometrust]: http://www.wellcome.ac.uk
[mssociety]: https://www.mssociety.org.uk/
[nihr]: http://www.nihr.ac.uk/research
[volumetricreconstruction]: https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction
[numericalsolver]: https://cmiclab.cs.ucl.ac.uk/mebner/NumericalSolver
[registrationtools]: https://cmiclab.cs.ucl.ac.uk/mebner/RegistrationTools
[pythonhelper]: https://cmiclab.cs.ucl.ac.uk/mebner/PythonHelper