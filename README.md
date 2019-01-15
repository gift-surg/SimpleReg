# SimpleReg 

SimpleReg is a Python-based open-source toolkit that provides tools helpful for (medical) image registration and processing. Interfaces to the following registration frameworks are available:

* [SimpleITK][simpleitk]
* [NiftyReg][niftyreg]
* [FLIRT][fsl]
* [ITK_NiftyMIC][itkniftymic] as an extension to [WrapITK][wrapitk]

The algorithm and software were developed by [Michael Ebner][mebner] at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

If you have any questions or comments, please drop me an email to `michael.ebner.14@ucl.ac.uk`.

## Installation

SimpleReg was developed in

* Mac OS X 10.10 and 10.12
* Ubuntu 14.04 and 16.04

and tested for Python 2.7.12 and 3.5.2.

Install required external tools and libraries by following the
* [Installation Instructions of SimpleReg Dependencies][simplereg-dependencies]

Clone the SimpleReg repository by
* `git clone git@github.com:gift-surg/SimpleReg.git` 

Install all Python-dependencies by 
* `pip install -r requirements.txt`

Install SimpleReg by running
* `pip install .`

Check installation via
* `python -m nose tests/installation_test.py`

## Licensing and Copyright
Copyright (c) 2018, [University College London][ucl].
This framework is made available as free open-source software under the [BSD-3-Clause License][bsd]. Other licenses may apply for dependencies.

## Funding
This work is partially funded by the UCL [Engineering and Physical Sciences Research Council (EPSRC)][epsrc] Centre for Doctoral Training in Medical Imaging (EP/L016478/1), the Innovative Engineering for Health award ([Wellcome Trust][wellcometrust] [WT101957] and [EPSRC][epsrc] [NS/A000027/1]), and supported by researchers at the [National Institute for Health Research][nihr] [University College London Hospitals (UCLH)][uclh] Biomedical Research Centre.

[citation]: https://www.sciencedirect.com/science/article/pii/S1053811917308042
[mebner]: https://www.linkedin.com/in/ebnermichael
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
[itkniftymic]: https://github.com/gift-surg/ITK_NiftyMIC/wikis/home
[niftymic-install]: https://github.com/gift-surg/NiftyMIC/wikis/niftymic-installation
[nsol]: https://github.com/gift-surg/NSoL
[simplereg]: https://github.com/gift-surg/SimpleReg
[simplereg-dependencies]: https://github.com/gift-surg/SimpleReg/wikis/simplereg-dependencies
[pysitk]: https://github.com/gift-surg/PySiTK
[wrapitk]: https://itk.org/Wiki/ITK/WrapITK_Status
[niftyreg]: https://github.com/KCL-BMEIS/niftyreg/wiki
[fsl]: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
[simpleitk]: http://www.simpleitk.org/