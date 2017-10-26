# SimpleReg 

SimpleReg is a Python-based open-source toolkit that provides the interface to various registration tools including

* [SimpleITK][simpleitk]
* [NiftyReg][niftyreg]
* [FLIRT][fsl]
* [ITK_NiftyMIC][itkniftymic] as an extension to [WrapITK][wrapitk]

Please not that currently **only Python 2** is supported.

The algorithm and software were developed by [Michael Ebner][mebner] at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

If you have any questions or comments (or find bugs), please drop me an email to `michael.ebner.14@ucl.ac.uk`.

## Installation

Install required external tools and libraries by following the
* [Installation Instructions of SimpleReg Dependencies][simplereg-dependencies]

Clone the SimpleReg repository by
* `git clone git@cmiclab.cs.ucl.ac.uk:GIFT-Surg/SimpleReg.git` 

Install all Python-dependencies by 
* `pip install -r requirements.txt`

Install SimpleReg by running
* `pip install -e .`

Check installation via
* `python -m nose tests/installation_test.py`

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
[itkniftymic]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/ITK_NiftyMIC/wikis/home
[niftymic-install]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/NiftyMIC/wikis/niftymic-installation
[nsol]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/NSoL
[simplereg]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/SimpleReg
[simplereg-dependencies]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/SimpleReg/wikis/simplereg-dependencies
[pysitk]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/PySiTK
[wrapitk]: https://itk.org/Wiki/ITK/WrapITK_Status
[niftyreg]: https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg/wikis/home
[fsl]: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
[simpleitk]: http://www.simpleitk.org/