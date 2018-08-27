# \file ITK.py
# \brief Basis class for ITK-based registration tools
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017

# Import libraries
import os
import sys
import six
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from simplereg.simple_itk_registration_base \
    import SimpleItkRegistrationBase


class SimpleItkRegistration(SimpleItkRegistrationBase):

    def __init__(
        self,
        fixed_sitk,
        moving_sitk,
        fixed_sitk_mask=None,
        moving_sitk_mask=None,
        registration_type="Rigid",
        metric="Correlation",
        metric_params=None,
        interpolator="Linear",
        # optimizer="ConjugateGradientLineSearch",
        # optimizer_params={
        #     "learningRate": 1,
        #     "numberOfIterations": 100,
        # },
        optimizer="RegularStepGradientDescent",
        optimizer_params={
            "minStep": 1e-6,
            "numberOfIterations": 200,
            "gradientMagnitudeTolerance": 1e-6,
            "learningRate": 1,
        },
        initializer_type=None,
        use_multiresolution_framework=False,
        optimizer_scales="PhysicalShift",
        shrink_factors=[2, 1],
        smoothing_sigmas=[1, 0],
        verbose=1,
    ):

        # Cast to Float64 since, e.g., integer images cannot be registered
        # fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat64)
        # moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat64)

        SimpleItkRegistrationBase.__init__(self,
                                           fixed_sitk=fixed_sitk,
                                           moving_sitk=moving_sitk,
                                           fixed_sitk_mask=fixed_sitk_mask,
                                           moving_sitk_mask=moving_sitk_mask,
                                           )

        self._registration_type = registration_type

        self._metric = metric
        self._metric_params = metric_params

        self._interpolator = interpolator

        self._optimizer = optimizer
        self._optimizer_params = optimizer_params

        self._optimizer_scales = optimizer_scales

        self._initializer_type = initializer_type
        self._shrink_factors = shrink_factors
        self._smoothing_sigmas = smoothing_sigmas

        self._verbose = bool(verbose)
        self._use_multiresolution_framework = bool(
            use_multiresolution_framework)

    def _run(self):

        dimension = self._fixed_sitk.GetDimension()

        registration_method = sitk.ImageRegistrationMethod()

        # Set the initial transform and parameters to optimize
        if self._registration_type == "Rigid":
            # VersorRigid2DTransform does not exist, unfortunately
            if dimension == 2:
                initial_transform = eval(
                    "sitk.Euler%dDTransform()" % (dimension))
            else:
                initial_transform = eval(
                    "sitk.VersorRigid%dDTransform()" % (dimension))

        elif self._registration_type == "Similarity":
            initial_transform = eval(
                "sitk.Similarity%dDTransform()" % (dimension))

        elif self._registration_type == "Affine":
            initial_transform = sitk.AffineTransform(dimension)

        else:
            raise ValueError("Registration type '%s' not known." %
                             (self._registration_type))

        if self._initializer_type is not None:
            if self._initializer_type in ["MOMENTS", "GEOMETRY"]:
                moving_sitk = self._moving_sitk
                initializer_type = self._initializer_type
            elif self._initializer_type == "SelfMOMENTS":
                moving_sitk = self._fixed_sitk
                initializer_type = "MOMENTS"
            elif self._initializer_type == "SelfGEOMETRY":
                moving_sitk = self._fixed_sitk
                initializer_type = "GEOMETRY"
            else:
                raise ValueError("Initializer type '%s' unknown"
                                 % (self._initializer_type))
            initial_transform = sitk.CenteredTransformInitializer(
                self._fixed_sitk,
                moving_sitk,
                initial_transform,
                eval("sitk.CenteredTransformInitializerFilter.%s" % (
                    initializer_type))
            )
        if self._metric_params is not None:
            if type(self._metric_params) is not dict:
                raise IOError("metric_params must be of type dict")
            else:
                if not self._metric_params:
                    raise IOError("metric_params cannot be an empty dict")

        registration_method.SetInitialTransform(
            initial_transform, inPlace=True)

        if self._moving_sitk_mask is not None:
            # Recasting avoids problems which can occur for some images
            registration_method.SetMetricMovingMask(
                sitk.Cast(self._moving_sitk_mask,
                          self._fixed_sitk_mask.GetPixelIDValue())
            )

        if self._fixed_sitk_mask is not None:
            registration_method.SetMetricFixedMask(self._fixed_sitk_mask)

        # Set interpolator
        eval("registration_method.SetInterpolator(sitk.sitk%s)" %
             (self._interpolator))

        # Set similarity metric
        if self._metric_params is None:
            eval("registration_method.SetMetricAs%s()" % (self._metric))
        else:
            eval("registration_method.SetMetricAs%s" %
                 (self._metric))(**self._metric_params)

        # Set Optimizer
        # if self._optimizer_params is None:
        #     eval("registration_method.SetOptimizerAs" + self._optimizer)()
        # else:
        eval("registration_method.SetOptimizerAs%s" %
             (self._optimizer))(**self._optimizer_params)

        # Set the optimizer to sample the metric at regular steps
        # registration_method.SetOptimizerAsExhaustive(numberOfSteps=50,
        # stepLength=1.0)

        # Estimating scales of transform parameters a step sizes, from the
        # maximum voxel shift in physical space caused by a parameter change
        eval("registration_method.SetOptimizerScalesFrom%s" %
             (self._optimizer_scales))()

        # Optional multi-resolution framework
        if self._use_multiresolution_framework:
            # Set the shrink factors for each level where each level has the
            # same shrink factor for each dimension
            registration_method.SetShrinkFactorsPerLevel(
                shrinkFactors=self._shrink_factors)

            # Set the sigmas of Gaussian used for smoothing at each level
            registration_method.SetSmoothingSigmasPerLevel(
                smoothingSigmas=self._smoothing_sigmas)

            # Enable the smoothing sigmas for each level in physical units
            # (default) or in terms of voxels (then *UnitsOff instead)
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # registration_method.DebugOn()
        if self._verbose:
            self._print_info_text()
        try:
            registration_transform_sitk = registration_method.Execute(
                self._fixed_sitk, self._moving_sitk)

        except RuntimeError as err:
            raise RuntimeError(err.message)
            # Debug:
            # sitkh.show_sitk_image(
            #     [self._fixed_sitk, self._moving_sitk],
            #     segmentation=self._fixed_sitk_mask)

            # print("WARNING: SetMetricAsCorrelation")
            # registration_method.SetMetricAsCorrelation()
            # registration_transform_sitk = registration_method.Execute(
            #     self._fixed_sitk, self._moving_sitk)

        if self._registration_type == "Rigid":
            try:
                # Euler Transform used:
                registration_transform_sitk = eval(
                    "sitk.Euler%dDTransform(registration_transform_sitk)" % (
                        dimension))
            except:
                # VersorRigid used: Transform from VersorRigid to Euler
                registration_transform_sitk = eval(
                    "sitk.VersorRigid%dDTransform(registration_transform_sitk)"
                    % (dimension))
                tmp = eval("sitk.Euler%dDTransform()" % (dimension))
                tmp.SetMatrix(
                    registration_transform_sitk.GetMatrix())
                tmp.SetTranslation(
                    registration_transform_sitk.GetTranslation())
                tmp.SetCenter(
                    registration_transform_sitk.GetCenter())
                registration_transform_sitk = tmp

        elif self._registration_type == "Similarity":
            registration_transform_sitk = eval(
                "sitk.Similarity%dDTransform(registration_transform_sitk)" % (
                    dimension))
        elif self._registration_type == "Affine":
            registration_transform_sitk = sitk.AffineTransform(
                registration_transform_sitk)

        if self._verbose:
            ph.print_info("Summary SimpleItkRegistration:")
            ph.print_info("\tOptimizer\'s stopping condition: %s" % (
                registration_method.GetOptimizerStopConditionDescription()))
            ph.print_info("\tFinal metric value: %s" % (
                registration_method.GetMetricValue()))

            sitkh.print_sitk_transform(registration_transform_sitk)

        self._registration_transform_sitk = registration_transform_sitk

    def _get_transformed_fixed_sitk(self):
        return sitkh.get_transformed_sitk_image(
            self._fixed_sitk, self.get_registration_transform_sitk())

    def _get_transformed_fixed_sitk_mask(self):
        return sitkh.get_transformed_sitk_image(
            self._fixed_sitk_mask, self.get_registration_transform_sitk())

    def _get_warped_moving_sitk(self):
        warped_moving_sitk = sitk.Resample(
            self._moving_sitk,
            self._fixed_sitk,
            self.get_registration_transform_sitk(),
            eval("sitk.sitk%s" % (self._interpolator)),
            0.,
            self._moving_sitk.GetPixelIDValue(),
        )

        return warped_moving_sitk

    def _get_warped_moving_sitk_mask(self):

        warped_moving_sitk_mask = sitk.Resample(
            self._moving_sitk_mask,
            self._fixed_sitk,
            self.get_registration_transform_sitk(),
            sitk.sitkNearestNeighbor,
            0,
            self._moving_sitk_mask.GetPixelIDValue(),
        )

        return warped_moving_sitk_mask

    def _print_info_text(self):
        ph.print_info("Registration: SimpleITK")
        ph.print_info("Transform Model: %s"
                      % (self._registration_type))
        ph.print_info("Interpolator: %s"
                      % (self._interpolator))
        ph.print_info("Metric: %s" % (self._metric))
        if self._metric_params is not None:
            for k, v in six.iteritems(self._metric_params):
                ph.print_info("Metric-%s: %s" % (str(k), str(v)))
        ph.print_info("CenteredTransformInitializer: %s"
                      % (self._initializer_type))
        ph.print_info("Optimizer: %s"
                      % (self._optimizer))
        ph.print_info("Optimizer Scales Estimator: %s" %
                      (self._optimizer_scales))
        ph.print_info("Use Multiresolution Framework: %s"
                      % (self._use_multiresolution_framework),
                      newline=not self._use_multiresolution_framework)
        if self._use_multiresolution_framework:
            print(
                " (" +
                "shrink factors = " + str(self._shrink_factors) +
                ", " +
                "smoothing sigmas = " + str(self._smoothing_sigmas) +
                ")"
            )
        ph.print_info("Use Fixed Mask: %s"
                      % (self._fixed_sitk_mask is not None))
        ph.print_info("Use Moving Mask: %s"
                      % (self._moving_sitk_mask is not None))
