#!/usr/bin/env python

import os
import pycpd
import argparse
import numpy as np
import SimpleITK as sitk
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pysitk.python_helper as ph
import nsol.principal_component_analysis as pca

import simplereg.data_reader as dr
import simplereg.data_writer as dw


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(
        X[:, 0],  X[:, 1], X[:, 2],
        color='red', label='Target', marker="x",
    )
    ax.scatter(
        Y[:, 0],  Y[:, 1], Y[:, 2],
        color='blue', label='Source'
    )
    ax.text2D(0.87, 0.92,
              'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error),
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax.transAxes,
              fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():

    time_start = ph.start_timing()

    # Read input
    parser = argparse.ArgumentParser(
        description="Perform rigid registration using landmarks",
        prog=None,
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )
    parser.add_argument(
        "-f", "--fixed",
        help="Path to fixed image landmarks.",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-m", "--moving",
        help="Path to moving image landmarks.",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path for obtained SimpleITK registration transform (.txt)",
        type=str,
        required=1,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Turn on/off verbose output",
        type=int,
        required=0,
        default=0,
    )
    parser.add_argument(
        "--pca", "-pca",
        action="store_true",
        help="If given, principal component analysis (PCA) is used "
        "to test various initializations for the point based registrations."
    )

    args = parser.parse_args()

    landmarks_fixed_nda = dr.DataReader.read_landmarks(args.fixed)
    landmarks_moving_nda = dr.DataReader.read_landmarks(args.moving)

    if args.pca:
        ph.print_subtitle("Use PCA to initialize registrations")
        pca_fixed = pca.PrincipalComponentAnalysis(landmarks_fixed_nda)
        pca_fixed.run()
        eigvec_fixed = pca_fixed.get_eigvec()
        mean_fixed = pca_fixed.get_mean()

        pca_moving = pca.PrincipalComponentAnalysis(landmarks_moving_nda)
        pca_moving.run()
        eigvec_moving = pca_moving.get_eigvec()
        mean_moving = pca_moving.get_mean()

        # test different initializations based on eigenvector orientations
        orientations = [
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]
        error = np.inf
        for i_o, orientation in enumerate(orientations):
            eigvec_moving_o = np.array(eigvec_moving)
            eigvec_moving_o[:, 0] *= orientation[0]
            eigvec_moving_o[:, 1] *= orientation[1]

            # get right-handed coordinate system
            cross = np.cross(eigvec_moving_o[:, 0], eigvec_moving_o[:, 1])
            eigvec_moving_o[:, 2] = cross

            # transformation to align fixed with moving eigenbasis
            R = eigvec_moving_o.dot(eigvec_fixed.transpose())
            t = mean_moving - R.dot(mean_fixed)

            ph.print_info(
                "Registration based on PCA eigenvector initialization "
                "%d/%d ... " % (i_o + 1, len(orientations)), newline=False)
            reg = pycpd.rigid_registration(**{
                "Y": landmarks_fixed_nda,
                "X": landmarks_moving_nda,
                "max_iterations": 100,
                "R": R,
                "t": t,
            })
            reg.register()
            params = reg.get_registration_parameters()
            scale, R, t = params
            error_o = reg.err
            print("done. Error: %.2f" % error_o)
            if error_o < error:
                error = error_o
                rotation_matrix_nda = np.array(R)
                translation_nda = np.array(t)
                ph.print_info("Currently best estimate")

    else:
        reg = pycpd.rigid_registration(**{
            "Y": landmarks_fixed_nda,
            "X": landmarks_moving_nda,
            "max_iterations": 100,
        })
        if args.verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            callback = partial(visualize, ax=ax)
        else:
            callback = None
        ph.print_info("Registration ... ", newline=False)
        reg.register(callback)
        if args.verbose:
            plt.show(block=False)
        # reg.register()
        scale, R, t = reg.get_registration_parameters()
        rotation_matrix_nda = R
        translation_nda = t
        print("done. Error: %.2f" % reg.err)

    rigid_transform_sitk = sitk.Euler3DTransform()
    rigid_transform_sitk.SetMatrix(rotation_matrix_nda.flatten())
    rigid_transform_sitk.SetTranslation(translation_nda)

    dw.DataWriter.write_transform(
        rigid_transform_sitk, args.output, verbose=True)

    elapsed_time_total = ph.stop_timing(time_start)
    ph.print_info("Computational Time: %s" % elapsed_time_total)

    return 0


if __name__ == '__main__':
    main()
