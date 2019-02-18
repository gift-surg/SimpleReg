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

    args = parser.parse_args()

    landmarks_fixed_nda = np.loadtxt(args.fixed)
    landmarks_moving_nda = np.loadtxt(args.moving)

    if args.verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        callback = partial(visualize, ax=ax)
    else:
        callback = None

    reg = pycpd.rigid_registration(**{
        "Y": landmarks_fixed_nda,
        "X": landmarks_moving_nda,
        "max_iterations": 100,
    })
    reg.register(callback)

    if args.verbose:
        plt.show(block=False)

    params = reg.get_registration_parameters()
    scale, rotation_matrix_nda, translation_nda = params

    rigid_transform_sitk = sitk.Euler3DTransform()
    rigid_transform_sitk.SetMatrix(rotation_matrix_nda.flatten())
    rigid_transform_sitk.SetTranslation(translation_nda)

    ph.create_directory(os.path.dirname(args.output))
    sitk.WriteTransform(rigid_transform_sitk, args.output)
    if args.verbose:
        ph.print_info(
            "Rigid registration transform written to '%s'" % args.output)

    elapsed_time_total = ph.stop_timing(time_start)
    ph.print_info("Computational Time: %s" % elapsed_time_total)

    return 0


if __name__ == '__main__':
    main()
