#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import pyviability as viab
# from pyviability import libviability
from pyviability import tsm_style as topo

import argparse, argcomplete
import functools as ft
from matplotlib import pyplot as plt
import numpy as np
import os

def str2array(string):
    assert string.startswith("[")
    assert string.endswith("]")
    return np.array(list(map(float, string[1:-1].split(","))))

def load_file(fname):
    # count lines etc. to get the sizes of the numpy arrays
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue
            splitted = line.split(";")
            if len(splitted) != 2:
                raise IOError(
                    f"unknown format on line number {i} in {fname} (reason: number of semi-colons)"
                )
            point_str = splitted[0].strip()
            print(repr(point_str))
            point_arr = str2array(point_str)
            print(repr(point_arr))
            if point_arr.ndim != 1:
                raise IOError(
                    f"unknown format on line number {i} in {fname} (reason: a point array should have ndim == 1)"
                )
            system_dimension = point_arr.shape[0]
            break
        num_lines = 1 + sum(1 for _ in f) # count the other lines
    # num_lines is the number of points in the result file
    points = np.empty((system_dimension, num_lines), dtype=np.float64)
    states = np.empty((num_lines), dtype=np.uint8)
    count = 0
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("#"):
                # ignore all header content for now
                continue
            line = line.strip()
            point_str, state_str = line.split(";")
            points[:, count] = str2array(point_str)
            states[count] = int(state_str)
            count += 1

    return points, states

def plotPhaseSpace( evol, boundaries, steps = 2000, xlabel = "", ylabel = "", colorbar = True, style = {}, alpha = None , maskByCond = None, invertAxes = False, ax = plt, lwspeed = False):
    # separate the boundaries
    Xmin, Ymin, Xmax, Ymax = boundaries

    # check boundaries sanity
    assert Xmin < Xmax
    assert Ymin < Ymax

    # build the grid
    X = np.linspace(Xmin, Xmax, steps)
    Y = np.linspace(Ymin, Ymax, steps)

    XY = np.array(np.meshgrid(X, Y))

    # if Condition give, set everything to zero that fulfills it
    if maskByCond:
        mask = maskByCond(XY[0], XY[1])
        XY[0] = np.ma.array(XY[0], mask = mask)
        XY[1] = np.ma.array(XY[1], mask = mask)

    # calculate the changes ... input is numpy array
    dX, dY = evol(XY,0) # that is where deriv from Vera is mapped to

    if invertAxes:
        data = [Y, X, np.transpose(dY), np.transpose(dX)]
    else:
        data = [X, Y, dX, dY]

    # separate linestyle
    linestyle = None
    if type(style) == dict and "linestyle" in style.keys():
        linestyle = style["linestyle"]
        style.pop("linestyle")

    # do the actual plot
    if style == "dx":
        c = ax.streamplot(*data, color=dX, linewidth=5*dX/dX.max(), cmap=plt.cm.autumn)
    elif style:
            speed = np.sqrt(data[2]**2 + data[3]**2)
            if "linewidth" in style and style["linewidth"] and lwspeed:
                style["linewidth"] = style["linewidth"] * speed/np.nanmax(speed)
            c = ax.streamplot(*data, **style)
    else:
        # default style formatting
        speed = np.sqrt(dX**2 + dY**2)
        c = ax.streamplot(*data, color=speed, linewidth=5*speed/speed.max(), cmap=plt.cm.autumn)

    # set opacity of the lines
    if alpha:
        c.lines.set_alpha(alpha)

    # set linestyle
    if linestyle:
        c.lines.set_linestyle(linestyle)

plotPS = lambda rhs, boundaries, style: plotPhaseSpace(rhs, [boundaries[0][0], boundaries[1][0], boundaries[0][1], boundaries[1][1]], colorbar=False, style=style)


parser = argparse.ArgumentParser()
parser.add_argument("result_file", metavar="result-file")

argcomplete.autocomplete(parser)
args = parser.parse_args()

if not os.path.isfile(args.result_file):
    parser.error(f"'{args.result_file} is not a file")

points, states = load_file(args.result_file)

print(points.shape)
print(states.shape)

def consum_rhsPS(xy, t=0, *, u):
    x, y = xy

    v = np.zeros_like(x)
    v[:] = u

    dx = x - y
    dy = v
    return [dx, dy]

management_PSs = [ft.partial(consum_rhsPS, u=-0.5), ft.partial(consum_rhsPS, u=+0.5)]
plot_x_limits = [0, 2]
plot_y_limits = [0, 3]

fig = plt.figure(figsize=(9, 9), tight_layout=True)
viab.plot_points(points.T, states, plot_unset=True)


[plotPS(rhs, [plot_x_limits, plot_y_limits], style) for rhs, style in
 zip(management_PSs, [topo.styleDefault, topo.styleMod1])]  # noqa

plt.show()


