# Chris Harshaw
# December 2015
#

import numpy as np
from numba import jit
from PIL import Image
from time import time
import matplotlib.cm as cm


def complex_grid(re_lim, re_num, im_lim, im_num):
    # create 2D array of complex numbers
    re = np.linspace(re_lim[0], re_lim[1], num=re_num)
    im = np.linspace(im_lim[0], im_lim[1], num=im_num)
    Re, Im = np.meshgrid(re, im)
    Z = Re + 1j * Im

    return Z


"""============================================================
newton_method()
	- program
	INPUTS:
		Z 			==> 2D numpy array of complex points
		f_val		==> function to evaluate
		df_val 		==> derivative of function to evaluate
		params		==> dictionary of parameters for function eval
		max_iter	==> maximum number of iterations in Newton's method (int)
		tol 		==> tolerance for convergence
		div_val 	==> tolerance for divergence
		a 			==> coefficient for generalized Newton
		disp_time 	==> if TRUE, display time that newton plot takes (false)
		known_roots ==> array of known roots; speeds up computation if roots are known
	OUTPUTS:
		roots 		==> computed roots 
		con_num 	==> number of iterations for convergence
		con_root 	==> root that the point converged to (labelled 0 to r-1)

============================================================"""


def newton_method(Z, f_val, df_val, params, max_iter=50, tol=1e-5, div_val=1e10, a=1.0, disp_time=True,
                  known_roots=None):
    # record run time
    if disp_time:
        start = time()

    # put in different form for increased computation speed

    con_num, con_val = run_newton_iterations(Z, f_val, df_val, params, max_iter, tol, div_val, a)
    im_num, re_num = Z.shape
    con_root = np.nan * np.ones((im_num, re_num))
    roots, con_root = process_roots(con_val, con_root, known_roots, tol, im_num, re_num)

    return roots, con_root, con_num


@jit(nopython=True)
def sqfrac(z_old, z_new):
    numerator = z_old - z_new
    return (np.imag(numerator)**2 + np.real(numerator)**2)/(np.imag(z_new)**2 + np.real(z_new)**2)

@jit(nopython=True)
def sqnorm(z):
    return np.imag(z)**2 + np.real(z)**2

def run_newton_iterations(Z, f_val, df_val, params, max_iter=50, tol=1e-5, div_val=1e10, a=1.0):
    im_num, re_num = Z.shape
    total_num = re_num * im_num
    ind = np.arange(total_num)
    Z_old = np.reshape(Z, (total_num))
    Z_mean = np.reshape(Z, (total_num))
    tol_sq = tol**2
    div_sq = div_val**2

    # create array for roots and iter_num
    con_val = np.nan * np.ones(Z_old.shape, dtype=complex)  # initialze NaN: diverge
    con_num = max_iter * np.ones(Z_old.shape)
    iter_reached = max_iter

    # for the maximum number of iterations
    for i in range(1, max_iter):

        # print iteration

        # update newton step
        Z_new = Z_old - a * (f_val(Z_old, **params) / df_val(Z_old, **params))

        # check for divergence
        div = np.array(np.where(sqnorm(Z) >= div_sq))  # note: covers divide by zero errors
        con_num[ind[div]] = i

        # check for convergence
        con = np.array(np.where(sqfrac(Z_old, Z_new) < tol_sq))
        con_val[ind[con]] = Z_new[con]
        con_num[ind[con]] = i

        # Z_new_mean = Z_mean * ((i + 1) / i) + Z_new / (i + 1)
        # check for convergence in mean
        # if i > 2:
        #    con = np.array(np.where(abs((Z_mean-Z_new_mean)/Z_new_mean) < tol))
        #    print("Convergence in mean occurred")
        #    con_val[ind[con]] = Z_new_mean[con]
        #    con_num[ind[con]] = i

        # update iterate
        Z_old = Z_new
        # Z_mean = Z_new_mean

        # remove converged and diverged points
        mask = np.ones(Z_old.shape, dtype=bool)
        mask[div] = 0
        mask[con] = 0
        Z_old = Z_old[mask]
        # Z_mean = Z_mean[mask]
        ind = ind[mask]

        # break if all points have converged or diverged
        if Z_old.size == 0:
            iter_reached = i
            break

    # reshape arrays and create one to store root numbers
    con_num = con_num.reshape((im_num, re_num))
    con_val = con_val.reshape((im_num, re_num))
    return con_num, con_val


@jit(parallel=True)
def process_roots(con_val, con_root, known_roots, tol, im_num, re_num):
    if known_roots is not None:

        # look for converged values close to known roots
        for j in range(len(known_roots)):
            root_ind = np.where(np.absolute(con_val - known_roots[j]) < tol)
            con_root[root_ind] = j

        # report the roots found
        roots_found = np.unique(con_root[np.where(np.isfinite(con_root))])
        roots_found = roots_found.astype('int')
        roots = known_roots[roots_found]
    else:
        # in the event that roots are unknown
        roots_list = []

        # look through each tested
        for i in range(im_num):
            for j in range(re_num):
                val = con_val[i, j]

                # if the point converged
                if not np.isnan(val):
                    new_root = True

                    # test if there are nearby converged points
                    for r in range(len(roots_list)):
                        if abs(val - roots_list[r][0]) < 2.0 * tol:
                            con_root[i, j] = r
                            roots_list[r].append(val)
                            new_root = False
                            break

                    # add the point if it is a new root
                    if new_root:
                        roots_list.append([val])
        roots = np.empty(len(roots_list), dtype='complex')

        # report the found roots as mean of nearby points
        for i in range(len(roots_list)):
            roots[i] = np.mean(np.array(roots_list[i], dtype='complex'), dtype='complex')
    return roots, con_root


# function to convert hex to rgb
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]


def config_colors(col_source, col_params):
    # get palette from matplotlib
    if col_source == 'matplotlib':
        try:
            num_col = col_params['col_num']
            step = 256 / num_col
            cmap = cm.get_cmap(col_params['cmap'], 256)
            colors = 255 * cmap(np.arange(0, 256, step))[:, :-1]
            colors = colors.astype(int)
        except ValueError:
            print('Colormap not recognized -- using default color scheme')
            colors = [[0, 255, 255], [128, 128, 255], [255, 0, 255], [255, 128, 128]]
            colors = np.array(colors)

    else:
        print('Color source not recognized -- using default color scheme')
        colors = [[0, 255, 255], [128, 128, 255], [255, 0, 255], [255, 128, 128]]
        colors = np.array(colors)

    return colors


"""============================================================
newton_plot()

	INPUTS:
		con_num 	==> number of iterations for convergence
		con_root 	==> root that the point converged to (numbered 0 to r-1)
		colors 		==> colors corresponding to roots numpy array nx3
		save_path	==> the relative file path for saving
		max_shade 	==> the maximum shading level. It is recommended to put 
							max_shade = max_iter when making videos

	OUTPUTS:
		img 		==> a PIL image objects

============================================================"""


@jit(parallel=True)
def rgb_fill(con_root, con_num, colors, max_shade):
    im_num, re_num = con_root.shape

    # initialize data tuple for image
    data = [(0, 0, 0)] * (re_num * im_num)

    # get number of colors
    c_num = len(colors)
    colors = np.array(colors)  # numpy array

    # configure shading
    if max_shade is None:
        max_shade = 50
    # fill data tuple for image with RGB values
    for i in range(im_num):
        for j in range(re_num):
            data[i * re_num + j] = color_root(con_root[i, j], con_num[i, j], colors, c_num, max_shade)

    return data

@jit(nopython=True)
def color_root(root_num, con_num, colors, c_num, max_shade):
    if np.isnan(root_num):
        return 0, 0, 0
    else:
        shade = 1.0 - 2.0 ** (con_num - max_shade)
        col = colors[int(root_num) % c_num, :]
        return np.round(col[0] * shade),np.round(col[1] * shade),np.round(col[2] * shade)


@jit(parallel=True)
def newton_plot(con_root, con_num, colors, save_path=None, max_shade=None):
    # get number of real and imaginary points
    im_num, re_num = con_root.shape

    data = rgb_fill(con_root, con_num, colors, max_shade)

    # create image object and fill with RGB data
    img = Image.new("RGB", (re_num, im_num))
    img.putdata(data)

    # save image if save path given
    if save_path:
        img.save(save_path)

    # return the image object
    return img
