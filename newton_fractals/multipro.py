# Chris Harshaw
# December 2015
#

import numpy as np
from newton_fractals import general_newton as gn
from newton_fractals import test_fun as tf
import os
import shutil
from time import time

# movie to be created
directory = "fractal_videos/fractal_stills6"
filename = "newton_fractal6.avi"
imagename = "fractal"
disptime = True

# create directory
if not os.path.exists(directory):
    os.makedirs(directory)
#else:
#    shutil.rmtree(directory)
#    os.makedirs(directory)

# create grid of complex numbers
re_lim = [-0.125, 0.125]
re_num = 1000
im_lim = [-0.125, 0.125]
im_num = 1000
Z = gn.complex_grid(re_lim, re_num, im_lim, im_num)

# generate polynomial functions
p = [1.0, 0.0, -2.0, 2.0]
dp = tf.poly_der(p)
params = {'p': p, 'dp': dp}
f_val = tf.poly_fun
df_val = tf.d_poly_fun
known_roots = np.roots(p)

# frame parameters
vid_len = 5  # length of gif
frame_ps = 18  # number of frames per second
quality = 22  # the quality of the encoding

# colors
colors = [(0, 255, 255), (128, 128, 255), (255, 0, 255), (255, 128, 128)]

# generalized newton parameter, a
a_seq = np.linspace(1.0072, 1.00701, 30000)
inds = range(len(a_seq))


# create image sequence


def worker_fun(i):
    a = a_seq[i]
    a0 = a_seq[i]
    a1 = a_seq[i+1]

    if disptime:
        start = time()
    img_file_name = directory + '/' + imagename + '%05d' % i + '.png'
    print('Creating frame ' + str(i) + ' of ' + str(a_seq.size) + ' with alpha =' + str(a))

    # newton's method
    roots, con_root, con_num = gn.newton_method(Z, f_val, df_val, params, max_iter=1500, tol=1e-3,a0 = a0, a1 = a1, disp_time=False, known_roots=known_roots)

    # create image in folder
    gn.newton_plot(con_root, con_num, colors, save_path=img_file_name, max_shade=1500)
    if disptime:
        elapsed = time() - start
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        print("Frame " + str(i) + ' of ' + str(a_seq.size) + ' took ' + "%d:%02d:%02d" % (h, m, s))



