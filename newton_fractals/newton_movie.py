# Chris Harshaw
# December 2015
#

import numpy as np
from newton_fractals import general_newton as gn
from newton_fractals import test_fun as tf
import subprocess
import os
import shutil
from time import time

# movie to be created
directory = "fractal_videos/fractal_stills4"
filename = "newton_fractal4.avi"
imagename = "fractal"
disptime = True

# create directory
if not os.path.exists(directory):
	os.makedirs(directory)
else:
	shutil.rmtree(directory)
	os.makedirs(directory)

# create grid of complex numbers
re_lim = [-0.25, 0.25]
re_num = 1000
im_lim = [-0.25, 0.25]
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
quality = 22 # the quality of the encoding

# colors
colors = [(0, 255, 255), (128, 128, 255), (255, 0, 255), (255, 128, 128)]

# generalized newton parameter, a
a_seq = np.linspace(1.0075, 1.00701, 3000)

# create image sequence
i = 1
for a in a_seq:
    # print progress
    if disptime:
        start = time()
    img_file_name = directory + '/' + imagename + '%05d' % i + '.png'
    print('Creating frame ' + str(i) + ' of ' + str(a_seq.size) + ' with alpha ='+str(a))
    i += 1

    # newton's method
    roots, con_root, con_num = gn.newton_method(Z, f_val, df_val, params, max_iter=1000, tol=1e-3, a=a, disp_time=False, known_roots=known_roots)

    # create image in folder
    gn.newton_plot(con_root, con_num, colors, save_path=img_file_name, max_shade=1000)
    if disptime:
        elapsed = time() - start
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        print("Run time: " + "%d:%02d:%02d" % (h, m, s))
# create the movie
ctrlStr = 'ffmpeg -r %d -i %s%%05d.png -c:v libx264 -preset slow -crf %d %s' %(frame_ps, directory + '/' + imagename, quality, filename)
subprocess.call(ctrlStr, shell=True)