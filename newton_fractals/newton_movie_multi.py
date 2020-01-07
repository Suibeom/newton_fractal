# Chris Harshaw
# December 2015
#

import numpy as np
from newton_fractals import multipro as mp
import subprocess
from multiprocessing import Pool
import time

# movie to be created
directory = "fractal_videos/fractal_stills6"
filename = "newton_fractal6.avi"
imagename = "fractal"

# frame parameters
vid_len = 5  # length of gif
frame_ps = 18  # number of frames per second
quality = 22  # the quality of the encoding

# colors

# generalized newton parameter, a
inds = range(len(mp.a_seq))

# create image sequence

p = Pool(4)

rs = p.map_async(mp.worker_fun, inds, chunksize=5)
p.close()
while True:
    if rs.ready():
        break
    remaining = rs._number_left*5
    print("Waiting for " + str(remaining) + " tasks to complete...")
    time.sleep(10)

# create the movie
ctrlStr = 'ffmpeg -r %d -i %s%%05d.png -c:v libx264 -preset slow -crf %d %s' % (
    frame_ps, directory + '/' + imagename, quality, filename)
subprocess.call(ctrlStr, shell=True)
