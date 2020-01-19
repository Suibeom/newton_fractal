# Chris Harshaw
# December 2015
#

import numpy as np
from newton_fractals import multipro as mp
import subprocess
import multiprocessing
import time
import glob
import re

# movie to be created
directory = "fractal_videos/fractal_stills14"
filename = "newton_fractal14.avi"
imagename = "fractal"

existing_files = glob.glob("./"+directory + "/" + imagename +"*.png")
frame_match_regex = r"\./" + re.escape(directory) + r"/" + re.escape(imagename) + r"(\d{5}).png"
existing_frames = [int(re.match(frame_match_regex, file)[1]) for file in existing_files]

# frame parameters
vid_len = 5  # length of gif
frame_ps = 30  # number of frames per second
quality = 22  # the quality of the encoding

# colors

# generalized newton parameter, a
inds = [a for a in range(len(mp.a_seq)) if a not in existing_frames]

# create image sequence

#mp.worker_fun(1)

p = multiprocessing.Pool(multiprocessing.cpu_count())

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
