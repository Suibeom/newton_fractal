FROM continuumio/anaconda3:latest

LABEL maintainer="Suibeom"

WORKDIR /newton_fractals

ARG VID_DIR=/newton_fractals/fractal_videos

ENV PYTHONPATH /newton_fractals

RUN mkdir -p ${VID_DIR}

COPY general_newton.py multipro.py newton_movie_multi.py test_fun.py /newton_fractals/newton_fractals/

VOLUME [${VID_DIR}]

RUN conda install pillow numba numpy matplotlib

CMD ["python", "./newton_fractals/newton_movie_multi.py"]

