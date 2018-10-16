# CUDAstuff
This repo contains all of my CUDA stuff. For each problem, I'm also making sure
to put in utility functions to look at the performance results of the CUDA
implementation as well maybe do some analysis on them. For now, I am making all
data available in a csv that can be passed to python, but in the future I want to
use Cython to create python binding functions and make it easier to do everything from within Python. 

## hello_world
Contains a basic introduction to CUDA and some good programs to test if your GPU 
is working correctly. 

## vector_sum
Parallel vector addition on the GPU. Includes utility functions to generate csv
of data.
