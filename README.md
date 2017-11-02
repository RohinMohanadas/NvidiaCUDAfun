# NvidiaCUDAfun
Fun with GPU Programming

# Device used
Nvidia GT 755M

# CUDA version
v9.0

# How to run
Compile using( Note: architecture depends on your GPU )
`nvcc -arch=sm_30 -I./include exercise<n>.cu -o exercise<n> [optional arguments in exercise3]`
Run using
`./exercise<n>`