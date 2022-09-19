# MobiusConv

The official PyTorch implementation of Möbius Convolution from the SIGGRAPH 2022 paper.

<img src="fig/operators.png" width="100%">

### [[Paper: Möbius Convolutions for Spherical CNNs]](https://www.mitchel.computer/papers/MobConv_2022.pdf)
### [[Talk: SIGGRAPH 2022]](https://www.youtube.com/watch?v=qNsr-IfQtjM&t=60s)

## Dependencies
- [PyTorch >= 1.10](https://pytorch.org)
- [CMake](https://cmake.org/)

Our implementation relies on PyTorch's support for complex numbers and other functionalities introduced in version 1.10. The majority of this code is not compatable with earlier PyTorch versions. We also use the python packages `progressbar2` and `mpmath` which can be installed with `pip`.

## Installation
Clone this repository and its submodules
```
$ git clone --recurse-submodules https://github.com/twmitchel/MobiusConv.git
```
The C++ executable `convcoeff` is called automatically during model initalization to precompute and store the coefficients in Equation (13) using a closed form solution. On Linux, the model can be built by running the following sequence of commands in main directory:
```
$ cd precomp
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Using Möbius Convolutions
The principal layer is an `MCResNetBlock` -- two Möbius convolutions, each followed by filter response normalization and a thresholded nonlinearity, with a residual connection between the input and output features. The layer can be initalized as follows
```python
from MobiusConv.nn import MCResNetBlock

# Spherical band-limit of input signals (corresponds to a 2B x 2B DH spherical grid - see TS2Kit doc for more information)
B = 64

# Number of input channels
CIn = 16

# Number of output channels
COut = 16

# Radial (longitudinal) band-limit of log-polar filters
D1 = 1

# Angular (latitudinal) band-limit of log-polar filters
D2 = 1

# Angular band-limit of representation
M = D2 + 1

# Number of quadrature samples in representation
Q = 30

# Wheter or not to use checkpointing (trade speed for less memory overhead, useful in large networks or at high resolutions)
checkpoint = False;

# Initalize an MCResNetBlock
MCRN = MCResNetBlock(CIn, COut, B, D1, D2, M, Q, checkpoint=checkpoint)
```
 `MobiusConv` and `MCResNetBlock` modules initalized with band-limit `B`, `CIn` input channels, and `COut` output channels, expect input features to be `torch.float` tensors of dimension `b` X `CIn` X `2B` X `2B` corresponding to `CIn`-channel features sampled on a `2B` X `2B` Driscoll-Healy spherical grid (see TS2Kit documentation) with `b` batch dimensions. Each file contains additional documentation in the form of inline comments.

#### Simple equivariance demo
An example of how to set up Möbius Convolutions and a simple equivariance demo can be found in the `demo_mobius_conv.ipynb` notebook.

#### UNet example: pooling + unpooling
An example of a UNet architecture with Möbius Convolutions can be found in the `nn/mc_unet.py`, including pooling and unpooling operations.

## Authorship and citation information

Author: Thomas (Tommy) Mitchel (thomas.w.mitchel 'at' gmail 'dot' com)

Please cite our paper if this code or our method contributes to a publication:
```
@inproceedings{10.1145/3528233.3530724,
author = {Mitchel, Thomas W. and Aigerman, Noam and Kim, Vladimir G. and Kazhdan, Michael},
title = {M\"{o}bius Convolutions for Spherical CNNs},
year = {2022},
isbn = {9781450393379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3528233.3530724},
doi = {10.1145/3528233.3530724},
booktitle = {ACM SIGGRAPH 2022 Conference Proceedings},
articleno = {30},
numpages = {9},
keywords = {Neural networks, M\"{o}bius transformations, Group equivariance, Convolution, Conformal transformations},
location = {Vancouver, BC, Canada},
series = {SIGGRAPH '22}
}
``
