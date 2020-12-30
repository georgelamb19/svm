# SVM: from scratch

This repository contains Python code for training and testing a multiclass soft-margin kernelised SVM implemented using NumPy.

## Overview

The loss function used is the L1 standard hinge loss.

The constrained optimisation problems are solved using:
- The Log barrier Interior point method with the feasible start Newton method
- The Sequenital Minimal Optimisation (SMO) method
- The CVXOPT python package: https://cvxopt.org/userguide/coneprog.html

Kernel functions available:
- Gaussian radial basis function (RBF) kernel
- Polynomial kernel

Generalisation to the multiclass setting is achieved using One vs One (OVO).

An example Jupyter notebook is provided for training and testing a support vector classifier on a reduced version of the MNIST dataset.

## Code

A list of optimisation algorithms that have been coded up in this repository include:
- Log barrier Interior point
- Feasible Newton
- SMO
- Backtracking linesearch obeying Armijo conditions
- Linesearch obeying strong Wolfe conditions
- Descent algorithm using steepest descent direction and Newton direction

## Resources

References I found very helpful:
- http://cs229.stanford.edu/summer2020/cs229-notes3.pdf (SVM theory)
- http://cs229.stanford.edu/materials/smo.pdf (psuedocode for the simplified SMO algorithm by Andrew Ng)
- https://web.stanford.edu/~boyd/cvxbook/ (psuedocode for the barrier method and feasible Newton method)
- https://link.springer.com/book/10.1007/978-0-387-40065-5 (for linesearch methods)
