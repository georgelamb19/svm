# svm

This repository contains Python code for training and testing a multiclass soft-margin kernelised SVM implemented using NumPy.

## Overview

The loss function used is the L1 standard hinge loss.

The constrained optimisation problems are solved using:
- Log barrier Interior point method with the feasible start Newton method,
- Sequenital Minimal Optimisation (SMO) method.
- CVXOPT python package: https://cvxopt.org/userguide/coneprog.html

Generalisation to the multiclass setting is achieved using One vs One (OVO).

An example Jupyter notebook is provided for training and testing a support vector classifier (SVC) on a reduced version of the MNIST dataset.

## Code

Includes implementations for

Log barrier Interior Point method with feasible start Newton,
Sequential Minimal Optimisation (SMO)
Kernels (Gaussian and Polynomial)
Both optimisations optimise the dual objective, so kernelisation is easily permitted. This repo also includes an example Jupyter Notebook training an SVM with both optimisation methods on the MNIST dataset.

Some notes: the code is for a binary SVM classifier, and the SMO implementation does not use advanced heuristics for picking the order of dual variables to optimise.

## Resources

References I found very helpful:

http://cs229.stanford.edu/materials/smo.pdf (SMO implementation heavily relies on this pseudocode)
http://cs229.stanford.edu/notes/cs229-notes3.pdf
https://mml-book.github.io/book/mml-book.pdf
