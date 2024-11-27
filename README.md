# random-perturbations-evp

```
# This repository is part of the project for "Offline-online approximation of multiscale eigenvalue problems with random defects":
#  https://github.com/DKolombage/random-perturbations-evp
# Copyright holder: Dilini Kolombage
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository provides code for the experiments of the paper "Offline-online approximation of multiscale eigenvalue problems with random defects" by Dilini Kolombage and Barbara Verfürth. The code is based on the module `gridlod` developed by Fredrik Hellman and Tim Keil, and `random_perturbations` by Barbara Verfürth. The subfolder `gridlod` consists of code for PGLOD. The subfolder `random_perturbations` extend the PGLOD with the offline-online strategy for random defects for the source problem. The subfolder `eigen_problem` extends this offline-online strategy for the eigenvalue problem with random defects. It further contains the code for an alternate online-offline strategy. The files in the `eigen_problem` were written by Dilini Kolombage.

## Setup

This setup works with a Ubuntu system. The following packages are required (tested versions):
 - python (v3.8.5)
 - numpy (v1.17.4)
 - scipy (v1.3.3)
 - scikit-sparse (v0.4.4) 
 - matplotlib (v3.1.2)
 
Note that some additional requirements might be necessary for the scikit-sparse installation. For more informations please check https://scikit-sparse.readthedocs.io/en/latest/overview.html.

Please see also the README of the `gridlod` submodule for required packages and setup.
After cloning this repo with git, initialize the submodule via

```
git submodule update --init --recursive
```

Now, build and activate a python3 virtual environment with

```
virtualenv -p python3 venv3
. venv3/bin/activate
```

and execute the following commands

```
echo $PWD/gridlod/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/gridlod.pth
```
Now you can use every file from the subfolders in the virtualenv. Install all the required python packages for gridlod. 

Please see READOME of `random-perturbations` for further instructions on testing the offline-online strategy for the source problem experiments.

## Reproduction of experiments

First of all, change into the correct folder eigen_problem

```
cd eigen_problem
```
Since the some files in eigen_problem folder requires access to the files in random_perturbations folder, the following command with the correct path to the random_perturbation folder might be necessary. 
`sys.path.insert(0, 'path to random_perturbations')`  e.g. sys.path.insert(0, '/home/kolombag/Documents/gridlod-random-perturbations/random_perturbations')


Reproduction of the experiments consists of two steps: (i) generating the data and/or (ii) visualization. 
Disclaimer! The re-generation of these experiments may take quite a while. A quick visualization is given in jupiter notebooks (.iypnb) via already stored data. 

### Generating the data
To reproduce the data of the experiments run

```
python3 E1_OLOD_1D_RC.py
```

for the experiments of Section 5.1.1;

```
python3 E2_OLOD_2D_RC.py
```

and

```
python3 E3_OLOD_2D_RE.py
```

for the experiments of Section 5.1.2;

```
python3 E4_1D_Check_s_val.py
```

and

```
python3 E5_2D_Check_s_val.py
```

for the experiments of Section 5.2. 

### Visualization
All dta from the experiments are available at the `All_Data` folder as .mat-files and can be easily visualize without generating the data. These visualizations are shownen in jupiter notebooks and each file is named by the same corresponding file name as described above.

Apart from the visualization of the errors, you can also reproduce the coefficient visualizations from the paper (Figures 1 and 2). These are given as a part of the notebooks 
``E2_OLOD_2D_RC.ipynb`` and ``E3_OLOD_2D_RE.py``
These images is completely independent from the other experiments.

## Note
This code is meant to illustrate the numerical methods presented in the paper. It is not optimized in any way, in particular no parallelization is implemented. 

