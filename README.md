# random-perturbations-evp

```
# This repository is part of the project for "Offline-online approximation of multiscale eigenvalue problems with random defects":
#  https://github.com/DKolombage/random-perturbations-evp
# Copyright holder: Dilini Kolombage
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository provides code for the experiments of the paper "Offline-online approximation of multiscale eigenvalue problems with random defects" by Dilini Kolombage and Barbara Verfürth. The code is based on the module `gridlod` developed by Fredrik Hellman and Tim Keil (https://github.com/fredrikhellman/gridlod/tree/0ed4c096df75040145978d48c5307ef5678efed3), and `random_perturbations` by Barbara Verfürth (https://github.com/BarbaraV/gridlod-random-perturbations). The subfolder `gridlod` consists of code for PGLOD. The subfolder `random_perturbations` extend the PGLOD with the offline-online strategy for random defects for the source problem. The subfolder `eigen_problem` extends this offline-online strategy for the eigenvalue problem with random defects. It further contains the code for an alternate online-offline strategy. The files in the `eigen_problem` were written by Dilini Kolombage.

## Setup

This setup works with a Ubuntu system. The following packages are required (tested versions):
 - python (v3.9.2)
 - numpy (v1.24.2)
 - scipy (v1.10.1)
 - scikit-sparse (v0.4.8) 
 - matplotlib (v3.7.1)
 - mpltools (v0.2.0)
 
Note that some additional requirements might be necessary for the scikit-sparse installation. For more informations please check https://scikit-sparse.readthedocs.io/en/latest/overview.html.

Please see the README of the `gridlod` module for required packages and setup. 

Please see also READOME of `random-perturbations` for further instructions on testing the offline-online strategy for the source problem experiments. Note that they have done the submodule initiation differently to ours and make sure the path setup is correct if you need to run the python scripts for experiments, i.e.(exp1-exp4).py files.

First clone this repo with git. Then, build and activate a python3 virtual environment with

```
python3 -m venv venv3
. venv3/bin/activate
```

Now install all the required python packages listed above. 

### Visualization of results
All data from the experiments are available at the `All_Data` folder as .mat-files and can be easily visualize without generating the data. These visualizations are shownen in jupiter notebooks and each file is named by the same corresponding file name as described below. To run the notebooks, you may first need to change the jupiter kernel into the virtual environment created above.

Apart from the visualization of the errors, you can also reproduce the coefficient visualizations from the paper (Figures 1 and 2). These are given as a part of the notebooks 
``E2_OLOD_2D_RC.ipynb`` and ``E3_OLOD_2D_RE.ipynb``
These images are completely independent from the other experiments.

## Reproduction of experiments

You could reproduce the data sets for the results given in the manuscript. When reproducing a certain experiment or image from the paper, please make sure to change the `plots.py` file according to the correct orders either (e.g. $H^{-4}$ and $H^{-5} or $H^{-5}$ and $H^{-6}$) as considered in the paper.

Disclaimer! The re-generation of these experiments may take quite a while. While the 1D-experiments are quite fast, it took us several hours to produce 2D-experimental results. 

Change into the correct folder 'eigen_problem':

```
cd eigen_problem
```

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

## Note
This code is meant to illustrate the numerical methods presented in the paper. It is not optimized in any way, in particular no parallelization is implemented. 

