import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup_path import add_repo_paths
add_repo_paths()

from MLOD_alg import *
from Reference_Solvers import *
from convergence import *
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpltools import annotation
import numpy as np
import matplotlib
from gridlod import util
from gridlod.world import World
#sys.path.insert(0, '/.../random-perturbations-evp/random_perturbations')
from random_perturbations import build_coefficient, lod_periodic

np.random.seed(1)

#random checkerboard
Nepsilon = np.array([128,128])
NFine = np.array([256,256])
NCoarse = np.array([4,4])
k=3
alpha=0.1
beta=1.
p=0.2
aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
np.random.seed(1)
apertGrid = aPert.reshape(NFine, order='C')
im1 = ax1.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap='Greens')
fig.colorbar(im1, ax=ax1)
plt.show()

#random erasure 
Nepsilon = np.array([64,64])
NFine = np.array([256,256])
alpha=0.1
beta=1.
p=0.2
incl_bl = np.array([0.25, 0.25])
incl_tr = np.array([0.75, 0.75])
np.random.seed(1)
aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

apertGrid = aPert.reshape(NFine, order='C')
im1 = ax1.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap='Greens')
fig.colorbar(im1, ax=ax1)
plt.show()

#Offline coefficients - random checkerboards

Nepsilon = np.array([32,32])
NFine = np.array([32,32])
NCoarse=np.array([16, 16])
k=3
alpha = 0.1
beta = 1.
incl_bl = np.array([0.25, 0.25])
incl_tr = np.array([0.75, 0.75])
NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, None)
middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
patch = lod_periodic.PatchPeriodic(world, k, middle)
xpFine = util.pCoordinates(world.NWorldCoarse, patch.iPatchWorldCoarse, patch.NPatchCoarse)

np.random.seed(1)
aRefList_rand = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, Nepsilon // NCoarse,
                                                             world.NCoarseElement, alpha, beta)
fig = plt.figure()
for ii in range(4):
    ax = fig.add_subplot(1, 4, ii+1)
    apertGrid = aRefList_rand[-1+ii].reshape(patch.NPatchFine, order='C')
    im = ax.imshow(apertGrid, origin='lower', extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greens')
fig.tight_layout()
fig.savefig("offline.png")
plt.show()
