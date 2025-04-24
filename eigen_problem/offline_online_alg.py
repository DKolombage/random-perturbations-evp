import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup_path import add_repo_paths
add_repo_paths()

import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as ln
import time
from gridlod.gridlod.world import World
from gridlod.gridlod import util, fem, lod, interp

from random_perturbations import build_coefficient, lod_periodic, indicator, algorithms


def KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=True):
    NpFine = np.prod(NFine+1)     # Number of "fine-nodes" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
    NpCoarse = np.prod(NCoarse+1) 
    dim = np.size(NFine)
    
    boundaryConditions = None
    percentage_comp = 0.15
    np.random.seed(1)

    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, boundaryConditions)
    xpFine = util.pCoordinates(NFine)

    def computeKmsij(TInd, a, IPatch):
        patch = lod_periodic.PatchPeriodic(world, k, TInd)
        aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi
    
    aRefList, KmsijList, muTPrimeList, timeBasis, timeMatrixList = algorithms.computeCSI_offline(world, Nepsilon // NCoarse,k,boundaryConditions,model,correctors=False)  
    #print(aRefList)                                                           
    aRef = np.copy(aRefList[-1])
    KmsijRef = np.copy(KmsijList[-1])
    muTPrimeRef = muTPrimeList[-1]
   
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))
    KFullpert = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijRef, periodic=True)
    KOLOD_λ1 = np.zeros((len(pList), NSamples))
    KOLOD_λ2 = np.zeros((len(pList), NSamples))

    for ii in range(len(pList)):
        p = pList[ii]
        for N in range(NSamples):
            if model['name'] == 'check':
                aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
            elif model['name'] == 'incl':
                left = model['left']
                right = model['right']
                aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,left,right,p)
            elif model['name'] == 'inclvalue':
                left = model['left']
                right = model['right']
                value = model['defval']
                aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,left,right,p, value)
            elif model['name'] in ['inclfill', 'inclshift', 'inclLshape']:
                aPert = build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,left,right,p,model)
            else:
                NotImplementedError('other models not available!')

            #combined LOD
            KFullcomb, _,_ = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,
                                                                   model,compute_indicator=False)
           
            # FEM Mass Matrix
            MFEM = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse) 

            if dim == 2:

                fixed_DoF = np.concatenate((np.arange(NCoarse[1] * (NCoarse[0] + 1), NpCoarse), 
                                                np.arange(NCoarse[0], NpCoarse - 1, NCoarse[0] + 1)))    
                free_DoF = np.setdiff1d(np.arange(NpCoarse), fixed_DoF)  

                KOOLOD_Free_DoF = KFullcomb[free_DoF][:, free_DoF]  

                MFEM.tolil()
                MFEM[np.arange(0, NCoarse[1]*(NCoarse[0]+1)+1, NCoarse[0]+1),:] \
                        += MFEM[np.arange(NCoarse[0], np.prod(NCoarse+1), NCoarse[0]+1),:]
                MFEM[:, np.arange(0, NCoarse[1] * (NCoarse[0] + 1) + 1, NCoarse[0] + 1)] \
                        += MFEM[:, np.arange(NCoarse[0], np.prod(NCoarse + 1), NCoarse[0] + 1)]
                MFEM[np.arange(NCoarse[0]+1), :] += MFEM[np.arange(NCoarse[1]*(NCoarse[0]+1), np.prod(NCoarse+1)), :]
                MFEM[:, np.arange(NCoarse[0] + 1)] += MFEM[:, np.arange(NCoarse[1] * (NCoarse[0] + 1), np.prod(NCoarse + 1))]
                MFEM.tocsc()
                MFEM_Free_DoF = MFEM[free_DoF][:, free_DoF]
            else:

                fixed_DoF = np.arange(NCoarse[0], np.prod(NCoarse + 1), NCoarse[0] + 1)
                free_DoF = np.setdiff1d(np.arange(NpCoarse-1), fixed_DoF)

                KOOLOD_Free_DoF = KFullcomb[free_DoF][:, free_DoF]

                fixed_DoF = np.arange(NCoarse[0], np.prod(NCoarse + 1), NCoarse[0] + 1)
                free_DoF = np.setdiff1d(np.arange(NpCoarse-1), fixed_DoF)

                MFEM.tolil()
                MFEM[0] += MFEM[-1]
                MFEM[:,0] += MFEM[:,-1]
                MFEM.tocsc() 

                MFEM_Free_DoF = MFEM[free_DoF][:, free_DoF]

            #evals, evecs = ln.eigsh(KOOLOD_Free_DoF , Neigen,  MFEM_Free_DoF, sigma =0.005, which='LM', return_eigenvectors = True, tol=1E-4) 
            evals, evecs = ln.eigs(KOOLOD_Free_DoF,Neigen, MFEM_Free_DoF,sigma=0.005, which='LM', tol=1e-4)
            evals = np.real(evals)
            #evecs = np.real(evecs)
                    
            KOLOD_λ1[ii, N] = evals[1]
            KOLOD_λ2[ii, N] = evals[2]
        if save_file:
            sio.savemat('KOOLOD_Eigenvalues' + '.mat', {'KOOLOD_1st_Evalue': KOLOD_λ1, 'KOOLOD_2nd_Evalue': KOLOD_λ2, 'pList': pList})
    return KOLOD_λ1, KOLOD_λ2 
