import numpy as np
import scipy.io as sio

from gridlod.world import World
from gridlod import util, fem, lod, interp
import algorithms, build_coefficient, lod_periodic

NFine = np.array([256, 256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([64,64])
NCoarse = np.array([16,16])
k=3
NSamples = 350
dim = np.size(NFine)

boundaryConditions = None
alpha = 1.
beta = 10.
left = np.array([0.25, 0.25])
right = np.array([0.75, 0.75])
pList = [0.01, 0.05, 0.1, 0.15]
modelfill = {'name': 'inclfill', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right}
modelshift = {'name': 'inclshift', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.75, 0.75]), 'def_tr': np.array([1., 1.])}
modelLshape = {'name': 'inclLshape', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.5, 0.5]), 'def_tr': np.array([0.75, 0.75])}
modelList = [modelfill, modelshift, modelLshape]
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()


def computeKmsij(TInd, a, IPatch):
    patch = lod_periodic.PatchPeriodic(world, k, TInd)
    aPatch = lod_periodic.localizeCoefficient(patch, a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
bFull = basis.T * MFull * f
faverage = np.dot(MFull * np.ones(NpFine), f)

for model in modelList:
    ii = 0

    abs_error = np.zeros((len(pList), NSamples))
    rel_error = np.zeros((len(pList), NSamples))

    aRefListdef, KmsijListdef, muTPrimeListdef, _, _ = algorithms.computeCSI_offline(world, Nepsilon // NCoarse, k,
                                                                                     boundaryConditions, model)
    aRef = np.copy(aRefListdef[-1])
    KmsijRef = np.copy(KmsijListdef[-1])
    muTPrimeRef = muTPrimeListdef[-1]
    for p in pList:
        for N in range(NSamples):
            aPert = build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,left,right,p,model)

            #true LOD
            middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
            patchRef = lod_periodic.PatchPeriodic(world, k, middle)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
            computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

            bFull = basis.T * MFull * f
            uFulltrue, _ = lod_periodic.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
            uLodCoarsetrue = basis * uFulltrue
            L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull*uLodCoarsetrue))

            # combined LOD
            KFullcombdef, _ ,_= algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefListdef,KmsijListdef,
                                                                      muTPrimeListdef,k,model,compute_indicator=False)
            bFull = basis.T * MFull * f
            uFullcombdef, _ = lod_periodic.solvePeriodic(world, KFullcombdef, bFull, faverage, boundaryConditions)
            uLodCoarsecombdef = basis * uFullcombdef

            abs_error_combined = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsecombdef,
                                                MFull * (uLodCoarsetrue - uLodCoarsecombdef)))
            abs_error[ii, N] = abs_error_combined
            rel_error[ii, N] = abs_error_combined / L2norm

        rmserr = np.sqrt(1. / NSamples * np.sum(rel_error[ii, :] ** 2))
        print("root mean square relative L2-error for new LOD over {} samples for p={} and model {} is: {}".
              format(NSamples, p, model['name'], rmserr))
        ii += 1

    sio.savemat('_meanErr2d_defchanges' + str(model['name']) + '.mat',
                {'abs': abs_error, 'relerr': rel_error, 'pList': pList})