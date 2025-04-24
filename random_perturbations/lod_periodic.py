import numpy as np
import scipy.sparse as sparse

from gridlod.gridlod import fem, util

class PatchPeriodic:
    ''' Patch object in periodic setting. Adapted from non-periodic setting in gridlod.world.Patch
        '''

    def __init__(self, world, k, TInd):
        self.world = world
        self.k = k
        self.TInd = TInd

        assert(2*k+1 <= np.min(world.NWorldCoarse))

        iElementWorldCoarse = util.convertpLinearIndexToCoordIndex(world.NWorldCoarse - 1, TInd)[:]
        self.iElementWorldCoarse = iElementWorldCoarse

        # Compute (NPatchCoarse, iElementPatchCoarse) from (k, iElementWorldCoarse, NWorldCoarse)
        d = np.size(iElementWorldCoarse)
        NWorldCoarse = world.NWorldCoarse
        iPatchWorldCoarse = (iElementWorldCoarse - k).astype('int64')
        self.iElementPatchCoarse = iElementWorldCoarse - iPatchWorldCoarse
        for i in range(d):
            if iPatchWorldCoarse[i] < 0:
                iPatchWorldCoarse[i] += NWorldCoarse[i]
        self.NPatchCoarse = (2*k + 1)*np.ones(d, dtype='int64')
        self.iPatchWorldCoarse = iPatchWorldCoarse

        self.NPatchFine = self.NPatchCoarse * world.NCoarseElement

        self.NpFine = np.prod(self.NPatchFine + 1)
        self.NtFine = np.prod(self.NPatchFine)
        self.NpCoarse = np.prod(self.NPatchCoarse + 1)
        self.NtCoarse = np.prod(self.NPatchCoarse)


def localizeCoefficient(patch, aFine, periodic=False):
    ''' localizes a coefficient aFine to patch. Optional argument whether erveything is to be interpreted in periodic
    manner. Adapted from gridlod.coef.localizeCoefficient, periodicty functionality is newly added'''
    
    iPatchWorldCoarse = patch.iPatchWorldCoarse
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = patch.world.NCoarseElement
    NPatchFine = NPatchCoarse * NCoarseElement
    iPatchWorldFine = iPatchWorldCoarse * NCoarseElement
    NWorldFine = patch.world.NWorldFine
    NtPatchFine = np.prod(NPatchFine)
    d = np.size(iPatchWorldCoarse)

    # a
    coarsetIndexMap = util.lowerLeftpIndexMap(NPatchFine - 1, NWorldFine - 1)
    coarsetStartIndex = util.convertpCoordIndexToLinearIndex(NWorldFine - 1, iPatchWorldFine)
    if periodic:
        coarsetIndCoord = (iPatchWorldFine.T + util.convertpLinearIndexToCoordIndex(NWorldFine - 1, coarsetIndexMap).T) \
                          % NWorldFine
        coarsetIndices = util.convertpCoordIndexToLinearIndex(NWorldFine - 1, coarsetIndCoord)  
        aFineLocalized = aFine[coarsetIndices]  
    else:
        aFineLocalized = aFine[coarsetStartIndex + coarsetIndexMap]
    return aFineLocalized


def assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=False):
    '''Compute the multiscale Petrov-Galerkin stiffness matrix given
    Kmsij for each coarse element. Adapted from gridlod.pglod.assembleMsStiffnessMatrix with newly added periodic
    functionality. In the periodic case, you are also allowed to hand over just a single Kmsij if this local matrix
    is the same for every element (e.g. for periodic coefficients).

    '''
    NWorldCoarse = world.NWorldCoarse

    NtCoarse = np.prod(world.NWorldCoarse)
    NpCoarse = np.prod(world.NWorldCoarse + 1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        if periodic and not (
                isinstance(KmsijT, tuple) or isinstance(KmsijT, list)):  # if only one matrix is given in periodic case
            Kmsij = KmsijT
        else:
            Kmsij = KmsijT[TInd]
        patch = patchT[TInd]

        NPatchCoarse = patch.NPatchCoarse

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldCoarse, patch.iPatchWorldCoarse)

        if periodic:
            rowsTpCoord = (patch.iPatchWorldCoarse.T + util.convertpLinearIndexToCoordIndex(NWorldCoarse,
                                                                                            patchpIndexMap).T) \
                          % NWorldCoarse
            rowsT = util.convertpCoordIndexToLinearIndex(NWorldCoarse, rowsTpCoord)
            colsTbase = TpStartIndices[TInd] + TpIndexMap
            colsTpCoord = util.convertpLinearIndexToCoordIndex(NWorldCoarse, colsTbase).T % NWorldCoarse
            colsT = util.convertpCoordIndexToLinearIndex(NWorldCoarse, colsTpCoord)
        else:
            rowsT = patchpStartIndex + patchpIndexMap
            colsT = TpStartIndices[TInd] + TpIndexMap
        dataT = Kmsij.flatten()

        cols.extend(np.tile(colsT, np.size(rowsT)))
        rows.extend(np.repeat(rowsT, np.size(colsT)))
        data.extend(dataT)

    Kms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

    return Kms


def assembleBasisCorrectors(world, patchT, basisCorrectorsListT, periodic=False):
    '''Compute the basis correctors given the elementwise basis
    correctors for each coarse element. Adapted from gridlod.pglod.assembleBasisCorrectors with newly added periodic
    functionality. In the periodic case, you are also allowed to hand over just a single basisCorrectorsList
    if these local correctors are the same for every element (e.g. for periodic coefficients).

    '''
    NWorldCoarse = world.NWorldCoarse
    NCoarseElement = world.NCoarseElement
    NWorldFine = NWorldCoarse * NCoarseElement

    NtCoarse = np.prod(NWorldCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)
    NpFine = np.prod(NWorldFine + 1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        if periodic and not (isinstance(basisCorrectorsListT, tuple)):  # if only one CorrectorsList is given in periodic case
            basisCorrectorsList = basisCorrectorsListT
        else:
            basisCorrectorsList = basisCorrectorsListT[TInd]
        patch = patchT[TInd]

        NPatchFine = patch.NPatchCoarse * NCoarseElement
        iPatchWorldFine = patch.iPatchWorldCoarse * NCoarseElement

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldFine, iPatchWorldFine)

        if periodic:
            rowsTpCoord = (iPatchWorldFine.T + util.convertpLinearIndexToCoordIndex(NWorldFine,
                                                                                            patchpIndexMap).T) \
                          % NWorldFine
            rowsT = util.convertpCoordIndexToLinearIndex(NWorldFine, rowsTpCoord)
            colsTbase = TpStartIndices[TInd] + TpIndexMap
            colsTpCoord = util.convertpLinearIndexToCoordIndex(NWorldCoarse, colsTbase).T % NWorldCoarse
            colsT = util.convertpCoordIndexToLinearIndex(NWorldCoarse, colsTpCoord)
        else:
            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap

        dataT = np.hstack(basisCorrectorsList)

        cols.extend(np.repeat(colsT, np.size(rowsT)))
        rows.extend(np.tile(rowsT, np.size(colsT)))
        data.extend(dataT)

    basisCorrectors = sparse.csc_matrix((data, (rows, cols)), shape=(NpFine, NpCoarse))

    return basisCorrectors

def solvePeriodic(world, KmsFull, rhs, faverage, boundaryConditions=None):
    ''' solves a pglod linear system with periodic boundary conditions,
    adapted from gridlod.pglod.solves'''

    NWorldCoarse = world.NWorldCoarse
    NpCoarse = np.prod(NWorldCoarse + 1)
    d = np.size(NWorldCoarse)

    MCoarse = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse)
    averageVector = MCoarse * np.ones(NpCoarse)
    bFull = np.copy(rhs)

    if d == 1:
        free = np.arange(1, NpCoarse-1)
    elif d == 2:
        fixed = np.concatenate((np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse),
                                np.arange(NWorldCoarse[0], NpCoarse-1, NWorldCoarse[0]+1)))
        free = np.setdiff1d(np.arange(NpCoarse), fixed)

        bFull[np.arange(0, NWorldCoarse[1] * (NWorldCoarse[0] + 1)+1, NWorldCoarse[0] + 1)] \
            += bFull[np.arange(NWorldCoarse[0], NpCoarse, NWorldCoarse[0]+1)]
        bFull[np.arange(NWorldCoarse[0] + 1)] += bFull[np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse)]
        averageVector[np.arange(0, NWorldCoarse[1] * (NWorldCoarse[0] + 1)+1, NWorldCoarse[0] + 1)] \
            += averageVector[np.arange(NWorldCoarse[0], NpCoarse, NWorldCoarse[0]+1)]
        averageVector[np.arange(NWorldCoarse[0] + 1)] += averageVector[np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse)]
    else:
        NotImplementedError('higher dimensions not yet implemented')

    KmsFree = KmsFull[free][:, free]
    constraint = averageVector[free].reshape((1,KmsFree.shape[0]))
    K = sparse.bmat([[KmsFree, constraint.T],
                     [constraint, None]], format='csc')
    bFree = bFull[free] - faverage * averageVector[free]  #right-hand side with non-zero average potentially not working correctly yet
    b = np.zeros(K.shape[0])
    b[:np.size(bFree)] = bFree
    x = sparse.linalg.spsolve(K,b)
    uFree = x[:np.size(bFree)]

    uFull = np.zeros(NpCoarse)
    uFull[free] = uFree


    if d == 1:
        uFull[NpCoarse-1] = uFull[0] #not relevant in 1d
    elif d == 2:
        uFull[np.arange(NWorldCoarse[0], NpCoarse-1, NWorldCoarse[0]+1)] \
            += uFull[np.arange(0, NWorldCoarse[1]*(NWorldCoarse[0]+1),NWorldCoarse[0]+1)]
        uFull[np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse)] += uFull[np.arange(NWorldCoarse[0]+1)]
    else:
        NotImplementedError('higher dimensiona not yet implemented')


    return uFull, uFree
