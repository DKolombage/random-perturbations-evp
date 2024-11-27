import numpy as np
import scipy.linalg
from gridlod import fem, util

def computeErrorIndicatorFineMultiple(patch, correctorsList, aRefList, mu, aPatchNew=None):
    ''' Compute the fine error indicator e(T) for given vector mu.

    This requires reference coefficients (already localized) and their correctors. New coefficient is optional, otherwise
    assumed to be weighted sum of mu and reference coefficients.
    '''

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    if aRefList[0].ndim != 1:
        NotImplementedError("matrix-valued coefficient not yet supported")

    lambdasList = list(patch.world.localBasis.T)

    NPatchCoarse = patch.NPatchCoarse
    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse * NCoarseElement

    nref = len(aRefList)
    a = aPatchNew

    ALocFine = world.ALocFine
    P = np.column_stack(lambdasList)

    TFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement - 1, NPatchFine - 1)
    iElementPatchFine = patch.iElementPatchCoarse * NCoarseElement
    TFinetStartIndex = util.convertpCoordIndexToLinearIndex(NPatchFine - 1, iElementPatchFine)
    TFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    TFinepStartIndex = util.convertpCoordIndexToLinearIndex(NPatchFine, iElementPatchFine)

    A = np.zeros_like(world.ALocCoarse)
    aBar = np.einsum('i, ij->j', mu, aRefList)
    if aPatchNew is None:
        a = aBar
    else:
        a = aPatchNew
        bTcoeff = np.sqrt(a)*(1-aBar/a)
        bT = bTcoeff[TFinetStartIndex + TFinetIndexMap]
        TNorm = fem.assemblePatchMatrix(NCoarseElement, ALocFine, bT**2)

    nnz = np.where(mu != 0)[0]

    def addtoA(A,kk):
        ii = kk[0]
        jj = kk[1]
        bij = (mu[ii]*np.sqrt(a)*(1-aRefList[ii]/a))*(mu[jj]*np.sqrt(a)*(1-aRefList[jj]/a))
        PatchNorm = fem.assemblePatchMatrix(NPatchFine, ALocFine,bij)
        Q1 = np.column_stack(correctorsList[ii])
        Q2 = np.column_stack(correctorsList[jj])
        A += np.dot(Q1.T, PatchNorm*Q2)
        if aPatchNew is not None:
            bii = mu[ii]*np.sqrt(a)*(1-aRefList[ii]/a)
            bjj = mu[jj] * np.sqrt(a) * (1 - aRefList[jj] / a)
            bTii = bT * bii[TFinetStartIndex + TFinetIndexMap]
            bTjj = bT * bjj[TFinetStartIndex + TFinetIndexMap]
            TNormPQ = fem.assemblePatchMatrix(NCoarseElement, ALocFine, bTjj)
            TNormQP = fem.assemblePatchMatrix(NCoarseElement, ALocFine, bTii)
            QT1 = Q1[TFinepStartIndex + TFinepIndexMap, :]
            QT2 = Q2[TFinepStartIndex + TFinepIndexMap, :]
            A -= np.dot(P.T, TNormPQ*QT2)
            A -= np.dot(QT1.T, TNormQP*P)

    assembleA = lambda kk: addtoA(A,kk)
    import itertools
    list(map(assembleA, itertools.product(nnz, repeat=2)))

    if aPatchNew is not None:
        A += np.dot(P.T, TNorm*P)

    BNorm = fem.assemblePatchMatrix(NCoarseElement, ALocFine, a[TFinetStartIndex + TFinetIndexMap])
    B = np.dot(P.T, BNorm * P)

    eigenvalues = scipy.linalg.eigvals(A[:-1, :-1], B[:-1, :-1])
    epsilonTSquare = np.max(np.real(eigenvalues))

    return np.sqrt(epsilonTSquare)