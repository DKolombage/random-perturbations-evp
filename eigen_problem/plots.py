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
from gridlod.gridlod import util
from gridlod.gridlod.world import World
from random_perturbations import build_coefficient, lod_periodic

np.random.seed(1)

        
# H-convergence and p-depenence plots
def plots_cvg(root, H_Convergence=False, p_Convergence=True, dim=2):
    pNC = sio.loadmat(root + '_pList_NCList' + '.mat')
    pList = pNC['pList'][0]
    NC_list = pNC['NC_list'][0]
    print("hplot", NC_list)
    if H_Convergence:
        #ax1=plt.figure().add_subplot()
        #ax2=plt.figure().add_subplot()
        ax3=plt.figure().add_subplot()
        data_array = sio.loadmat(root + '_RMSErr_H' + '.mat')
        err_Lam1 = data_array['rmserr_lamb1']
        err_Lam2 = data_array['rmserr_lamb2']
        err_Lam = data_array['rmserr_lamb']
        for i in range(len(pList)):
            labelplain = 'p={' + str(pList[i]) + '}'
            #ax1.loglog(NC_list, err_Lam1[:, i], label=r'${}$'.format(labelplain), marker='>')
            #ax2.loglog(NC_list, err_Lam2[:,i], label=r'${}$'.format(labelplain), marker='<')
            ax3.loglog(NC_list, err_Lam[:,i], label=r'${}$'.format(labelplain), marker='<')
        #ax1.loglog(NC_list, [err_Lam1[0,0]*0.5**(j*3)for j in range(len(NC_list))], lw = 1.0, color="black",  linestyle='dashed',label='$\mathscr{O}(H^3)$')
        #ax2.loglog(NC_list, [err_Lam2[0,0]*0.5**(j*3)for j in range(len(NC_list))], lw = 1.0, color="black",  linestyle='dashed',label='$\mathscr{O}(H^3)$')
        #ax3.loglog(NC_list, [err_Lam[0,0]*0.5**(j*3)for j in range(len(NC_list))], lw = 1.0, color="black",  linestyle='dashed',label='$\mathscr{O}(H^3)$')
        #ax1.loglog(NC_list, [err_Lam1[0,0]*0.5**(j*2)for j in range(len(NC_list))], lw = 1.0, color="black",  linestyle='dashed',label='$\mathscr{O}(H^2)$')
        #ax2.loglog(NC_list, [err_Lam2[0,0]*0.5**(j*2)for j in range(len(NC_list))], lw = 1.0, color="black",  linestyle='dashed',label='$\mathscr{O}(H^2)$')
        ax3.loglog(NC_list, [err_Lam[0,0]*0.5**(j*2)for j in range(len(NC_list))], lw = 1.0, color="black",  linestyle='dashed',label='$\mathscr{O}(H^2)$')
        #ax1.legend()
        #ax2.legend()
        ax3.legend()
        #ax1.set_xlabel('$H^{-1}$')
        #ax1.set_ylabel('root means square error of $λ_1$')
        #ax2.set_xlabel('$H^{-1}$')
        #ax2.set_ylabel('root means square error of $λ_2$')
        ax3.set_xlabel('$H^{-1}$')
        ax3.set_ylabel('root means square error of $λ$')
        plt.show()

    if p_Convergence:
        fig = plt.figure()
        #ax4 = fig.add_subplot(1, 2, 1)
        #ax5 = fig.add_subplot(1, 2, 2)
        ax6=plt.figure().add_subplot()

        if dim == 1:
            i = -5  
            for N in NC_list[2:4]:
                err = sio.loadmat(root + '_meanErr_H' + str(N) + '.mat')
                Error_λ1 = err['absErr_1']
                Error_λ2 = err['absErr_2']
                Error = err['absErr']
                NSamples = len(Error_λ2[0, :])
                rms_λ1 = []
                rms_λ2 = []
                rms=[]
                for ii in range(len(pList)):
                    rms_λ1.append(np.sqrt(1. / NSamples * np.sum(Error_λ1[ii, :] ** 2)))
                    rms_λ2.append(np.sqrt(1. / NSamples * np.sum(Error_λ2[ii, :] ** 2)))
                    rms.append(np.sqrt(1. / NSamples * np.sum(Error[ii, :] ** 2)))
                labelplain = 'H=2^{' + str(i) + '}'
                #ax4.plot(pList, rms_λ1, '-*', label=r'${}$'.format(labelplain))
                #ax5.plot(pList, rms_λ2, '-*', label=r'${}$'.format(labelplain))
                ax6.plot(pList, rms, '-*', label=r'${}$'.format(labelplain))
                i -= 1
        else:
            i = -4
            for N in NC_list[1:3]:
                err = sio.loadmat(root + '_meanErr_H' + str(N) + '.mat')
                Error_λ1 = err['absErr_1']
                Error_λ2 = err['absErr_2']
                Error = err['absErr']
                NSamples = len(Error_λ2[0, :])
                rms_λ1 = []
                rms_λ2 = []
                rms=[]
                for ii in range(len(pList)):
                    rms_λ1.append(np.sqrt(1. / NSamples * np.sum(Error_λ1[ii, :] ** 2)))
                    rms_λ2.append(np.sqrt(1. / NSamples * np.sum(Error_λ2[ii, :] ** 2)))
                    rms.append(np.sqrt(1. / NSamples * np.sum(Error[ii, :] ** 2)))
                labelplain = 'H=2^{' + str(i) + '}'
                #ax4.plot(pList, rms_λ1, '-*', label=r'${}$'.format(labelplain))
                #ax5.plot(pList, rms_λ2, '-*', label=r'${}$'.format(labelplain))
                ax6.plot(pList, rms, '-*', label=r'${}$'.format(labelplain))
                i -= 1
        #ax4.legend()
        #ax5.legend()
        ax6.legend()
        #ax4.set_xlabel('p')
        #ax4.set_ylabel('root means square error of $λ_1$')
        #ax5.set_xlabel('p')
        #ax5.set_ylabel('root means square error of $λ_2$')
        ax6.set_xlabel('p')
        ax6.set_ylabel('root means square error of $λ$')
        plt.show()

        
# Comparison plots between sum constraint one model and alternate model
def plot_s_vs_unity_errors(root1, root2, p_cvg = True, H_cvg = False, relative = False):
    pNC = sio.loadmat(root1 + '_pList_NCList' + '.mat')
    pList = pNC['pList'][0]
    NC_list = pNC['NC_list'][0]
    #slist = sio.loadmat(root1+'s_values'+'.mat')

    data_array_unity = sio.loadmat(root1 + '_RMSErr_H' + '.mat')
    err_Lam_unity = data_array_unity['rmserr_lamb']
    data_array_s=sio.loadmat(root2+'_RMSErr_H' + '.mat')
    err_Lam_s = data_array_s['rmserr_lamb']
    if relative:
        ax1=plt.figure().add_subplot()
        su_diff = np.divide((err_Lam_unity-err_Lam_s), err_Lam_unity)
        for i in range(len(pList)):
            labelplain = 'p={' + str(pList[i]) + '}'
            ax1.loglog(NC_list, su_diff[:,i], label=r'${}$'.format(labelplain), marker='+')
        ax1.legend()
        ax1.set_xlabel('$H^{-1}$')
        ax1.set_ylabel('relative error between uty_error vs s_error$')
    if H_cvg:
        ax2=plt.figure().add_subplot()
        for i in range(0,3):
            labelplain1 = 'p_{uty}={' + str(pList[i]) + '}'
            labelplain2 = 'p_{s}={' + str(pList[i]) + '}'
            ax2.loglog(NC_list, err_Lam_unity[:,i],  label=r'${}$'.format(labelplain1), marker ='+')
            ax2.loglog(NC_list, err_Lam_s[:,i], label=r'${}$'.format(labelplain2), marker ='<')
        ax2.legend()
        ax2.set_xlabel('$H^{-1}$')
        ax2.set_ylabel('root means square error of $λ_2$')
    
    if p_cvg:
        fig = plt.figure()
        ax4 = fig.add_subplot(1, 2, 1)
        ax5 = fig.add_subplot(1, 2, 2)
        ax6=plt.figure().add_subplot()
        #i = -5 # change the number accordingly for all H comparisons                                  
        for N in NC_list[2:4]:                                                      
            err_1 = sio.loadmat(root1 + '_meanErr_H' + str(N) + '.mat')
            err_s = sio.loadmat(root2 + '_meanErr_H' + str(N) + '.mat')
            Error_λ1_1 = err_1['absErr_1']
            Error_λ2_1 = err_1['absErr_2']
            Error_1 = err_1['absErr']
            Error_λ1_s = err_s['absErr_1']
            Error_λ2_s = err_s['absErr_2']
            Error_s = err_s['absErr']
            NSamples = len(Error_λ2_1[0, :])
            rms_λ1_1 = []
            rms_λ2_1 = []
            rms_1=[]
            rms_λ1_s = []
            rms_λ2_s = []
            rms_s=[]
            for ii in range(len(pList)):
                rms_λ1_1.append(np.sqrt(1. / NSamples * np.sum(Error_λ1_1[ii, :] ** 2)))
                rms_λ2_1.append(np.sqrt(1. / NSamples * np.sum(Error_λ2_1[ii, :] ** 2)))
                rms_1.append(np.sqrt(1. / NSamples * np.sum(Error_1[ii, :] ** 2)))
                rms_λ1_s.append(np.sqrt(1. / NSamples * np.sum(Error_λ1_s[ii, :] ** 2)))
                rms_λ2_s.append(np.sqrt(1. / NSamples * np.sum(Error_λ2_s[ii, :] ** 2)))
                rms_s.append(np.sqrt(1. / NSamples * np.sum(Error_s[ii, :] ** 2)))
            labelplain1 = 'H_{uty}=2^{-5}'                      
            labelplain2 = 'H_s=2^{-5}' 
            labelplain3 = 'H_{uty}=2^{-6}' 
            labelplain4 = 'H_s=2^{-6}' 
            if N == NC_list[2]:                                                   # comment: for two colour comparison 
                ax6.plot(pList, rms_1, 'g--*', label=r'${}$'.format(labelplain1))
                ax6.plot(pList, rms_s, 'g-+', label=r'${}$'.format(labelplain2))
                ax4.plot(pList, rms_λ1_1, 'g--*', label=r'${}$'.format(labelplain1))
                ax4.plot(pList, rms_λ1_s, 'g-+', label=r'${}$'.format(labelplain2))
                ax5.plot(pList, rms_λ2_1, 'g--*', label=r'${}$'.format(labelplain1))
                ax5.plot(pList, rms_λ2_s, 'g-+', label=r'${}$'.format(labelplain2))
            else:
                ax6.plot(pList, rms_1, 'm--*', label=r'${}$'.format(labelplain3))
                ax6.plot(pList, rms_s, 'm-+', label=r'${}$'.format(labelplain4))
                ax4.plot(pList, rms_λ1_1, 'm--*', label=r'${}$'.format(labelplain3))
                ax4.plot(pList, rms_λ1_s, 'm-+', label=r'${}$'.format(labelplain4))
                ax5.plot(pList, rms_λ2_1, 'm--*', label=r'${}$'.format(labelplain3))
                ax5.plot(pList, rms_λ2_s, 'm-+', label=r'${}$'.format(labelplain4))
           # i -= 1                                                                 
        #ax4.legend()
        #ax5.legend()
        ax6.legend()
        #plt.xticks(np.arange(0, 0.21, 0.05), minor=True)   # set minor ticks on x-axis
        #plt.yticks(np.arange(0, 0.61, 0.02), minor=True)   # set minor ticks on y-axis
        #plt.tick_params(which='minor', length=0) 
        #plt.grid()
        #plt.grid(which='minor', alpha=0.3)  
        #ax4.set_xlabel('p')
        #ax4.set_ylabel('root means square error of $λ_1$')
        #ax5.set_xlabel('p')
        #ax5.set_ylabel('root means square error of $λ_2$')
        ax6.set_xlabel('p')
        ax6.set_ylabel('root means square error of $λ$')
        fig.savefig("comparison.svg") #, dpi=150
    plt.show()



# Plot coefficients
# Same code from plot_coeff as a function
def plots_coeffs(Nepsilon, NFine, alpha, beta, pList, type):
    if type == "rand_checkerboard":
        for p in pList:
            aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            apertGrid = aPert.reshape(NFine, order='C')
            im1 = ax1.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap='Greens')
            fig.colorbar(im1, ax=ax1)
            plt.show()
    elif type == "random_erasure":
        for p in pList:
            incl_bl = np.array([0.25, 0.25])
            incl_tr = np.array([0.75, 0.75])
            aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            apertGrid = aPert.reshape(NFine, order='C')
            im1 = ax1.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap='Greens')
            fig.colorbar(im1, ax=ax1)
            plt.show()
    else:
        for p in pList:
            incl_bl = np.array([0.25, 0.25])
            incl_tr = np.array([0.75, 0.75])
            Lshape_bl = np.array([0.5, 0.5])
            Lshape_tr = np.array([0.75, 0.75])
            shift_bl = np.array([0.75, 0.75])
            shift_tr=np.array([1., 1.])
            model1={'name': 'inclfill'}
            model2={'name':'inclshift', 'def_bl': shift_bl, 'def_tr': shift_tr}
            model3={'name':'inclLshape', 'def_bl': Lshape_bl, 'def_tr': Lshape_tr}

            aPertList = []
            aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p))
            aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 0.5))
            aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 5.))
            aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model1))
            aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model2))
            aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model3))
            fig = plt.figure()

            for ii in range(6):
                ax = fig.add_subplot(2, 3, ii+1)
                bounds = np.array([0.2, 0.8, 1.5, 5.5, 10.5])
                mycmap = plt.cm.get_cmap('Greys')
                norm = matplotlib.colors.BoundaryNorm(bounds, mycmap.N)
                apertGrid = aPertList[ii].reshape(NFine, order='C')
                im = ax.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap=mycmap, norm=norm)

            plt.show()

# Offline coefficients - Random checkerboard
def plot_offline_coeff(Nepsilon, NFine, NCoarse, k, alpha, beta, type):
    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, None)
    middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
    patch = lod_periodic.PatchPeriodic(world, k, middle)
    xpFine = util.pCoordinates(world.NWorldCoarse, patch.iPatchWorldCoarse, patch.NPatchCoarse)
    if type=="checker_board":        
        aRefList_rand = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, Nepsilon // NCoarse, world.NCoarseElement, alpha, beta)
        fig = plt.figure()
        for ii in range(4):
            ax = fig.add_subplot(1, 4, ii+1)
            apertGrid = aRefList_rand[-1+ii].reshape(patch.NPatchFine, order='C')
            im = ax.imshow(apertGrid, origin='lower', extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greens')
        #plt.figure(figsize=(10,4))
        #plt.subplot_tool()
        fig.tight_layout()
        fig.savefig("offline.png")
        plt.show()

    if type=="erasure":
        incl_bl = np.array([0.25, 0.25])
        incl_tr = np.array([0.75, 0.75])
        aRefList_incl = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse, Nepsilon // NCoarse, world.NCoarseElement, alpha, beta, incl_bl, incl_tr)
        fig = plt.figure()
        for ii in range(4):
            ax = fig.add_subplot(1, 4, ii+1)
            apertGrid = aRefList_incl[-1+ii].reshape(patch.NPatchFine, order='C')
            im = ax.imshow(apertGrid, origin='lower',
                        extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greens')
        plt.show()
    return    
    
