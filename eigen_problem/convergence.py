from MLOD_alg import *
from Reference_Solvers import *
import math
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpltools import annotation
from offline_online_alg import *


def errors(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, model, solver , reference_solver="FEM", save_files = True, root=None):

    Niter = 4
    NC_list = []
    rmserr_p_λ1 = []
    rmserr_p_λ2=[]
    rmserr_p=[]

    for j in range(Niter):
        NCoarse *= 2
        if reference_solver == "FEM" and solver == "KOOLOD":
            K_λ1, K_λ2 = KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False) #KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, save_file=False)
            M_λ1, M_λ2 =  FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, model, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)
            Mean_lambda_FEM = (K_λ1 + K_λ2)/2
            Mean_lambda_KOOLOD = (M_λ1 + M_λ2)/2
            absErrorList = abs(Mean_lambda_FEM-Mean_lambda_KOOLOD)


        elif reference_solver == "FEM" and solver == "LOD":
            K_λ1, K_λ2 = KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            M_λ1, M_λ2 =  FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, model, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)
            Mean_lambda_FEM = (K_λ1 + K_λ2)/2
            Mean_lambda_LOD = (M_λ1 + M_λ2)/2
            absErrorList = abs(Mean_lambda_FEM-Mean_lambda_LOD)


        elif reference_solver == "LOD" and solver == "KOOLOD":
            K_λ1, K_λ2 = KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            M_λ1, M_λ2 =  KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)
            Mean_lambda_LOD = (K_λ1 + K_λ2)/2
            Mean_lambda_KOOLOD = (M_λ1 + M_λ2)/2
            absErrorList = abs(Mean_lambda_LOD-Mean_lambda_KOOLOD)


        elif reference_solver == "exact" and solver == "LOD":
            Exact_λ = Exact_EigenSolver(Neigen)
            K_λ1, K_λ2 = KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            absErrorList_λ1 = abs(K_λ1-Exact_λ)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-Exact_λ)
        else:
            print("Unrecognized reference solver!")

        rmserr_λ1 = np.sqrt(1. / NSamples * np.sum(absErrorList_λ1** 2, axis = 1))
        rmserr_λ2 = np.sqrt(1. / NSamples * np.sum(absErrorList_λ2** 2, axis = 1))
        rmserr = np.sqrt(1. / NSamples * np.sum(absErrorList** 2, axis = 1))
        rmserr_p_λ1.append(rmserr_λ1)
        rmserr_p_λ2.append(rmserr_λ2)
        rmserr_p.append(rmserr)
        #print("rmsp1", rmserr_p_λ1)
        #print("rmsp2", rmserr_p_λ2)
        NC_list.append(np.copy(NCoarse[0]))

        if save_files:
            if not root == None:
                sio.savemat(root + '_pList_NCList'+'.mat', {'pList': pList, 'NC_list': NC_list})
                sio.savemat(root + '_meanErr_H' + str(NCoarse[0]) + '.mat', {'absErr_1': absErrorList_λ1, 'absErr_2': absErrorList_λ2, 'absErr': absErrorList, 'pList': pList})
            else: 
                sio.savemat('_pList_NCList'+'.mat', {'pList': pList, 'NC_list': NC_list})
                sio.savemat('_meanErr_H' + str(NCoarse[0]) + '.mat', {'absErr_1': absErrorList_λ1, 'absErr_2': absErrorList_λ2, 'absErr': absErrorList, 'pList': pList})

    err1 = np.array(rmserr_p_λ1)
    err2 = np.array(rmserr_p_λ2)
    err = np.array(rmserr_p)

    if save_files: 
        if not root == None:
            sio.savemat(root + '_RMSErr_H'  + '.mat', {'rmserr_lamb1': err1, 'rmserr_lamb2': err2, 'rmserr_lamb': err, 'pList': pList, 'NC_list': NC_list})
        else:
            sio.savemat('_RMSErr_H'  + '.mat', {'rmserr_lamb1': err1, 'rmserr_lamb2': err2, 'rmserr_lamb': err, 'pList': pList, 'NC_list': NC_list})
    return err1, err2, err




def convergence(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, model, solver , reference_solver="FEM", save_files = True, plot=True, root=None):

    Niter = 4
    NC_list = []
    rmserr_p_λ1 = []
    rmserr_p_λ2=[]
    rmserr_p=[]

    for j in range(Niter):
        NCoarse *= 2
        if reference_solver == "FEM" and solver == "KOOLOD":
            K_λ1, K_λ2 = KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False) #KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, save_file=False)
            M_λ1, M_λ2 =  FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)
            Mean_lambda_FEM = (K_λ1 + K_λ2)/2
            Mean_lambda_KOOLOD = (M_λ1 + M_λ2)/2
            absErrorList = abs(Mean_lambda_FEM-Mean_lambda_KOOLOD)

        elif reference_solver == "FEM" and solver == "LOD":
            K_λ1, K_λ2 = KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            M_λ1, M_λ2 =  FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)
            Mean_lambda_FEM = (K_λ1 + K_λ2)/2
            Mean_lambda_LOD = (M_λ1 + M_λ2)/2
            absErrorList = abs(Mean_lambda_FEM-Mean_lambda_LOD)

        elif reference_solver == "LOD" and solver == "KOOLOD":
            K_λ1, K_λ2 = KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            M_λ1, M_λ2 =  KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)
            Mean_lambda_LOD = (K_λ1 + K_λ2)/2
            Mean_lambda_KOOLOD = (M_λ1 + M_λ2)/2
            absErrorList = abs(Mean_lambda_LOD-Mean_lambda_KOOLOD)

        elif reference_solver == "exact" and solver == "LOD":
            Exact_λ = Exact_EigenSolver(Neigen)
            K_λ1, K_λ2 = KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            absErrorList_λ1 = abs(K_λ1-Exact_λ)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-Exact_λ)
        else:
            print("Unrecognized reference solver!")

        rmserr_λ1 = np.sqrt(1. / NSamples * np.sum(absErrorList_λ1**2, axis = 1))
        rmserr_λ2 = np.sqrt(1. / NSamples * np.sum(absErrorList_λ2**2, axis = 1))
        rmserr = np.sqrt(1. / NSamples * np.sum(absErrorList**2, axis = 1))
        rmserr_p_λ1.append(rmserr_λ1)
        rmserr_p_λ2.append(rmserr_λ2)
        rmserr_p.append(rmserr)

        NC_list.append(np.copy(NCoarse[0]))

        if save_files:
            if not root == None:
                sio.savemat(root + '_pList_NCList'+'.mat', {'pList': pList, 'NC_list': NC_list})
                sio.savemat(root + '_meanErr_H' + str(NCoarse[0]) + '.mat', {'absErr_1': absErrorList_λ1, 'absErr_2': absErrorList_λ2, 'absErr': absErrorList, 'pList': pList})
            else: 
                sio.savemat('_pList_NCList'+'.mat', {'pList': pList, 'NC_list': NC_list})
                sio.savemat('_meanErr_H' + str(NCoarse[0]) + '.mat', {'absErr_1': absErrorList_λ1, 'absErr_2': absErrorList_λ2,'absErr': absErrorList, 'pList': pList})

    err1 = np.array(rmserr_p_λ1)
    err2 = np.array(rmserr_p_λ2)
    err = np.array(rmserr_p)

    if save_files: 
        if not root == None:
            sio.savemat(root + '_RMSErr_H'  + '.mat', {'rmserr_lamb1': err1, 'rmserr_lamb2': err2, 'rmserr_lamb': err, 'pList': pList, 'NC_list': NC_list})
        else:
            sio.savemat('_RMSErr_H'  + '.mat', {'rmserr_lamb1': err1, 'rmserr_lamb2': err2, 'rmserr_lamb': err,'pList': pList, 'NC_list': NC_list})
    if plot:
        ax1=plt.figure().add_subplot()
        ax2=plt.figure().add_subplot()
        ax3 =plt.figure().add_subplot()
        for i in range(len(pList)):
            labelplain = 'p={' + str(pList[i]) + '}'
            ax1.loglog(NC_list, err1[:,i], label=r'${}$'.format(labelplain), marker='>')
            ax2.loglog(NC_list, err2[:,i], label=r'${}$'.format(labelplain), marker='<')
            ax3.loglog(NC_list, err[:,i], label=r'${}$'.format(labelplain), marker='<')
            if len(pList) ==1:
                ax1.loglog(NC_list, [err1[0,0]*0.5**(j*3)for j in range(len(NC_list))], lw = 1.0, color="red",  linestyle='dashed')
                ax2.loglog(NC_list, [err2[0,0]*0.5**(j*3)for j in range(len(NC_list))], lw = 1.0, color="red",  linestyle='dashed')
                ax3.loglog(NC_list, [err[0,0]*0.5**(j*3)for j in range(len(NC_list))], lw = 1.0, color="red",  linestyle='dashed')
            #for j in range(1, n+1):
            #    ax1.loglog(NC_list, err_Lam1[:, i]*0.5**(i*j), lw = 0.5, color="red")
            else:
                ax1.loglog(NC_list, [err1[0,0]*0.5**(i*2) for i in range(len(pList))], lw = 1.0, color="red",  linestyle='dashed') # order 2 reference line
                ax2.loglog(NC_list, [err2[0,0]*0.5**(i*2) for i in range(len(pList))], lw = 1.0, color="red",  linestyle='dashed')
                ax3.loglog(NC_list, [err[0,0]*0.5**(i*2) for i in range(len(pList))], lw = 1.0, color="red",  linestyle='dashed')
        ax2.legend()
        ax1.legend()
        ax3.legend()
        ax1.set_xlabel('$H^{-1}$')
        ax1.set_ylabel('Root Mean squard error of $λ_1$')
        ax2.set_xlabel('$H^{-1}$')
        ax2.set_ylabel('Root Mean squard error of $λ_2$')
        ax3.set_xlabel('$H^{-1}$')
        ax3.set_ylabel('Root Mean squard error of $λ$')
        plt.show()

        fig = plt.figure()
        ax4 = fig.add_subplot(1, 2, 1)
        ax5 = fig.add_subplot(1, 2, 2)
        ax6 =plt.figure().add_subplot()
        i = -2
        print("Hplot", NC_list)
        for N in NC_list:
            if not root == None:
                Err = sio.loadmat(root + '_meanErr_H' + str(N) + '.mat')
            else:
                Err = sio.loadmat('_meanErr_H' + str(N) + '.mat')
            Error_λ1 = Err['absErr_1']
            pList = Err['pList'][0]
            Error_λ2 = Err['absErr_2']
            Error_lambda = Err['absErr']
            NSamples = len(Error_λ2[0, :])
            rms_λ1 = []
            rms_λ2 = []
            rms=[]
            for ii in range(len(pList)):
                rms_λ1.append(np.sqrt(1. / NSamples * np.sum(Error_λ1[ii, :] ** 2)))
                rms_λ2.append(np.sqrt(1. / NSamples * np.sum(Error_λ2[ii, :] ** 2)))
                rms.append(np.sqrt(1. / NSamples * np.sum(Error_lambda[ii, :] ** 2)))
            labelplain = 'H=2^{' + str(i) + '}'
            ax4.plot(pList, rms_λ1, '-*', label=r'${}$'.format(labelplain))
            ax5.plot(pList, rms_λ2, '-*', label=r'${}$'.format(labelplain))
            ax6.plot(pList, rms, '-*', label=r'${}$'.format(labelplain))
            i -= 1
        ax4.legend()
        ax5.legend()
        ax6.legend()
        ax4.set_xlabel('p')
        ax4.set_ylabel('root means square error of $λ_1$')
        ax5.set_xlabel('p')
        ax5.set_ylabel('root means square error of $λ_2$')
        ax6.set_xlabel('p')
        ax6.set_ylabel('root means square error of $λ$')
        plt.show()
    return print("Root mean square absolute error of λ1:\n", err1), print("Root mean square absolute error of λ2: \n", err2), print("Root mean square absolute error of λ: \n", err) 
   