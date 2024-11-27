from convergence import *
import numpy as np

def EOC(NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, Neigen, root = None, use_stored_data=True):
    Alpha_list_Lam1 = []
    Alpha_list_Lam2 = []
    Alpha_list_Lam = []
    if use_stored_data == True:
        if not root == None:
            data_array = sio.loadmat( root + '_RMSErr_H' + '.mat')
        else:
            data_array = sio.loadmat('_RMSErr_H' + '.mat')
        Errors_List_Lam1 = data_array['rmserr_lamb1']
        Errors_List_Lam2 = data_array['rmserr_lamb2']
        Errors_List_Lam = data_array['rmserr_lamb']
        NC_list = data_array['NC_list'][0]
        #print(NC_list)
        #print("ELL1", Errors_List_Lam1, Errors_List_Lam1[0,0])
        for i in range(len(NC_list)-1):
            with np.errstate(divide='ignore'):
                EOC_values_Lam1 = (np.log10((Errors_List_Lam1[i,:]/Errors_List_Lam1[i+1,:])))/(np.log10(NC_list[i]/NC_list[i+1]))
                EOC_values_Lam2 = (np.log10((Errors_List_Lam2[i,:]/Errors_List_Lam2[i+1,:])))/(np.log10(NC_list[i]/NC_list[i+1]))
                EOC_values_Lam = (np.log10((Errors_List_Lam[i,:]/Errors_List_Lam[i+1,:])))/(np.log10(NC_list[i]/NC_list[i+1]))
                Alpha_list_Lam1.append(EOC_values_Lam1)
                Alpha_list_Lam2.append(EOC_values_Lam2)
                Alpha_list_Lam.append(EOC_values_Lam)
    else:
        Errors_List_Lam1 , Errors_List_Lam2, Errors_List_Lam, NC_list = convergence(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, reference_solver="FEM", save_files = False, plot=False)
        #print(NC_list)
        for i in range(len(NC_list)-1):
            with np.errstate(divide='ignore'):
                EOC_values_Lam1 = (np.log10((Errors_List_Lam1[i]/Errors_List_Lam1[i+1])))/(np.log10(NC_list[i]/NC_list[i+1]))
                EOC_values_Lam2 = (np.log10((Errors_List_Lam2[i]/Errors_List_Lam2[i+1])))/(np.log10(NC_list[i]/NC_list[i+1]))
                EOC_values_Lam = (np.log10((Errors_List_Lam[i]/Errors_List_Lam[i+1])))/(np.log10(NC_list[i]/NC_list[i+1]))
                Alpha_list_Lam1.append(EOC_values_Lam1)
                Alpha_list_Lam2.append(EOC_values_Lam2)
                Alpha_list_Lam.append(EOC_values_Lam)
            
    return print("Experimental Order of Convergence for λ1: \n",Alpha_list_Lam1), print("Experimental Order of Convergence for λ2: \n", Alpha_list_Lam2),print("Experimental Order of Convergence for λ: \n", Alpha_list_Lam),  print("Set of Coarse elements:\n", NC_list)





