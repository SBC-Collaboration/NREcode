# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 09:49:18 2020

Code to create simulated boot-strap data files for the PICO NR study

Takes real data set and samples with replacement to create non-parametric
bootstrap data set

Saves data files in the same format as original PICO NR data so that it
can be directly used by PCGL / MCMC code 

@author: DDurnford
"""

# Import libraries
import numpy as np
#import PICOcalGlobalLikelihood_reparametrization_multi_v2 as pcgl
import PICOcalGlobalLikelihood_reparametrization_multi_v2 as pcgl
import SBCcode as sbc
import os
import sys
args = sys.argv
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(13453462)

#prep PCGL code
pcgl.prep(['scratch/NR_Data/'])

# number of experiments
n_exp = len(pcgl.experiment_list)

# --- Function to create boot-strap data sets
# Inputs:  fName = name for simulation directory
# Outputs: Creates directory for data set, sub-directories for each exp
#          Save data files as "data.bin" in each subdirectory
#          Copies simout.bin file
#          Also saves a copy of PCGL code used in the same directory for archival purposes
def BSGen(fName):
    
    #warning if fName is the same as the top directory for the real data
    if fName == pcgl.topdir:
        print('Warning! Trying to over-write existing data')
        return
    
    #get cwd
    cwd = os.getcwd()
    
    #Create directory for results
    os.system('mkdir ' + fName)
    os.chdir(fName)
    
    #save a copy of this code and PCGL code
    os.system('cp ' + pcgl.__file__ + ' ' + pcgl.__file__[len(cwd)+1:-3] + '_copy.py')
    os.system('cp ' + args[0] + ' ' + args[0][len(cwd)+1:-3] + '_copy.py')
    
    #Info file for this data set
    S = 'This boot-strap data set was produced on ' + date.today().strftime("%d/%m/%Y") + ', using PICO NR data taken from '\
    + pcgl.topdir
    text_file = open("MC_info.txt", "w")
    text_file.write(S)
    text_file.close()
    
    # ----------------   Re-sampling   -------------------
    
    
    #Loop over experiments
    for i_exp in range(n_exp):
        
        #make directory for this experiment
        os.system('mkdir ' + pcgl.experiment_list[i_exp])
        
        #copy simulation file
        os.system('cp ' + pcgl.simfile_list[i_exp] + ' ' + pcgl.experiment_list[i_exp] + '/')
        
        #start new dictionary by copying original data file
        dict_exp = pcgl.neutron_data[i_exp].copy()
        
        #change 'counts' key only if there's actually any data (only exception is pico2l_2013_ht)
        if pcgl.neutron_data[i_exp]['E_T'].size > 0:
            
            #loop over thresholds
            for et in range(len(pcgl.neutron_data[i_exp]['E_T'])):
                
                #counts for this threshold
                old_counts = pcgl.neutron_data[i_exp]['counts'][et]
                
                #create re-sampled counts vec
                mult_vec = np.arange(1,pcgl.neutron_data[i_exp]['max_mult'][et]+1)
                new_data = np.random.choice(a=mult_vec,p=old_counts/np.sum(old_counts),size=int(np.sum(old_counts)))
                new_counts,toss = np.histogram(new_data,np.arange(1,pcgl.neutron_data[i_exp]['max_mult'][et]+2)-0.5)
                new_counts = new_counts.astype(int32)                
                
                #replace in dictionary
                if np.shape(old_counts) != np.shape(new_counts):
                    print('Warning! Format issue!')
                    print(np.shape(old_counts))
                    print(np.shape(new_counts))
                    return 0
                    
                dict_exp['counts'][et] = new_counts
        
        #save new data file
        sbc.DataHandling.WriteBinary.WriteBinaryNtupleFile(pcgl.experiment_list[i_exp] + '/data.bin',dict_exp)
        
    #return to original directory
    os.chdir(cwd)
    
    #Done! :)
    return

#==============================================================================
#                                  Plotting code
#==============================================================================
        
#code to load a data set
def makeGOFdata(diro):
    
    experiment_list = ['2013_97','2013_61',
                   # '2013_40',
                   '2014_97',
                   '2014_61',
                   '2014_50',
                   #'2014_34',
                   'pico2l_2013_lt',
                   'pico2l_2013_ht',
                   # 'pico2l_2015',
                   # 'SbBe1',
                   # 'SbBe4',
                   # 'SbBe4_1inPb',
                   'SbBe4_2inPb',
                   #'pico60_Cf_run15',
                   ]

    # Now find where stuff lives
    topdir_searchlocations = [diro]
    
    for topdir in topdir_searchlocations:
        if os.path.isdir(topdir):
            break
    
    datafile_list = [os.path.join(topdir, exp, 'data.bin')
                 for exp in experiment_list]
    neutron_data = [sbc.read_bin(datafile) for datafile in datafile_list]
        
    rate = []
    
    #91 keV
    rate.append(neutron_data[2]['counts'][0].astype(float)) #.ravel()/pcgl.neutron_data[2]['counts'][0][0])
    rate.append(neutron_data[0]['counts'][0].astype(float)) #.ravel()/pcgl.neutron_data[0]['counts'][0][0])
    rate.append(neutron_data[2]['counts'][1].astype(float)) #.ravel()/pcgl.neutron_data[2]['counts'][1][0])
    
    #61 keV
    rate.append(neutron_data[3]['counts'][0].astype(float)) #.ravel()/pcgl.neutron_data[3]['counts'][0][0])
    rate.append(neutron_data[1]['counts'][0].astype(float)) #.ravel()/pcgl.neutron_data[1]['counts'][0][0])
    rate.append(neutron_data[3]['counts'][1].astype(float)) #.ravel()/pcgl.neutron_data[3]['counts'][1][0])   

    #50 keV
    rate.append(neutron_data[4]['counts'][0].astype(float)) #.ravel()/pcgl.neutron_data[4]['counts'][0][0])
    rate.append(neutron_data[4]['counts'][1].astype(float)) #.ravel()/pcgl.neutron_data[4]['counts'][1][0])
    
    #SbBe
    rate.append(neutron_data[7]['counts'][0][1:].astype(float)) #.ravel()/pcgl.neutron_data[7]['counts'][0][0])
    rate.append(neutron_data[7]['counts'][1][1:].astype(float)) #.ravel()/pcgl.neutron_data[7]['counts'][1][0])
    rate.append(neutron_data[7]['counts'][2][1:].astype(float)) #.ravel()/pcgl.neutron_data[7]['counts'][2][0])
    
    #Ambe
    rate.append(neutron_data[5]['counts'][0].astype(float)) #.ravel()/pcgl.neutron_data[5]['counts'][0][0])
    
    return rate
    
def treatTheta(theta):
    
    #re-shapes
    dEpts_reparam = np.reshape(theta[:20], (5,2,2))
    #new array
    dEpts = np.zeros([5,2,2])
    
    #just exp's when reparam_fenceposts == 0
    for i_th in range(2):
        dEpts[0,i_th,:] = np.exp( dEpts_reparam[0,i_th,:])
        dEpts[1:,i_th,:] = np.exp(dEpts_reparam[1:,i_th,:])
    
    #sums up contributions
    Epts = np.cumsum(dEpts, axis=0)
    
    return Epts

#function to extract errors
def getErrorsFull(S,L):
    
    S = S[np.isinf(L)==False]
    L = L[np.isinf(L)==False]

    #get error values    
    Scollect = S[L > (np.max(L) - 1),:]
    
    #BF checkk
    BF = S[np.argmax(L)]
    if BF in Scollect:
        print('yeee!')
    
    #treat parameters to get physical ones
    nstep = len(Scollect[:,0])
    Streat = np.zeros((nstep,34))
    
    for i in range(nstep):
        Streat[i,:20] = treatTheta(Scollect[i,:]).ravel()
        Streat[i,20:] = Scollect[i,20:]
        
    #build minimum bands
    Smin = np.zeros(34)
    Smax = np.zeros(34)
    for j in range(34):
        Smin[j] = np.min(Scollect[:,j])
        Smax[j] = np.max(Scollect[:,j])

    #get extants    
    LE = np.zeros(34)
    UE = np.zeros(34)
    
    #compund errors
    #LE[0:20] = treatTheta(Smin).ravel()
    #UE[0:20] = treatTheta(Smax).ravel()

    #regular errors
    for j in range(34):
        LE[j] = np.min(Streat[:,j])
        UE[j] = np.max(Streat[:,j])

    #Done!
    return LE, UE
    
def pcgl_process(theta, i_exp, whichnuisance=np.ones(pcgl.n_nuisance, dtype=np.bool)):
    ''' Posterior log-likelihod function in this module

        This function inputs parameters and outputs a total
        log-likelihood.  Optional argument whichnuisance is a
        boolean that identifies which nuisance parameters are
        included in the input parameter list theta (the rest
        are held at zero)
        '''
    '''
    #re-shapes
    dEpts_reparam = np.reshape(theta[:20], (5,2,2))
    
    #new array
    dEpts = np.zeros([5,2,2])
    
    #just exp's when reparam_fenceposts == 0
    for i_th in range(2):
        dEpts[0,i_th,:] = np.exp( dEpts_reparam[0,i_th,:])
        dEpts[1:,i_th,:] = np.exp(dEpts_reparam[1:,i_th,:])
    
    #sums up contributions
    Epts = np.cumsum(dEpts, axis=0)
    '''

    xpts = np.reshape(theta[:20], (5,2,2)) / pcgl.threshold_fenceposts[:, np.newaxis]

    eb = np.zeros(pcgl.n_nuisance, dtype=np.float64)
    eb[whichnuisance] = theta[pcgl.n_Epts:]
        
    nu = pcgl.SimulatedCounts(xpts, eb, i_exp,pcgl.eb_1sig)

    return nu

#Code to get BF model
def makeGOFsim(LE,UE):
    
    SL = []
    SU = []
    
    #91 kev
    SL.append(pcgl_process(LE,2)[0])
    SU.append(pcgl_process(UE,2)[0])
    SL.append(pcgl_process(LE,0)[0])
    SU.append(pcgl_process(UE,0)[0])
    SL.append(pcgl_process(LE,2)[1])
    SU.append(pcgl_process(UE,2)[1])
    
    #61 kev
    SL.append(pcgl_process(LE,3)[0])
    SU.append(pcgl_process(UE,3)[0])
    SL.append(pcgl_process(LE,1)[0])
    SU.append(pcgl_process(UE,1)[0])
    SL.append(pcgl_process(LE,3)[1])
    SU.append(pcgl_process(UE,3)[1])
    
    #50 kev
    SL.append(pcgl_process(LE,4)[0])
    SU.append(pcgl_process(UE,4)[0])
    SL.append(pcgl_process(LE,4)[1])
    SU.append(pcgl_process(UE,4)[1])
    
    #SbBe
    SL.append(pcgl_process(LE,7)[0][1:])
    SU.append(pcgl_process(UE,7)[0][1:])
    SL.append(pcgl_process(LE,7)[1][1:])
    SU.append(pcgl_process(UE,7)[1][1:])
    SL.append(pcgl_process(LE,7)[2][1:])
    SU.append(pcgl_process(UE,7)[2][1:])
    
    #AmBe
    SL.append(pcgl_process(LE,5)[0])
    SU.append(pcgl_process(UE,5)[0])
        
    return SU, SL

#Code to make figure
def makeGOFfig():
    
    #directories to look at 
    expList = ['PICO_BS1','PICO_BS2','PICO_BS3','PICO_BS4','PICO_BS5']
    #expList = ['PICO_BS6','PICO_BS7','PICO_BS8','PICO_BS9','PICO_BS10']
    cols = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple']    
    
    #fit results
    #SU, SL = makeGOFsim(LE,UE)
    
    #x coordinates
    xc = np.array([1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33,36,37,38,41,42,45,46,49,50,53,54,55,56,57,58,59])
    xcs = np.linspace(-0.39,0.39,5)    
    
    fig1,ax = plt.subplots(figsize=(18,6))
    
    #Do this once with actual data for scaling
    rateK = makeGOFdata(pcgl.topdir)

    #fix everything
    so = []
    for i in range(len(rateK)):
        m = len(rateK[i])
        for j in range(m):
            so.append(rateK[i][j])#/rateK[i][0])
    so = np.array(so)
    
    xcc = np.arange(0,61)
    so = np.insert(so,0,0)
    so = np.insert(so,4,np.zeros(2))
    so = np.insert(so,9,np.zeros(2))
    so = np.insert(so,14,np.zeros(2))
    so = np.insert(so,19,np.zeros(2))
    so = np.insert(so,24,np.zeros(2))
    so = np.insert(so,29,np.zeros(2))
    so = np.insert(so,34,np.zeros(2))
    so = np.insert(so,39,np.zeros(2))
    so = np.insert(so,43,np.zeros(2))
    so = np.insert(so,47,np.zeros(2))
    so = np.insert(so,51,np.zeros(2))
    so = np.insert(so,60,0)
    
    
    #plt.fill_between(xcc,su,step='mid',edgecolor='darkblue',linewidth=1.5,facecolor='darkblue',alpha=0.25)    
    #plt.fill_between(xcc,su,sl,step='mid',edgecolor='darkblue',linewidth=1.5,alpha=0.55,facecolor='white')    
    #plt.step(xcc,su,where='mid',color='darkblue',linewidth=1.5)    
    #plt.step(xcc,sl,where='mid',color='darkblue',linewidth=1.5) 
    plt.step(xcc,so,where='mid')
    
    #loop over data sets
    for ide in range(len(expList)):
        
        rate = makeGOFdata('PICO_BS/'+expList[ide])

        #fix everything
        Rerr = []
        R = []
        for i in range(len(rate)):
            m = len(rate[i])
            for j in range(m):
                R.append(rate[i][j])# / rateK[i][0])
                if rate[i][j] == 0.:
                    Rerr.append(1.)#/rateK[i][0])
                else:
                    Rerr.append(np.sqrt(rate[i][j]))# / rateK[i][0])
        R = np.array(R)
        Rerr = np.array(Rerr)    
    
        plt.errorbar(xc+xcs[ide],R,yerr=Rerr,linewidth=0,elinewidth=0.6,color=cols[ide],
                     marker='o',markersize=2.3)

    plt.axhline(y=820+14,linewidth=0.82,color='k')
    plt.axhline(y=820-14,linewidth=0.82,color='k')
    plt.yscale('log',nonposy='clip')
    plt.ylim([0.6,3000])
    plt.xlim([0,60])
    #frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    
    plt.axvline(x=14.5,linestyle='--',color='gray',linewidth=1.7)
    plt.axvline(x=29.5,linestyle='--',color='gray',linewidth=1.7)
    plt.axvline(x=39.5,linestyle='--',color='gray',linewidth=1.7)
    plt.axvline(x=51.5,linestyle='--',color='gray',linewidth=1.7)

    #Residual plot
    #frame2=fig1.add_axes((.1,.1,.8,.2))         

    ax.set_xticks([1,2,3,6,7,8,11,12,13,
                   16,17,18,21,22,23,26,27,28,
                   31,32,33,36,37,38,
                   41,42,45,46,49,50,
                   53,54,55,56,57,58,59])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
    
    ax.set_xticklabels(['$1$','$2$','$3+$','$1$','$2$','$3+$','$1$','$2$','$3+$','$1$','$2$','$3+$','$1$','$2$','$3+$','$1$','$2$','$3+$','$1$','$2$','$3+$','$1$','$2$','$3+$','$2$','$3+$','$2$','$3+$','$2$','$3+$','$1$','$2$','$3$','$4$','$5$','$6$','$7+$'])
    
    ax.text((7+0.25)/(60), 0.96, '$\mathrm{97\,keV\;Beam}$',horizontalalignment='center',verticalalignment='center',fontsize=14,transform=ax.transAxes)    
    ax.text(22/(60), 0.96, '$\mathrm{61\,keV\;Beam}$',horizontalalignment='center',verticalalignment='center',fontsize=14,transform=ax.transAxes)    
    ax.text(34.5/(60), 0.96, '$\mathrm{50\,keV\;Beam}$',horizontalalignment='center',verticalalignment='center',fontsize=14,transform=ax.transAxes)    
    ax.text((45+0.25)/(60), 0.96, '$\mathrm{SbBe}$',horizontalalignment='center',verticalalignment='center',fontsize=14,transform=ax.transAxes)    
    ax.text(55.75/(60), 0.96, '$\mathrm{AmBe}$',horizontalalignment='center',verticalalignment='center',fontsize=14,transform=ax.transAxes)
    
    ax.text(2/(60), 0.885, '$\mathrm{3.0\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)
    ax.text(7/(60), 0.885, '$\mathrm{3.2\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes) 
    ax.text(12/(60), 0.885, '$\mathrm{3.6\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)
    
    ax.text(17/(60), 0.885, '$\mathrm{2.9\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)
    ax.text(22/(60), 0.885, '$\mathrm{3.1\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes) 
    ax.text(27/(60), 0.885, '$\mathrm{3.6\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)
    
    ax.text(32/(60), 0.885, '$\mathrm{2.5\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)
    ax.text(37/(60), 0.885, '$\mathrm{3.5\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes) 

    ax.text(41.5/(60), 0.885, '$\mathrm{2.1\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)
    ax.text(45.5/(60), 0.885, '$\mathrm{2.6\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes) 
    ax.text(49.5/(60), 0.885, '$\mathrm{3.2\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes)

    ax.text(56/(60), 0.885, '$\mathrm{3.2\,keV}$',horizontalalignment='center',verticalalignment='center',fontsize=11,transform=ax.transAxes) 

    plt.xlabel('Bubble Multiplicity',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    
    fig1.savefig('BS_Data_Plot.png',dpi=500, bbox_inches='tight')
    plt.show()
    
    #return su,sl
