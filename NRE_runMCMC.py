# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:02:18 2020

Code to run MCMC (with fast-burn in) for PICO NR study

parallelization done with python library Multiprocessing

Inputs are (in order):
 - directory to find data in
 - Period of MCMC run
 - epoch_nstep
 - bin_number
 - stepsize
 - chi2 hard cap

@author: DDurnford
"""

# libraries
import emcee
import numpy as np
import PICOcalGlobalLikelihood_reparametrization_multi_v2 as pcgl
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
np.load.__defaults__=(None, True, True, 'ASCII')
import pickle
from scipy.stats import binned_statistic
import sys
args = sys.argv
np.random.seed(42)

# include all nuisance parameters
which_nuisance = np.array([np.ones(pcgl.n_nuisance,dtype = np.bool)])
dim_nuisance = np.sum(which_nuisance)

# number of thresholds
num_threshold = pcgl.threshold_fenceposts.size

# number of species
num_elements = 2

# number of parameters in the model
ndim = 10*num_threshold + dim_nuisance

#------ Initial Guess

# BF from Period 34 (ddurnford fit of PICO data)
guess_theta = np.array([ 1.65750550e+00,  1.19668186e+00,  1.66530667e+00,  1.27574295e+00,
       -2.82076273e+00, -2.71818698e+00, -3.01324190e+00, -1.88755528e+00,
        1.66976041e+00, -5.64587118e+00,  1.75194971e+00, -5.41992168e+00,
        6.43072211e-01, -5.24568677e-01,  3.59527604e-01, -6.14857566e-01,
       -4.19287206e-01,  7.85916476e-01,  4.71423407e-02,  1.75578191e+00,
        5.53690885e-03, -3.31378126e-01,  3.86920360e-01,  1.09323458e+00,
       -7.06982858e-02, -1.43923824e+00,  8.82628498e-01,  2.78938373e-01,
       -7.56704066e-01,  9.73561639e-01,  6.23926470e-01, -2.66908442e-01,
       -1.10396359e+00, -5.22685251e-02])

#-------- Volume calculation
      
# reasonable bounds for volume calcuation
binsa = np.array([ 1.01808316,  0.89609191,  1.29266798,  1.16315096, -3.88617265,
       -3.64865946, -5.60787692, -3.18800453,  0.36706077, -7.83267239,
        0.81973171, -8.1652399 , -0.59245043, -2.89515001, -0.07374429,
       -2.70995565, -1.58162291, -0.91317244, -2.98916088, -1.78958249,
       -0.75211146, -1.44435034, -0.60465208,  0.6712873 , -1.08475804,
       -2.42844962, -0.26551765, -0.74018606, -1.62686749,  0.2526427 ,
       -0.36140405, -1.30059274, -2.05057406, -0.21927138])
binsb = np.array([ 2.56330499,  1.23492372,  2.56346639,  1.46296621, -0.78377603,
        0.16873003, -2.05195839, -0.66289017,  2.34041311, -2.87832399,
        3.90205553, -4.91489277,  1.72977452,  0.20070191,  2.24981077,
        0.75238084,  2.00114598,  2.08220374,  0.81442556,  2.24036402,
        1.11866961,  0.21818037,  1.73594775,  2.0517152 ,  0.50993029,
       -0.87082394,  0.92066029,  1.26558695, -0.06077413,  1.63325533,
        1.52532272,  0.80405223,  0.06672319,  0.05886753])
        
def calcVol(S,L,additive = 100):
    ''' This calculates the "1-sigma volume" contained by explored mcmc samples

    Inputs: S, samples of mcmc
			L, log_prob values
			additive =  positive additive constant to keep volumes > 0

    Outputs: "volume"
    '''

	#number of dimensions
    ndim = S.shape[1]

	#initialize
    v = 0.
    nb = 60
    
    #main loop
    for i in range(ndim):
        
        maxi,edd,indi = binned_statistic(S[:,i],L,'max',nb,(binsa[i],binsb[i]))
        bc = edd[0:-1] + 0.5*(edd[1]-edd[0])
        if all(np.isnan(maxi)) == True:
            continue
        maxi[np.isnan(maxi) == True] = np.min(maxi[np.isnan(maxi)==False])

        v += np.trapz(maxi - additive,bc)
                
    return v

#-------- Production run Parameters

# What data to look at?
topdir = args[1]

# Period for MCMC run
Period = args[2]

print('------ Period ' + Period + ' ------')

#Prep PCGL code
pcgl.prep([topdir])

# storage directory for MCMC
storeDir = 'Epoch_storage'

# MCMC parameters
epoch_nstep = int(args[3])     # how many steps per epoch (5 for rough, 10 for fine)
bin_number = int(args[4])      # 100 for rough, 500 for fine
ntemps = 1                     # historical, kept for formatting reasons
num_walkers = 100              # Number of walkers (just leave it at 100)
stepsize = float(args[5])      # 2 for faster exploration, 1.2 for fine tuning
nw_i = num_walkers
nd_i = ndim

#reset to more reasonable value
pcgl.chisq_hard_cap = float(args[6])

# Number of CPUs to use (#8 by default)
nCPU = 8

# load from existing epoch?
state_file = storeDir + '/Period'+str(Period)+'_state'
if os.path.exists(state_file) == True:
    load_epoch = True
else:
    load_epoch = False

# initialize convergence criteria
max0 = -1e100
maxL = -2e100
strike = 0

# Set up initial starting point
epoch_starting_points = np.zeros((num_walkers,ndim))
if load_epoch == True:

    samples_file = storeDir + '/Period'+str(Period)+'_samples.txt'
    log_prob_file = storeDir + '/Period'+str(Period)+'_logProb.txt'
    lt = storeDir + '/Period'+str(Period)+'_state'
    samples = np.loadtxt(samples_file)
    log_prob = np.loadtxt(log_prob_file)
    epoch_starting_points = np.load(lt)[-1,:,:]
    nw_i = len(epoch_starting_points[:,0])
    nd_i = len(epoch_starting_points[0,:])
    
    # determine last epoch
    prog_file = storeDir + '/Period' + str(Period) + '_progress.txt'
    prog = np.loadtxt(prog_file)
    epoch_hist,maxL_list,vol_list = np.hsplit(prog,3)
    if len(epoch_hist) > 1:
        epoch_hist = epoch_hist[:,0]
    epoch_last = int(epoch_hist[-1])
    
    # List of Epochs to run
    epoch_list=np.arange(epoch_last + 1,1000)

else:

    # List of Epochs to run
    epoch_list=np.arange(0,1000)

    for j in range(nw_i):
        epoch_starting_points[j,:] = guess_theta+np.random.normal(0.,0.001,size = ndim)
        samples = np.array([])
        log_prob = np.array([])

# Launch production run!
for i_epoch in epoch_list:
    
    #reset sampler    
    if 'sampler' in globals():
        sampler.reset()
    
    #printout
    print('   --- Epoch '+str(i_epoch)+', Period '+str(Period)+' ---')
    print(' # of walkers = '+str(np.shape(epoch_starting_points)),flush=True)
    print('',flush=True)
    
    # Set up multiprocessing
    with Pool(processes = nCPU) as pool:
        
        #set up sampler
        #note that "threads" option does nothing when using pool
        sampler = emcee.EnsembleSampler(nw_i, nd_i, pcgl.PICOcalLL_post,a=stepsize,
                                   args=(which_nuisance),pool=pool)

        #run MCMC for this epoch
        if np.shape(epoch_starting_points.shape) == (3,):
            result = sampler.run_mcmc(epoch_starting_points[0], epoch_nstep)
        else:
            result = sampler.run_mcmc(epoch_starting_points, epoch_nstep)
    
    #----- File names
    
    samples_file = storeDir + '/Period'+str(Period)+'_samples.txt'
    log_prob_file = storeDir + '/Period'+str(Period)+'_logProb.txt'
    state_file = storeDir + '/Period'+str(Period)+'_state'

    #----- Load old files

    if os.path.exists(samples_file) == False and os.path.exists(log_prob_file) == False:
        samples = np.zeros((1,ndim))
        log_prob = np.zeros(1)-1e100   
    else:
        samples = np.loadtxt(samples_file)
        log_prob = np.loadtxt(log_prob_file)
    
    #----- New data and concat

    samples_epoch = sampler.get_chain(flat = True)
    log_prob_epoch = sampler.get_log_prob(flat = True)
    
    samples = np.concatenate((samples,samples_epoch))
    log_prob = np.concatenate((log_prob,log_prob_epoch))

    #----- Cut from new max
    if i_epoch > 10:
        maxL = np.max(log_prob)
        samples = samples[log_prob > maxL - 4,:]
        log_prob = log_prob[log_prob > maxL - 4]

    #----- save progress
    
    np.savetxt(samples_file, samples, fmt = '%1.30e')
    np.savetxt(log_prob_file, log_prob, fmt = '%1.30e')
    pickle.dump(sampler.get_chain(), open( state_file, "wb" ))

    # reset and build starting array
    epoch_starting_points = np.zeros(ndim)    
        
    for i_dim in range(ndim):
        for i_bin in range(bin_number):
            
            b = np.max(sampler.chain[:,:,i_dim])
            a = np.min(sampler.chain[:,:,i_dim])
            
            bin_size = (b-a)/bin_number
            
            index = np.asarray(np.where((sampler.chain[:,:,i_dim] >= a + i_bin*bin_size) & (sampler.chain[:,:,i_dim] <a + (i_bin+1)*bin_size)))
            unique, unique_indices = np.unique(sampler.lnprobability[index[0,:],index[1,:]],return_index=True)
            
            if unique.size != 0:    
                epoch_starting_points = np.vstack((epoch_starting_points,sampler.chain[index[0,unique_indices[-1]],index[1,unique_indices[-1]]])) 
             
    epoch_starting_points = np.delete(epoch_starting_points,0,axis=0)
    
    if epoch_starting_points.shape[0]%2 == 1:
        epoch_starting_points = np.insert(epoch_starting_points,0, epoch_starting_points[0,:],axis = 0)
        
    epoch_starting_points = np.expand_dims(epoch_starting_points,axis=0)
    
    #--- calculate volume

    vol_epoch = calcVol(samples,log_prob,-58.51266139248701)	#additive constant from historical PICO NR fit
    
    #--- save volume and maxL progress

    #load old results
    prog_file = storeDir + '/Period' + str(Period) + '_progress.txt'
    
    if os.path.exists(prog_file) == False:
        epoch_hist = np.array([])
        maxL_list = np.array([])
        vol_list = np.array([])
    else:
        prog = np.loadtxt(prog_file)
        epoch_hist,maxL_list,vol_list = np.hsplit(prog,3)
        if len(epoch_hist) > 1:
            epoch_hist = epoch_hist[:,0]
            maxL_list = maxL_list[:,0]
            vol_list = vol_list[:,0]
    
    #add new results
    vol_list = np.concatenate((vol_list,np.array([vol_epoch])))
    maxL_list = np.concatenate((maxL_list,np.array([maxL])))
    epoch_hist = np.concatenate((epoch_hist,np.array([i_epoch])))    
    
    #volume trend
    vol_diff = (vol_list[1:] - vol_list[0:-1])/vol_list[0:-1]
    
    #save file
    np.savetxt(prog_file,np.array([epoch_hist,maxL_list,vol_list]).T)
    
    #--- print out progress
    print('',flush=True)
    print('Max logL was '+str(maxL))
    print('Vol was '+str(vol_epoch))
    print('',flush=True)
    nw_i = epoch_starting_points.shape[-2]
    
    #--- Convergence criteria ----------------

	#has to be at least 1 epoch
    if i_epoch > 0:

		# add 1 "strike" if progress (in maxL and volume) is negligible
        if maxL - max0 >= 0. and maxL - max0 < 0.01 and vol_diff[-1] < 0.001:
            strike += 1
        else:	# if progress increases again, remove strike
            strike += -1
            strike = np.max(strike,0)
        max0 = maxL

	#require at least 150 epochs and 25 strikes to terminate
    if strike > 25 and i_epoch >= 150:
        break
