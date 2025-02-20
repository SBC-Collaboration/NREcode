# -*- coding: utf-8 -*-
"""
Created on July 23 2021

Code to run MCMC (with fast-burn in) for XeSBC NR study

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
import XeSBC_GlobalLikelihood_reparametrization_multi_v4 as xegl
from matplotlib.tri import LinearTriInterpolator
from matplotlib.tri import Triangulation
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
which_nuisance = np.array([np.ones(xegl.n_nuisance,dtype = np.bool)])
dim_nuisance = np.sum(which_nuisance)

# number of thresholds
num_threshold = xegl.threshold_fenceposts.size

# number of species
num_elements = 1

# number of parameters in the model
ndim = 5*num_threshold + dim_nuisance

#------ Initial Guess

# need something!
#guess_theta = np.array([-0.10536052,  0.69314718, -1.51412773, -1.60943791, -1.10866262,
#       -1.2039728 , -1.10866262, -1.2039728 , -1.51412773, -1.60943791,
#        0.        ,  0.        ,  0.        ,  0.        ,  0.        ])

# Guess from Period 0 mcmc run
guess_theta = np.array([ 0.12530298,  1.59249128, -1.01141287,  0.11066013, -0.42281489,-1.96481931, -1.35529153, -1.45358755, 0.86736413, -3.86214427,-0.02433887,  0.18126786, -0.02606193, -0.16020627,  0.3269579 ])


#-------- Volume calculation
      
# reasonable bounds for volume calcuation

def calcVol2(S,L):
    #This calculates the "1-sigma volume" contained by explored mcmc samples
    #
    #Inputs: S, samples of mcmc
			 #L, log_prob values
    #
    #Outputs: "volume"
    

    #number of dimensions
    ndim = S.shape[1]

    #initialize
    v = 0.
    
    #main loop
    for i in range(ndim):
        
        # calculate span of 1-sigma samples
        va = np.min(S[L>=np.max(L)-1,i])
        vb = np.max(S[L>=np.max(L)-1,i])
        
        # safety NaN condition
        if np.isnan(va) == True or np.isnan(vb) == True:
            v += 0
        else:
            v += vb - va
                
    return v

#-------- Production run Parameters

# What data to look at?
topdir = args[1]

# Period for MCMC run
Period = args[2]

print('------ Period ' + Period + ' ------')

#Prep xegl code
xegl.prep([topdir])

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
xegl.chisq_hard_cap = float(args[6])

# Number of CPUs to use (#8 by default)
nCPU = 15

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
nEvasive = 0

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
        epoch_starting_points[j,:] = guess_theta+np.random.normal(0.,0.1,size = ndim)
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
    print('strike '+str(strike))
    print('',flush=True)
    
    # Set up multiprocessing
    with Pool(processes = nCPU) as pool:
        
        #set up sampler
        #note that "threads" option does nothing when using pool
        sampler = emcee.EnsembleSampler(nw_i, nd_i, xegl.XeSBCcal_post,a=stepsize,
                                   pool=pool)

        #run MCMC for this epoch
        try:
            if np.shape(epoch_starting_points.shape) == (3,):
                result = sampler.run_mcmc(epoch_starting_points[0], epoch_nstep)
            else:
                result = sampler.run_mcmc(epoch_starting_points, epoch_nstep)
        except:
            print('///////////////////////// Evasive! ///////////////////////////')
            if np.shape(epoch_starting_points.shape) == (3,):
                result = sampler.run_mcmc(epoch_starting_points[0]*np.random.uniform(-0.1,0.1,epoch_starting_points[0].shape), epoch_nstep)
            else:
                result = sampler.run_mcmc(epoch_starting_points*np.random.uniform(-0.1,0.1,epoch_starting_points.shape), epoch_nstep)
            nEvasive += 1
            if nEvasive >= 3:
                break

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

    vol_epoch = calcVol2(samples,log_prob)	#additive constant from historical PICO NR fit
    
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
        if maxL - max0 >= 0. and maxL - max0 < 0.01 and vol_diff[-1] < 0.001 and vol_list[-1] > 0.:
            strike += 1
        else:	# if progress increases again, remove strike
            strike += -5
            strike = max(strike,0)
        max0 = maxL

	#require at least 150 epochs and 25 strikes to terminate
    if strike > 25 and i_epoch >= 400:
        break
