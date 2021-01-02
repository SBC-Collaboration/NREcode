# -*- coding: utf-8 -*-
"""
Created on Dec 12 2020

Code to run MCMC (with fast-burn in) for PICO NR study, WIMP sensitivity
version with "horizontal re-seeding"

parallelization done with python library Multiprocessing

Inputs are (in order):
 - directory to find data in
 - Period of MCMC run
 - epoch_nstep
 - bin_number
 - stepsize
 - chi2 hard cap
 - WIMP mass

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
import scipy.io as sio
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

# number of dimensions to consider for WIMP recasting
mDim = 8

#------ Initial Guess

# BF from Period 34 (ddurnford fit of PICO data)
guess_theta = np.array([ 1.65750550e+00,  1.19668186e+00,  1.66530667e+00,  1.27574295e+00, -2.82076273e+00, -2.71818698e+00, -3.01324190e+00, -1.88755528e+00,1.66976041e+00, -5.64587118e+00,  1.75194971e+00, -5.41992168e+00,6.43072211e-01, -5.24568677e-01,  3.59527604e-01, -6.14857566e-01,-4.19287206e-01,  7.85916476e-01,  4.71423407e-02,  1.75578191e+00,5.53690885e-03, -3.31378126e-01,  3.86920360e-01,  1.09323458e+00,-7.06982858e-02, -1.43923824e+00,  8.82628498e-01,  2.78938373e-01,-7.56704066e-01,  9.73561639e-01,  6.23926470e-01, -2.66908442e-01,-1.10396359e+00, -5.22685251e-02])

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
    
def calcVol2(S,L):
    ''' This calculates the "1-sigma volume" contained by explored mcmc samples
    
    New, simpler version with no additive constant required, although it does 
    allow for the volume to decrease from one epoch to the next

    Inputs: S, samples of mcmc
			L, log_prob values
			additive =  positive additive constant to keep volumes > 0

    Outputs: "volume"
    '''

    #select 1 sigma samples    
    Ss = S[L > np.max(L)-1]

    #number of dimensions
    nD = np.shape(Ss)[1]
    
    #initialize volume
    vol = 0.
    
    #for each dimension, add up range subtended by 1 sigma samples
    for i in range(nD):
        vol += (np.max(Ss[:,i]) - np.min(Ss[:,i]))
        
    return vol
    
#-------- Some constants for WIMP sensitivity
SI_denominator = 3*12.011 + 8*18.998403163
SI_C_numerator = 3*12.011
SI_F_numerator = 8*18.998403163
mass_array = np.array([1.5849e+00,   1.7378e+00,   1.9055e+00,   2.0893e+00,   2.2909e+00,   2.5119e+00,   2.7542e+00, 
   3.0200e+00,   3.3113e+00,   3.6308e+00,   3.9811e+00,   4.3652e+00,   4.7863e+00,   5.2481e+00,   5.7544e+00,
   6.3096e+00,   6.9183e+00,   7.5858e+00,   8.3176e+00,   9.1201e+00,   1.0000e+01,   1.0965e+01,   1.2023e+01,
   1.3183e+01,   1.4454e+01,   1.5849e+01,   1.7378e+01,   1.9055e+01,   2.0893e+01,   2.2909e+01,   2.5119e+01,
   3.1623e+01,   3.9811e+01,   5.0119e+01,   6.3096e+01,   7.9433e+01,   1.0000e+02,   1.2589e+02,   1.5849e+02,
   1.9953e+02,   2.5119e+02,   3.1623e+02,   1.0000e+03,   3.1623e+03,   1.0000e+04,   3.1623e+04, 1.0000e+05])

bin_length = 13001
bin_width = 0.01
which_mass = np.zeros(mass_array.shape[0], dtype=np.bool)

#-------- Load WIMP spectra and define WIMP masses 
# (taken from chimera:/home/mjn693/Documents/LL/Python_objects/)
WIMPspectra_production = sio.loadmat('WIMPspectra_production.mat')

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
bin_number = 40
ntemps = 1                     # historical, kept for formatting reasons
num_walkers = 100              # Number of walkers for initial start
stepsize = float(args[5])      # 2 for faster exploration, 1.2 for fine tuning
nw_i = num_walkers
nd_i = ndim

#reset to more reasonable value
pcgl.chisq_hard_cap = float(args[6])

# WIMP mass and interaction type
wimp_mass = float(args[7])
int_type = args[8]

#determine mass index
mass_index = np.argmin(np.abs(wimp_mass - mass_array))
if np.abs(mass_array[mass_index] - wimp_mass) > 0.25:
    print('Warning! No close WIMP mass in table!')
    exit()

# Number of CPUs to use (#8 by default)
nCPU = 10

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

def treatTheta(theta):
    ''' This converts direct sample input into efficiency curve points

    Inputs: theta
    Outputs: Epts
    '''
    
    # re-shapes
    dEpts_reparam = np.reshape(theta[:20], (5,2,2))
    
    # new array
    dEpts = np.zeros([5,2,2])
    
    # just exp's when reparam_fenceposts == 0
    for i_th in range(2):
        dEpts[0,i_th,:] = np.exp( dEpts_reparam[0,i_th,:])
        dEpts[1:,i_th,:] = np.exp(dEpts_reparam[1:,i_th,:])
    
    # sums up contributions
    Epts = np.cumsum(dEpts, axis=0)
    
    return Epts

def wimp_treat(theta,mass_index,int_type):
    ''' Calculates WIMP sensitivity for given theta, mass interaction type

    Inputs: theta, wimp mass index, interaction type (sting)
    Outputs: 8 WIMP sensitivity combinations (see Jin's thesis)
    '''
    
    # Treat theta
    thetaT = treatTheta(theta).ravel()
    
    # extract C and F points at both thresholds
    C_245 = thetaT[::4]
    F_245 = thetaT[1::4]
    C_329 = thetaT[2::4]*(3.29/3.)
    F_329 = thetaT[3::4]*(3.29/3.)
    
    # create interpolation of efficiency curves
    C_interp_245 = np.interp(1.0 + np.arange(0,bin_length*bin_width,bin_width) ,C_245,[0, .2, .5, .8, 1.0])
    F_interp_245 = np.interp(1.0 + np.arange(0,bin_length*bin_width,bin_width) ,F_245,[0, .2, .5, .8, 1.0])
    C_interp_329 = np.interp(1.0 + np.arange(0,bin_length*bin_width,bin_width) ,C_329,[0, .2, .5, .8, 1.0])
    F_interp_329 = np.interp(1.0 + np.arange(0,bin_length*bin_width,bin_width) ,F_329,[0, .2, .5, .8, 1.0])
    
    # what interaction type? For SD...
    if int_type == 'SD':
        
        # get rate for fluorine only (for SD)
        drde_f = WIMPspectra_production['SD_F_table'][mass_index,:]
        
        # calculate WIMP sensitivity for both thresholds
        WS_245 = np.sum(F_interp_245*drde_f)*bin_width
        WS_329 = np.sum(F_interp_329*drde_f)*bin_width
        
    # For SI...
    elif int_type == 'SI':
        
        # get rate for fluorine and carbon
        drde_f = WIMPspectra_production['SI_F_table'][mass_index,:]
        drde_c = WIMPspectra_production['SI_C_table'][mass_index,:]
        
        # calculate WIMP sensitivity for both thresholds
        WS_245 = ((SI_F_numerator/SI_denominator * np.sum(F_interp_245*drde_f)) + (SI_C_numerator/SI_denominator * np.sum(C_interp_245* drde_c))) * bin_width
        WS_329 = ((SI_F_numerator/SI_denominator * np.sum(F_interp_329* drde_f)) + (SI_C_numerator/SI_denominator * np.sum(C_interp_329* drde_c))) * bin_width
        
    # invalid interaction type
    else:
        print('Invalid interaction type!')
        exit()
        
    # 8 combinations of variables
    linear_combs = np.array([WS_329,-WS_329,-WS_245,WS_245,WS_245+WS_329,WS_245-WS_329,-WS_245-WS_329,-WS_245+WS_329])    
        
    # Done!
    return linear_combs
    
# -----------------------------------------------------------------------------

# Set up initial starting point
epoch_starting_points = np.zeros((num_walkers,ndim))
if load_epoch == True:

    # load files
    samples_file = storeDir + '/Period'+str(Period)+'_samples.txt'
    log_prob_file = storeDir + '/Period'+str(Period)+'_logProb.txt'
    wimp_file = storeDir + '/Period'+str(Period)+'_wimp.txt'
    lt = storeDir + '/Period'+str(Period)+'_state'
    samples = np.loadtxt(samples_file)
    log_prob = np.loadtxt(log_prob_file)
    wimp_samples = np.loadtxt(wimp_file)
    epoch_starting_points = np.load(lt)[-1,:,:]
    nw_i = len(epoch_starting_points[:,0])
    nd_i = len(epoch_starting_points[0,:])
    
    # determine last epoch
    prog_file = storeDir + '/Period' + str(Period) + '_progress.txt'
    prog = np.loadtxt(prog_file)
    epoch_hist,maxL_list,vol_list,nw_list = np.hsplit(prog,4)
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
        wimp_samples = np.array([])

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
    wimp_file = storeDir + '/Period'+str(Period)+'_wimp.txt'
    state_file = storeDir + '/Period'+str(Period)+'_state'

    #----- Load old files

    if os.path.exists(samples_file) == False and os.path.exists(log_prob_file) == False:
        samples = np.zeros((1,ndim))
        log_prob = np.zeros(1)-1e100
        wimp_samples = np.zeros((1,mDim))
    else:
        samples = np.loadtxt(samples_file)
        log_prob = np.loadtxt(log_prob_file)
        wimp_samples = np.loadtxt(wimp_file)
    
    #----- New data and concat

    samples_epoch = sampler.get_chain(flat = True)
    log_prob_epoch = sampler.get_log_prob(flat = True)
    
    # ---- wimp treatment

    nSamples_epoch = np.shape(samples_epoch)[0]
    wimp_samples_epoch = np.zeros((nSamples_epoch,mDim))
    for j in range(nSamples_epoch):
        wimp_samples_epoch[j,:] = wimp_treat(samples_epoch[j,:],mass_index,int_type)
    
    wimp_samples = np.concatenate((wimp_samples,wimp_samples_epoch))
    samples = np.concatenate((samples,samples_epoch))
    log_prob = np.concatenate((log_prob,log_prob_epoch))

    #----- Cut from new max
    if i_epoch > 10:
        maxL = np.max(log_prob)
        samples = samples[log_prob > maxL - 4,:]
        wimp_samples = wimp_samples[log_prob > maxL - 4,:]
        log_prob = log_prob[log_prob > maxL - 4]

    #----- save progress
    
    np.savetxt(samples_file, samples, fmt = '%1.30e')
    np.savetxt(log_prob_file, log_prob, fmt = '%1.30e')
    np.savetxt(wimp_file, wimp_samples, fmt = '%1.30e')
    pickle.dump(sampler.get_chain(), open( state_file, "wb" ))
    
    #----- reset and build starting array (re-seeding)

        #----- reset and build starting array (re-seeding)

    # bin edges for horizontal FBI bins on WIMP parameters, with small offset
    bin_edges = np.unique(np.concatenate((np.linspace(maxL,maxL-1.,22),np.linspace(maxL-1.,maxL-4.,21)))[::-1]) + 0.025
    
    # if no non-infinite samples found or bin edges are degenerate, just create
    # starting points from initial guess with small random offsets
    if np.isinf(maxL) == True or len(bin_edges) < 2 or maxL == -2.e+100:
        
        # initialize starting points
        epoch_starting_points = np.zeros((num_walkers,ndim))

        # loop through initial number of walkers
        for j in range(nw_i):
            epoch_starting_points[j,:] = guess_theta+np.random.normal(0.,0.01,size = ndim)
    
    # otherwise, proceed to re-binning procedure
    else:
        
        # initialize starting points
        epoch_starting_points = np.zeros(ndim)
        starting_index = []
        
        # loop through (wimp) dimensions
        for i_dim in range(mDim):
            
            # loop through bins
            for i_bin in range(bin_number):
                
                # selection for this bin
                cut = (log_prob_epoch >= bin_edges[i_bin]) & (log_prob_epoch < bin_edges[i_bin+1])
                
                # if bin is non-empty...
                if len(np.unique(cut)) > 1:
                
                    # find min and max points in this dimension
                    point_a = np.argmin(wimp_samples_epoch[cut,i_dim])
                    point_b = np.argmax(wimp_samples_epoch[cut,i_dim])
                
                    # add to starting points if unique, otherwise add instance of initial guess to avoid too-few-walker error
                    if point_a not in starting_index:
                        starting_index.append(point_a)
                        epoch_starting_points = np.vstack((epoch_starting_points,samples_epoch[point_a,:]))
                    else:
                        epoch_starting_points = np.vstack((epoch_starting_points,guess_theta+np.random.normal(0.,0.01,size = ndim)))
                    if point_b not in starting_index:
                        starting_index.append(point_b)
                        epoch_starting_points = np.vstack((epoch_starting_points,samples_epoch[point_b,:]))
                    else:
                        epoch_starting_points = np.vstack((epoch_starting_points,guess_theta+np.random.normal(0.,0.001,size = ndim)))

                # otherwise add two instances of initial guess to avoid too-few-walker error
                else:
                    epoch_starting_points = np.vstack((epoch_starting_points,guess_theta+np.random.normal(0.,0.01,size = ndim)))
                    epoch_starting_points = np.vstack((epoch_starting_points,guess_theta+np.random.normal(0.,0.01,size = ndim)))
                 
        # delete empty initialization row
        epoch_starting_points = np.delete(epoch_starting_points,0,axis=0)
        
        # reshape
        if epoch_starting_points.shape[0]%2 == 1:
            epoch_starting_points = np.insert(epoch_starting_points,0, epoch_starting_points[0,:],axis = 0)   
        epoch_starting_points = np.expand_dims(epoch_starting_points,axis=0)
        
    #--- calculate volume

    vol_epoch = calcVol2(wimp_samples,log_prob)	 
    
    #--- save volume and maxL progress

    #load old results
    prog_file = storeDir + '/Period' + str(Period) + '_progress.txt'
    
    if os.path.exists(prog_file) == False:
        epoch_hist = np.array([])
        maxL_list = np.array([])
        vol_list = np.array([])
        nw_list = np.array([])
    else:
        prog = np.loadtxt(prog_file)
        epoch_hist,maxL_list,vol_list,nw_list = np.hsplit(prog,4)
        if len(epoch_hist) > 1:
            epoch_hist = epoch_hist[:,0]
            maxL_list = maxL_list[:,0]
            vol_list = vol_list[:,0]
            nw_list = nw_list[:,0]
    
    #add new results
    vol_list = np.concatenate((vol_list,np.array([vol_epoch])))
    maxL_list = np.concatenate((maxL_list,np.array([maxL])))
    epoch_hist = np.concatenate((epoch_hist,np.array([i_epoch]))) 
    nw_list = np.concatenate((nw_list,np.array([nw_i])))
    
    #volume trend
    vol_diff = (vol_list[1:] - vol_list[0:-1])/vol_list[0:-1]
    
    #save file
    np.savetxt(prog_file,np.array([epoch_hist,maxL_list,vol_list,nw_list]).T)
    
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