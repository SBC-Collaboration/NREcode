# -*- coding: utf-8 -*-
"""
Created on Aug 28 2024

@author: ddurnford

Wrapper to run MC fits, XeSBC study from July 2024, 2 threshold fenceposts
"""

# libraries
import numpy as np
import os
import gc
import sys
args = sys.argv

# Period (MC iter)
MC_iter = int(args[1])

# stage of fit
fit_stage = int(args[2])

# ------ Specify data set and base period

dirName = 'XeNR_MCstudy_Aug2024/XeNR_MCstudy_Aug2024_'+str(MC_iter)
Period = '302MC_'+str(MC_iter)

#----- Archive everything

os.system('cp XeNR_Wrapper_i2A_v5_July2024_MC.py '+dirName+'/')
os.system('cp run_i2A_xenr_mcmc_July2024_MC.sh '+dirName+'/')	#bash script for compute canada
os.system('cp XeSBC_GlobalLikelihood_reparametrization_multi_v5_i2A_July2024.py '+dirName+'/')
os.system('cp XeSBC_runMCMC_i2A_v5_July2024_MC.py '+dirName+'/')

# what to do depending on fit stage
# 1 = start phase a
# 2 = restart phase a
# 3 = start phase b
# 4 = restart phase b

if fit_stage == 1 or fit_stage == 2:
    
    # ---- Stage 1
    period = Period+'a'
    epoch_steps = 8
    bin_number = 250
    step_size = 2.
    chi2_cap = 1e20
    epoch_limit = 100
    
    # Launch code
    os.system('python3.10 XeSBC_runMCMC_i2A_v5_July2024_MC.py ' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' + 
              str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap) + ' ' + str(epoch_limit))

elif fit_stage == 3:
    
    # ---- Stage 2

    period = Period
    epoch_steps = 13
    bin_number = 500
    step_size = 1.
    epoch_limit = 250
    
    #Load first stage to get reasonable cap
    LD = np.loadtxt('Epoch_storage/Period'+period+'a_logProb.txt')
    chi2_cap = np.max(LD)+100.
    del LD
    gc.collect()
    
    ### --- WARNING! Only use if beginning stage2, otherwise comment out these 4 lines
    os.system('cp Epoch_storage/Period'+period+'a_logProb.txt Epoch_storage/Period'+period+'b_logProb.txt')
    os.system('cp Epoch_storage/Period'+period+'a_state Epoch_storage/Period'+period+'b_state')
    os.system('cp Epoch_storage/Period'+period+'a_samples.txt Epoch_storage/Period'+period+'b_samples.txt')
    os.system('cp Epoch_storage/Period'+period+'a_progress.txt Epoch_storage/Period'+period+'b_progress.txt')
    
    period = Period + 'b'
    print(period)

    # Launch code
    os.system('python3.10 XeSBC_runMCMC_i2A_v5_July2024_MC.py ' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' +
              str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap) + ' ' + str(epoch_limit))

elif fit_stage == 4:
    
    # ---- Stage 2

    period = Period
    epoch_steps = 13
    bin_number = 500
    step_size = 1.
    epoch_limit = 250
    
    #Load first stage to get reasonable cap
    LD = np.loadtxt('Epoch_storage/Period'+period+'a_logProb.txt')
    chi2_cap = np.max(LD)+100.
    del LD
    gc.collect()
    
    period = Period + 'b'
    print(period)

    # Launch code
    os.system('python3.10 XeSBC_runMCMC_i2A_v5_July2024_MC.py ' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' +
              str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap) + ' ' + str(epoch_limit))
