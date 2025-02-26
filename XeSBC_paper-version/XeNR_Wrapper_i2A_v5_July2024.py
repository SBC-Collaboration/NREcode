# -*- coding: utf-8 -*-
"""
Created on July 10 2024

@author: ddurnford

Wrapper function for a fit of a Xe NR study, originally intended for compute canada

-/- NOW WITH CORRECTED NUISANCE PARAMETERS

  June 27, 2024: This is just a copy made for the new fit being done with corrected data/sim files,
					produced by Eric Dahl. 
  July 10, 2024: small update to the files again
  
Instructions:
 - Specify name (directory) of data set, period of fit
 - For stage 1, specify fit parameters below and execute code. Leave stage 2 code commented out.
 - For stage 2, specify fit parameters below, (un)comment lines for (stage 2) stage 1 fit respecitvely, and execute code.
 - If you are initiating a stage 2 fit, leave lines 58-61 uncommented. If continuing a fit, comment them.
"""

#libraries
import numpy as np
import os
import gc

# ------ Specify data set

dirName = 'Dahl_new_July10-2024'
Period = '302'

#----- Archive everything

os.system('cp XeNR_Wrapper_i2A_v5_July2024.py '+dirName+'/')
os.system('cp run_i2A_xenr_mcmc_July2024.sh '+dirName+'/')	#bash script for compute canada
os.system('cp XeSBC_GlobalLikelihood_reparametrization_multi_v5_i2A_July2024.py '+dirName+'/')
os.system('cp XeSBC_runMCMC_i2A_v5_July2024.py '+dirName+'/')

# ---- Stage 1
period = Period+'a'
epoch_steps = 8
bin_number = 250
step_size = 2.
chi2_cap = 1e20

# Launch code
os.system('python3.10 XeSBC_runMCMC_i2A_v5_July2024.py ' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' + 
          str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap))
'''          
# ---- Stage 2

period = Period
epoch_steps = 13
bin_number = 500
step_size = 1.

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
os.system('python3.10 XeSBC_runMCMC_i2A_v5_July2024.py ' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' +
          str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap))
'''
