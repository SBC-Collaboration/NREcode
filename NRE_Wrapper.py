# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:23:30 2020

@author: ddurnford

Wrapper function for a fit of a PICO NR study, originally intended for compute canada

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
dirName = '/home/runze/Documents/analysis/emcee_inputdata/tree/main/XeBC_productionruns/BiBe'
# dirName = 'PICO_MC1'
Period = '36'

#----- Archive everything

os.system('cp NRE_Wrapper.py '+dirName+'/')
os.system('cp run_mcmc.sh '+dirName+'/')	#bash script for compute canada
os.system('cp PICOcalGlobalLikelihood_reparametrization_multi_v2.py '+dirName+'/')
os.system('cp NRE_runMCMC.py '+dirName+'/')

# ---- Stage 1
period = Period+'a'
epoch_steps = 5
bin_number = 100
step_size = 2.
chi2_cap = 1e20

# Launch code
os.system('python3.5 NRE_runMCMC.py ' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' + 
          str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap))
"""        
# ---- Stage 2

period = Period
epoch_steps = 10
bin_number = 500
step_size = 1.2

#Load first stage to get reasonable cap
LD = np.loadtxt('Epoch_storage/Period'+period+'a_logProb.txt')
chi2_cap = np.max(LD)+4.
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
os.system('python3.5 PCGL_runMCMC.py PICO_VMC/' + dirName + ' ' + str(period) + ' ' + str(epoch_steps) + ' ' +
          str(bin_number) + ' ' + str(step_size) + ' ' + str(chi2_cap))

"""
