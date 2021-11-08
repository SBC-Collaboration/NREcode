#libraries
import numpy as np
import SBCcode as sbc
import os
import warnings

''' This module calculates goodness of fit for the global
    PICO calibration set

    The key function to be called (e.g. from emcee) is
    PICOcalChisq(theta).  Parameters defining the fit are
    all present in the introductory block that is executed
    when the module is imported.
'''


''' The following code is run when the library is imported -- it sets
    fenceposts in probability and thermodynamic threshold,
    it loads the MonteCarlo outputs, it defines nuisance parameters,
    and it defines the data to be fit.
'''
warnings.simplefilter("ignore", RuntimeWarning, 459, True)

#Choose single or multiple threshold fenceposts
single_ET = False

# First define fenceposts for Epts
p_fenceposts = np.array([0, .2, .5, .8, 1], dtype=np.float64)

if not single_ET:
    threshold_fenceposts = np.array([ 2.45 , 3.2 ],dtype=np.float64)
    reparam_fenceposts = np.array([ 0.0 , 0.0 ],dtype=np.float64)
else:
    threshold_fenceposts = np.array([2.45], dtype=np.float64)
    reparam_fenceposts = np.array([0.0], dtype=np.float64)

#A of atoms
species_list = np.array([12, 19],dtype=np.int32)

#Number of parameters
n_Epts = p_fenceposts.size * threshold_fenceposts.size * species_list.size

# now define our experiment list
# for PICO NR study, it was 2013_97, 2013_61, 2014_97, 2014_61, 2014_50, 
#   pico2l_2013_lt, pico2l_2013_ht, and SbBe4_2inPb
experiment_list = ['2013_97',
                   '2013_61',
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
                   
#default # of nuisance parameters
n_nuisance = 14

# Load and prepare data and simulation
def prep(topdir_searchlocations):
    
    #declare global variables
    global n_nuisance
    global n_nuisance_all
    global neutron_data
    global neutron_sims
    global fidfun_list
    global vetofun_list
    global chisq_hard_cap
    global eb_bestfit
    global eb_1sig

    for topdir in topdir_searchlocations:
        if os.path.isdir(topdir):
            break
    # now load the simulation outputs
    simfile_list = [os.path.join(topdir, exp, 'simout.bin')
                    for exp in experiment_list]
    neutron_sims = [sbc.read_bin(simfile) for simfile in simfile_list]
    # neutron_sims is now a list of dictionaries, each dictionary
    # with fields id(n), pos(n,3), Er(n), and species(n)
    
    # now roll our hidden variable dice
    np.random.seed(1234)
    n_hiddenvars = 21  # we can try this many times (-1) to make each bubble
    for nsi in range(len(neutron_sims)):
        nsim = neutron_sims[nsi]
        if nsi == 1:
            nsim['hiddenvars'] = np.random.rand(nsim['Er'].size, 7)
        else:
            nsim['hiddenvars'] = np.random.rand(nsim['Er'].size, n_hiddenvars)
    # hiddenvars is shape (n, n_hiddenvars)
    # Note: only hidden_vars = 7 for 2013_61 due to large SIM file
    
    # checking if there are 256 bubbles in Monte_Carlo, as it will create problems in SimulatedCounts with cumsum(dtype=uint8)
        unique, counts = np.unique(nsim['id'], return_counts=True)
        max_recoils = np.max(counts)
        if max_recoils > 255:
            warnings.warn("max recoils exceeds uint8 limit,check SimulatedCounts")
    
    # now load the data tables (thresholds, counts, exposures, livetimes)
    datafile_list = [os.path.join(topdir, exp, 'data.bin')
                     for exp in experiment_list]
    neutron_data = [sbc.read_bin(datafile) for datafile in datafile_list]
    # neutron_data is now a list of dictionaries, each dictionary
    # with fields E_T(m), max_mult(m), exp(m), lt(m),
    #             bkg_rate(m,k), counts(m,k), nuisance(m,3,b,k)

    ## creating masks to deal with problematic datasets
    for i in range(len(neutron_data)):
        neutron_data[i]['mask'] = np.ones(neutron_data[i]['counts'].shape) 
        
    ## adding individual mask
    ## SbBe gamma dominate singles (experiment SbBe4_2inPb)
    neutron_data[7]['mask'][:,0] = 0
        
    
    # Threshold cut to constrain interpolation range  (1.9-3.9)  
    # threshold_cut_high = 3.9 for final PICO NR study, changed by DD
    threshold_cut_low= 1.9
    threshold_cut_high= 3.9
    
    # apply threshold cut
    if single_ET:
        for nd in neutron_data:
            ETcut = (nd['E_T'] > threshold_fenceposts[0] - .7) * (nd['E_T'] < threshold_fenceposts[0] + .7)
            for k in nd.keys():
                nd[k] = nd[k][ETcut]
    else:
        for nd in neutron_data:
            ETcut = (nd['E_T'] < threshold_cut_high)*(nd['E_T'] > threshold_cut_low)
            for k in nd.keys():
                nd[k] = nd[k][ETcut]
                
    #Number of nuisance parameters
    n_nuisance_all = [nd['nuisance'].shape[2] for nd in neutron_data]
    if not np.all(np.diff(np.array(n_nuisance_all)) == 0):
        print('Inconsistent numbers of nusicance parameters between experiments')

    n_nuisance = n_nuisance_all[0]
    
    # also need to set cuts -- start out with true fiducial cuts and
    # false veto cuts, can overwrite cuts for individual experiments
    # later as needed.  Cuts can be made on both pos and hidden variables
    fidfun_list = [lambda pos, hv: np.ones(hv.shape, dtype=np.bool)
                   for exp in experiment_list]
    
    vetofun_list = [lambda pos, hv: np.zeros(hv.shape, dtype=np.bool)
                    for exp in experiment_list]
    
    # now overwrite specific fidfun's and vetofun's, using the ordering
    # in experiment_list above
    # PICO0.1 2013 having a foam between C3F8 & buffer,
    # vetofun to kill any event with a bubble in foam.
    #vetofun_list[0] = lambda pos, hv: pos[:, 2] > 0.995  # 2013_97
    vetofun_list[1] = lambda pos, hv: pos[:, 2] > 1.1438  # 2013_61
    
    fidfun_list[5] = lambda pos, hv: hv < .7757  # pico2l_2013_lt
    fidfun_list[6] = lambda pos, hv: hv < .6241  # pico2l_2013_ht

    #vectors for nuisance mean and # sigma
    eb_bestfit = np.zeros(n_nuisance)
    eb_1sig = np.ones(n_nuisance)
    
    #chi2 cap, set to a better value with MCMC code
    chisq_hard_cap = 1e100

def PICOcalLL(theta, whichnuisance=np.ones(n_nuisance, dtype=np.bool)):
    ''' Top level log-likelihod function in this module

        This function inputs parameters and outputs a total
        log-likelihood.  Optional argument whichnuisance is a
        boolean that identifies which nuisance parameters are
        included in the input parameter list theta (the rest
        are held at zero)
        '''
    dEpts_reparam = np.reshape(theta[:n_Epts], (p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size))
    
    dEpts = np.zeros([p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size])
    
    for i_th in range(threshold_fenceposts.shape[0]):
        dEpts[0,i_th,:] = reparam_fenceposts[i_th] + np.exp( dEpts_reparam[0,i_th,:])
        dEpts[1:,i_th,:] = np.exp(dEpts_reparam[1:,i_th,:])
        
    Epts = np.cumsum(dEpts, axis=0)

    #if not CheckEpts(Epts):
    #    return -np.inf

    xpts = Epts / threshold_fenceposts[:, np.newaxis]

    eb = np.zeros(n_nuisance, dtype=np.float64)
    eb[whichnuisance] = theta[n_Epts:]

    total_chisq = np.sum( (eb - eb_bestfit) * (eb - eb_bestfit))
    
    #from IPython.core.debugger import Tracer; Tracer()() 
    
    if not CheckParametrization(Epts):
        #temp = np.square(Epts[:,0,0]-Epts[:,0,1])
        #temp[temp>0]=0
        #total_chisq= total_chisq + np.sum(np.square(temp))*10
        return -np.inf

    for i_exp in range(len(experiment_list)):
        
        if neutron_data[i_exp]['E_T'].size == 0:
            continue
        
        nu = SimulatedCounts(xpts, eb, i_exp,eb_1sig)
        # nu.shape = neutron_Data[i_exp]['counts'].shape = (m, k)
        nu[nu < .1] = .1
        # that's to prevent bad -inf's from showing up...  really ought
        # to be handled in bkg_rate though.
        #this_chisq = PoissonChisq(neutron_data[i_exp]['counts'], nu)
        this_chisq = PoissonChisq(neutron_data[i_exp]['counts'], nu) * neutron_data[i_exp]['mask']
        bub_mult = np.cumsum(np.ones(nu.shape, dtype=np.int32), axis=1)
        bub_keep = bub_mult <= neutron_data[i_exp]['max_mult'][:, np.newaxis]
        total_chisq = total_chisq + np.sum(this_chisq[bub_keep].ravel())

        #from IPython.core.debugger import Tracer; Tracer()() 
        
        print(this_chisq)
        
        
    return -0.5 * total_chisq

def PICOcalLL_prior(theta, whichnuisance=np.ones(n_nuisance, dtype=np.bool)):
    ''' Prior log-likelihod function in this module

        This function inputs parameters and outputs a total
        log-likelihood.  Optional argument whichnuisance is a
        boolean that identifies which nuisance parameters are
        included in the input parameter list theta (the rest
        are held at zero)
        '''
    
    dEpts_reparam = np.reshape(theta[:n_Epts], (p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size))
    
    dEpts = np.zeros([p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size])
    
    for i_th in range(threshold_fenceposts.shape[0]):
        dEpts[0,i_th,:] = reparam_fenceposts[i_th] + np.exp( dEpts_reparam[0,i_th,:])
        dEpts[1:,i_th,:] = np.exp(dEpts_reparam[1:,i_th,:])
    
    Epts = np.cumsum(dEpts, axis=0)

    eb = np.zeros(n_nuisance, dtype=np.float64)
    eb[whichnuisance] = theta[n_Epts:]
    #eb = eb_bestfit
    
    total_chisq = np.sum( (eb - eb_bestfit) * (eb - eb_bestfit))
    
    if not CheckParametrization(Epts):
        #temp = np.square(Epts[:,0,0]-Epts[:,0,1])
        #temp[temp>0]=0
        #total_chisq= total_chisq + np.sum(np.square(temp))*10
        return -np.inf

    return -0.5 * total_chisq

def PICOcalLL_post(theta, whichnuisance=np.ones(n_nuisance, dtype=np.bool)):
    ''' Posterior log-likelihod function in this module

        This function inputs parameters and outputs a total
        log-likelihood.  Optional argument whichnuisance is a
        boolean that identifies which nuisance parameters are
        included in the input parameter list theta (the rest
        are held at zero)
        '''

    # Treat input
    dEpts_reparam = np.reshape(theta[:n_Epts], (p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size))
    
    dEpts = np.zeros([p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size])
    
    for i_th in range(threshold_fenceposts.shape[0]):
        dEpts[0,i_th,:] = reparam_fenceposts[i_th] + np.exp( dEpts_reparam[0,i_th,:])
        dEpts[1:,i_th,:] = np.exp(dEpts_reparam[1:,i_th,:])
    
    Epts = np.cumsum(dEpts, axis=0)
    
    # Note: new CheckParametrization, not old CheckEpts
    if not CheckParametrization(Epts):
        return -np.inf

    xpts = Epts / threshold_fenceposts[:, np.newaxis]

    eb = np.zeros(n_nuisance, dtype=np.float64)
    eb[whichnuisance] = theta[n_Epts:]

    # initialize    
    total_chisq = 0

    # loop over experiments
    for i_exp in range(len(experiment_list)):

        # In this case one experiment (pico2l_2013_ht) has no data that
        # contributes, so skip
        if neutron_data[i_exp]['E_T'].size == 0:
            continue
        
        # expected number of counts 
        nu = SimulatedCounts(xpts, eb, i_exp,eb_1sig)
        # nu.shape = neutron_Data[i_exp]['counts'].shape = (m, k)

        #calculate chi2
        this_chisq = PoissonChisq(neutron_data[i_exp]['counts'], nu) * neutron_data[i_exp]['mask']

        # in case nu has bad values        
        this_chisq[np.isinf(this_chisq) == True] = 0.   

        # enforce max_mult of bubbles, sum up chi2          
        bub_mult = np.cumsum(np.ones(nu.shape, dtype=np.int32), axis=1)
        bub_keep = bub_mult <= neutron_data[i_exp]['max_mult'][:, np.newaxis]
        total_chisq = total_chisq + np.sum(this_chisq[bub_keep].ravel())
        
    #Control MCMC to be above chisq_hard_cap set manually
    if (total_chisq +  np.sum( (eb - eb_bestfit) * (eb - eb_bestfit))) > chisq_hard_cap:
        return -np.inf
        
    # Note: apply prior here as well because emcee.EnsembleSampler no longer includes prior
    return -(total_chisq +  np.sum( (eb - eb_bestfit) * (eb - eb_bestfit)))

def PICOcalResiduals(theta, whichnuisance=np.ones(n_nuisance, dtype=np.bool)):
    ''' Residuals (signed log-likelihood) function in this module

        This function inputs parameters and outputs a total
        log-likelihood.  Optional argument whichnuisance is a
        boolean that identifies which nuisance parameters are
        included in the input parameter list theta (the rest
        are held at zero)
        '''

    dEpts = np.reshape(theta[:n_Epts], (p_fenceposts.size,
                                        threshold_fenceposts.size,
                                        species_list.size))
    Epts = np.cumsum(dEpts, axis=0)

    if not CheckEpts(Epts):
        return -np.inf

    xpts = Epts / threshold_fenceposts[:, np.newaxis]

    eb = np.zeros(n_nuisance, dtype=np.float64)
    eb[whichnuisance] = theta[n_Epts:]

    chilist = eb[whichnuisance]

    for i_exp in range(len(experiment_list)):
        nu = SimulatedCounts(xpts, eb, i_exp)
        # nu.shape = neutron_Data[i_exp]['counts'].shape = (m, k)
        bub_mult = np.cumsum(np.ones(nu.shape, dtype=np.int32), axis=1)
        bub_keep = bub_mult <= neutron_data[i_exp]['max_mult'][:, np.newaxis]
        new_chilist = PoissonChi(neutron_data[i_exp]['counts'], nu)
        chilist = np.append(chilist,
                            new_chilist[bub_keep])
    return chilist

def CheckEpts(Epts):
    ''' Checks if this set of Epts is allowed or not

        Checks that Epts is everywhere above thermodynaimc
        threshold, monotoninically increasing in p and t axes,
        and monotonically decreasing in s axis
        
        also adding a 3MeV hard cap (maximum carbon recoil for AmBe) 
        to prevent MCMC walker from getting lost
        '''

    if np.any((Epts < threshold_fenceposts[:, np.newaxis]).ravel()):
        return False

    if np.any(np.diff(Epts, axis=0).ravel() < 0):
        return False

    if np.any(np.diff(Epts, axis=1).ravel() < 0):
        return False

    if np.any(np.diff(Epts, axis=2).ravel() > 0):
        return False

    if np.any(Epts.ravel()) > 3000:
        return False
    
    warnings.warn("this function should not run, check source code when you see this message.")
    
    return True

#-------- Version copied from Artemis_v4 --------------------------------------

## mimicing CheckEpts for reparametrization version
def CheckParametrization(Epts): 
    
    #enforces seitz threshold
    if np.any((Epts < threshold_fenceposts[:, np.newaxis]).ravel()):
        return False

    #enforces monotonicity in threshold (i.e. that eff for C for 3.2 keV is 
    #always higher than C for 2.45 keV)
    if np.any(np.diff(Epts, axis=1).ravel() < 0):
        return False

    #enforces monotonicity in species (F always lower than C)
    if np.any(np.diff(Epts, axis=2).ravel() > 0):
        return False
    
    #just make sure things don't go terribly wrong, should be visible in posterior plot if that's the case
    if np.any(np.abs(Epts.ravel()) > 100): 
        return False
        
    #adding upper caps on energies as in CheckEpts
    if np.any(Epts.ravel() > 3000):
        return False
        
    #NOTE: 
    #   - monotonicity in recoil energy is naturally enforced by Epts cumcum
    #   - positiveness of recoil energies is enfored by logged parametrization
    
    return True
    
#------------------------------------------------------------------------------
    
def SimulatedCounts(xpts, eb, i_exp, eb_1sig):
    ''' calculate expected counts given simulated data and nuisances

        xpts:  (p,t,s) nd array of recoil energies
        eb:   (b,) nd array of low-level nuisance parameters
        i_exp:  scalar identifying which experiment to lookup
        eb_1sig: scalar corresponding 3.2keV results

        output:  nu, size given by neutron_data[i_exp]['counts']
        '''

    #from IPython.core.debugger import Tracer; Tracer()() 
    
    exp_rescale = np.sum(neutron_data[i_exp]['nuisance'][:, 0, :, :] *
                         (eb[:, np.newaxis] * eb_1sig[:, np.newaxis]), axis=1)
    lt_rescale = np.sum(neutron_data[i_exp]['nuisance'][:, 1, :, :] *
                        (eb[:, np.newaxis] * eb_1sig[:, np.newaxis]), axis=1)
    thr_rescale = np.sum(neutron_data[i_exp]['nuisance'][:, 2, :, :] *
                         (eb[:, np.newaxis] * eb_1sig[:, np.newaxis]), axis=1)
    # all the rescales have shape (m, k) where m is # of thresholds

    bkg_counts = neutron_data[i_exp]['bkg_rate'] *\
        neutron_data[i_exp]['lt'][:, np.newaxis] *\
        np.exp(lt_rescale)
    # bkg_counts.shape = (m, k)

    ET_eff = neutron_data[i_exp]['E_T'] * np.exp(thr_rescale[:, 0])
    # ET_eff.shape = (m,)

    p = EfficiencyInterpolation_nearest_neighbor(neutron_sims[i_exp]['Er'],
                                neutron_sims[i_exp]['species'],
                                ET_eff, xpts)
    # p.shape = (m, n), where n is number of recoils in sim

    bubbles = neutron_sims[i_exp]['hiddenvars'][:, 1:] < p[:, :, np.newaxis]
    # bubbles.shape = (m, n, t), were t is number of trials per bubble

    fidcut = fidfun_list[i_exp](neutron_sims[i_exp]['pos'],
                                neutron_sims[i_exp]['hiddenvars'][:, 0])
    vetocut = vetofun_list[i_exp](neutron_sims[i_exp]['pos'],
                                  neutron_sims[i_exp]['hiddenvars'][:, 0])
    # cuts have shape (n,)

    # now, for each trial and threshold, we need to figure out
    # how many bubbles each simulated neutron creates, and
    # adjust for fiducial and veto cuts
    ev_midposts = np.nonzero(np.diff(neutron_sims[i_exp]['id']))[0]
    ev_posts = np.zeros(ev_midposts.size + 2, dtype=np.intp)
    ev_posts[1:-1] = ev_midposts + 1
    ev_posts[-1] = neutron_sims[i_exp]['id'].size

    bubcount = np.zeros((bubbles.shape[0],
                         bubbles.shape[1] + 1,
                         bubbles.shape[2]),
                        dtype=np.uint8)
    np.cumsum(bubbles, axis=1, out=bubcount[:, 1:, :], dtype=np.uint8)
    nbub = np.diff(bubcount[:, ev_posts, :], axis=1)

    vetoed_bubbles = vetocut[:, np.newaxis] * bubbles
    vetocount = np.zeros((bubbles.shape[0],
                          bubbles.shape[1] + 1,
                          bubbles.shape[2]),
                         dtype=np.uint8)
    np.cumsum(vetoed_bubbles, axis=1, out=vetocount[:, 1:, :], dtype=np.uint8)
    vetoed = np.diff(vetocount[:, ev_posts, :], axis=1) > 0
    nbub[vetoed] = 0

    fid_bubbles = fidcut[:, np.newaxis] * bubbles
    fidcount = np.zeros((bubbles.shape[0],
                         bubbles.shape[1] + 1,
                         bubbles.shape[2]),
                        dtype=np.uint8)
    np.cumsum(fid_bubbles, axis=1, out=fidcount[:, 1:, :], dtype=np.uint8)
    infid = np.diff(fidcount[:, ev_posts, :], axis=1) > 0
    singles_out_of_fid = (nbub == 1) * (~infid)
    nbub[singles_out_of_fid] = 0
    # now we have nbub all set
    # with nbub.shape = (m, i, t) where i is number of neutrons

    #debugging when i_exp == 7, code fails as ET_eff = []
    
    #from IPython.core.debugger import Tracer; Tracer()() 
    
    # now we'll count events by multiplicity!
    nbub_all = nbub.reshape(ET_eff.size, -1)
    # nbub_all.shape = (m, t*i)

    nbub_test = np.array(range(bkg_counts.shape[1])) + 1
    # nbub_test.shape = (k,)

    sim_counts = np.sum(nbub_all[:, np.newaxis, :] ==
                        nbub_test[np.newaxis, :, np.newaxis],
                        axis=2)
    # nbub_hist.shape = (m, k)

    sim_counts[:, -1] += np.sum(nbub_all > bkg_counts.shape[1],
                                axis=1)

    trials_rescale = 1.0 / bubbles.shape[2]

    nu = bkg_counts + (sim_counts *
                       np.exp(exp_rescale) *
                       neutron_data[i_exp]['exp'][:, np.newaxis] *
                       trials_rescale)
    return nu

def EfficiencyInterpolation(E_r, s, E_T, xpts):
    ''' This calculates bubble nucleation efficiencies from xpts

        Inputs: E_r, 1-D ndarray of recoil energies, length n
                s,   1-D ndarray of recoil species, length n
                E_T, 1-D ndarray of detector thresholds, length m
                xpts, 3-D ndarray of E_r/E_T, shape is
                len(p_fenceposts), len(threshold_fenceposts), len(species_list)

        Outputs: 2-D ndarray of nucleation probabilities, shape m, n
        '''
    
    if threshold_fenceposts.size == 1:
        xps = xpts
        
    else:
        ET_ix = np.searchsorted(threshold_fenceposts, E_T)
        # ET_ix.shape = (m,)

        # Set extrapolation mode for E_T
        ET_ix[ET_ix < 1] = 1
        ET_ix[ET_ix >= threshold_fenceposts.size] =\
            threshold_fenceposts.size - 1

        # do this interp by hand so we get the index broadcasting right
        xps = xpts[:, ET_ix - 1, :] +\
            (xpts[:, ET_ix, :] - xpts[:, ET_ix - 1, :]) *\
            ((E_T[:, np.newaxis] -
              threshold_fenceposts[ET_ix - 1, np.newaxis]) /
             (threshold_fenceposts[ET_ix, np.newaxis] -
              threshold_fenceposts[ET_ix - 1, np.newaxis]))

    E_ps = xps * E_T[:, np.newaxis]
    # E_ps.shape = (len(p_fenceposts), m, len(species_list))

    # Now calculate probabilities for each species, threshold pair
    p = np.zeros((E_T.size, E_r.size))
    for i_ET in range(p.shape[0]):
        for i_s, this_s in enumerate(species_list):
            s_cut = (s == this_s)
            p[i_ET, s_cut] = np.interp(E_r[s_cut],
                                       E_ps[:, i_ET, i_s],
                                       p_fenceposts,
                                       left=0, right=1)
    return p

def EfficiencyInterpolation_nearest_neighbor(E_r, s, E_T, xpts):
    ''' This calculates bubble nucleation efficiencies from xpts

        Inputs: E_r, 1-D ndarray of recoil energies, length n
                s,   1-D ndarray of recoil species, length n
                E_T, 1-D ndarray of detector thresholds, length m
                xpts, 3-D ndarray of E_r/E_T, shape is
                len(p_fenceposts), len(threshold_fenceposts), len(species_list)

        Outputs: 2-D ndarray of nucleation probabilities, shape m, n
        '''
 
    if threshold_fenceposts.size == 1:
        xps = xpts
        E_ps = xps * E_T[:, np.newaxis]
    
    ## adding a special case for 2 threshold interpolation
    ## hardcode 2.8keV to separate two thresholds
    else:
        E_ps = np.zeros((len(p_fenceposts), len(E_T), len(species_list)))
        
        for i_Eth in range(len(E_T)):
            i_neighbor = np.argmin((abs(E_T[i_Eth]-threshold_fenceposts)))            
            xps = xpts[:,i_neighbor,:]
            E_ps[:,i_Eth,:] = xps * E_T[i_Eth]
                                   
    # E_ps.shape = (len(p_fenceposts), m, len(species_list))           
    # Now calculate probabilities for each species, threshold pair
    p = np.zeros((E_T.size, E_r.size))
    for i_ET in range(p.shape[0]):
        for i_s, this_s in enumerate(species_list):
            s_cut = (s == this_s)
            p[i_ET, s_cut] = np.interp(E_r[s_cut],
                                       E_ps[:, i_ET, i_s],
                                       p_fenceposts,
                                       left=0, right=1)
    return p

def PoissonChisq(n, nu):
    ''' Calculates Poisson chi-square '''
    # see Baker and Cousins for reference
    chisq_0 = 2 * (nu - n)
    chisq_1 = -2 * n * np.log(nu / n)
    chisq_1[np.isnan(chisq_1)] = 0
    return chisq_0 + chisq_1


def PoissonChi(n, nu):
    ''' Calculates signed sqrt of Poisson chi-square '''
    return np.sqrt(PoissonChisq(n, nu)) * np.sign(nu - n)


def PoissonChisqWBkg(n, nu, k, r):
    ''' Calculates Poisson chi-square with bkg dataset
        n = # of events observed in "signal" dataset
        nu = # of signal events expected
        k = # of events observed in "background" dataset
        r = exposure of "signal" dataset relative to "background" '''
    b = n + k - nu * ((1 + r) / r)
    a = 1 + r
    mu = (b + np.sqrt(b * b + 4 * a * k * (nu / r))) / (2 * a)
    mu[r == 0] = k[r == 0]  # special case -- known background
    chisq_0 = 2 * (nu + mu + mu * r - n - k)
    chisq_1 = -2 * n * np.log((nu + mu * r) / n)
    chisq_1[np.isnan(chisq_1)] = 0
    chisq_2 = -2 * k * np.log(mu / k)
    chisq_2[np.isnan(chisq_2)] = 0
    return chisq_0 + chisq_1 + chisq_2


def PoissonChiWBkg(n, nu, k, r):
    ''' Calculates signed sqrt of Poisson chi-square w bkg '''
    return np.sqrt(PoissonChisqWBkg(n, nu, k, r)) * np.sign(nu - n + k * r)
