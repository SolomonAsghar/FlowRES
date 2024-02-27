'''
Functions and classes that use a FlowNet to conduct enhanced MCMC of active systems.
'''
import math
import numpy as np
import tensorflow as tf
import sys
sys.path.append(r'/home/solomon.asghar/NF_TPS/software/')
from util import *
from time import time
from ABW_ProbEqs_Active import *
from Plotting import *

def Chain_Initialiser_Active(Net, tran_coeff, buffer_size=10000):
    Pos_Noise = np.random.normal(0, tran_coeff, size=(buffer_size, Net.max_len, 2))
    Ang_Noise = np.random.vonmises(0, 1/(Net.rot_coeff**2), size=(buffer_size, Net.max_len, 1))
    Noise = np.concatenate([Pos_Noise, Ang_Noise], axis=2)
    Starts_pos = np.tile(Net.start, (buffer_size,1,1))
    Starts_ang = np.random.uniform(-math.pi, math.pi, size=(buffer_size,1,1))
    Starts = np.concatenate([Starts_pos, Starts_ang], axis=2)
    Trajs_diff = np.concatenate([Starts,Noise], axis=1)
    Trajs = np.cumsum(Trajs_diff, axis=1)
    Trajs[:,:,2] = PM_pi(Trajs[:,:,2])    
    return Trajs[:,1:]

def Gaussian_Generator(size):
    '''
    Generates a Gaussian matrix to be used as a prior.
        
    size [int,int]: the size of the desired prior = size of the desired paths
    '''
    while True: # or we only generate once
        Gaussian = np.random.normal(size=size)
        yield Gaussian
        
def Angle_Walk_Generator(Net, size):
    '''
    Generates orientation walks.

    Net [FlowNet]: A network, created via CreateFlowNet from Network_Passive.py 
    size [int,int]: the size of the desired paths
    '''
    while True: # or we only generate once
        # Create rotational walk
        starts = np.random.uniform(-math.pi, math.pi, size=(size[0],1,1))    
        
        Ang_Increments = np.random.vonmises(0, 1/(Net.rot_coeff**2), size=(size[0], size[1], 1))
        Ang_Increments = np.concatenate((starts, Ang_Increments), axis=1)
        Ang_Walk = np.cumsum(Ang_Increments, axis=1)
        Ang_Walk = PM_pi(Ang_Walk)

        # Combine to yeild walk
        yield Ang_Walk[:,1:]

def loss_ML_true(Net, starts, dX=0.001):
    '''
    Maximum-Likelihood loss (i.e. Kullback-Leilbler divergence between prior distribution and distribution of samples mapped onto the prior).

    starts [float,float,float]: Starting x,y,theta of all paths
    dX [float]: Delta used for numerical differentiation
    '''
    ####
    Angles = Net.output_z_angles
    Ang_Increments = Angles[:,1:] - Angles[:,:-1] # First has uniform distribution, therefore constant probability so can be ignored
    Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    ang_latent_log_prob = - Net.rot_coeff * tf.reduce_sum(tf.cos(Ang_Increments))
    ####
    pos_latent_log_prob = 0.5 * tf.reduce_sum(Net.output_z_positions**2)

    return pos_latent_log_prob + ang_latent_log_prob - tf.reduce_sum(Net.log_Rxz)


class FlowRes_MCMC():
    '''
    Does MCMC using FlowRES proposals, generating an asymptotically accurate ensemble and training a network.
    
    Net [FlowNet]: A network, created via CreateFlowNet from Network_Passive.py
    num_chains [int]: The number of markov chains used when sampling, c_total in the paper.
    Sample_Buffer_Generator [func]: A function that generates an ensemble composed of each chains initial path, {w_c(0} in the paper. 
    potential [func]: A function defining the potential used by the system we want to simulate.
    '''
    def __init__(self, Net, num_chains, Sample_Buffer_Generator, potential):
        self.Net = Net
        self.num_chains = num_chains
        self.initial_chains = Sample_Buffer_Generator(Net, Net.tran_coeff, num_chains)
        self.potential = potential
        self.Build_Mixed_Model()
        self.Build_XtoZ_log_Rxz_Model()
        self.Build_ZtoX_log_Rzx_Model()

    def Build_Mixed_Model(self):
        '''
        Build a model that can do both XtoZ and ZtoX
        '''
        inputs = [[self.Net.input_x_positions, self.Net.input_x_angles], [self.Net.input_z_positions, self.Net.input_z_angles]]
        outputs = [[self.Net.output_z_positions, self.Net.output_z_angles], [self.Net.output_x_positions, self.Net.output_x_angles]]
        self.Net.Mixed_Model = tf.keras.Model(inputs, outputs)


    def Build_XtoZ_log_Rxz_Model(self):
        '''
        Build a model that transforms X (path distribution) to Z (base distribution) and also returns log(Rxz)
        '''
        inputs_x = [self.Net.input_x_positions, self.Net.input_x_angles]
        outputs_z = [self.Net.output_z_positions, self.Net.output_z_angles, self.Net.log_Rxz]
        self.Net.transform_XtoZ_log_Rxz = tf.keras.Model(inputs_x, outputs_z)

    def Build_ZtoX_log_Rzx_Model(self):
        '''
        Build a model that transforms Z (base distribution) to X (path distribution) and also returns log(Rzx)
        '''
        inputs_z = [self.Net.input_z_positions, self.Net.input_z_angles]
        outputs_x = [self.Net.output_x_positions, self.Net.output_x_angles, self.Net.log_Rzx]
        self.Net.transform_ZtoX_log_Rzx = tf.keras.Model(inputs_z, outputs_x)
    
    def Transform_ZtoX(self, z_pos, z_ang):
        '''
        A function that takes Z samples and transforms them into X
        '''
        return self.Net.net_ZtoX.predict_on_batch([z_pos, z_ang])
    
    def Compile_ML_Model(self, batch_size, lr):
        '''
        Compile the latent MCMC model so that it may be used for training.

        batch_size [int]: Size of each batch used in training
        lr [float]: Learning rate used while training.
        '''
        self.batch_size = batch_size
        starts = np.tile([self.Net.start], (batch_size,1,1)).astype('float32')
        
        ### Set up ML stuff ###
        def loss_ML(y_true, y_pred):
            return loss_ML_true(self.Net, starts)
        def Fake_Loss(y_true, y_pred):
            return 0.0

        losses = [loss_ML, Fake_Loss]
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        self.Net.net_XtoZ.compile(optimizer=opt, loss=losses)
        
        size = (batch_size, self.Net.max_len, 2)
        self.Pos_Base = Gaussian_Generator(size)
        ang_starts = np.tile(self.Net.start[2:], (batch_size,1,1))
        self.Ang_Base = Angle_Walk_Generator(self.Net, size)

    def Train(self, x_pos, x_ang, epochs=1, verbose=1):
        '''
        Train the network via maximum liklihood

        x_pos [numpy array]: The paths we train on
        epochs [int]: How many epochs to train for
        verbose [0, 2 or 1]: How much progress information we want to see while training. 0 = silent, 1 = epoch count and progress per epoch, 2 = only epoch count.
        '''
        history = self.Net.net_XtoZ.fit(x=[x_pos, x_ang],
                                        y=[x_pos, x_ang],
                                        batch_size=self.batch_size,
                                        epochs=epochs,
                                        verbose=verbose)
        return history


    def Acc_or_Rej(self, acc_prob, x_pos, x_ang, new_x_pos, new_x_ang):
        '''
        Accept or reject the new trajs based on the acc_prob

        acc_prob [numpy array]: Contains logged acceptance probabilities for each path x_pos_new given the corresponding x_pos,  Pos_acc(x_pos, x_pos_new) in the paper.
        x_pos [numpy array]: Paths for all chains at the current iteration, {w_c(m)}
        new_x_pos [numpy array]: Proposal paths for each chains next iteration, {w_c'}
        '''
        # Accept or reject the new trajectories
        random_prob = np.log(np.random.uniform(size=np.shape(acc_prob)))
        acceptance_mask = acc_prob > random_prob

        acceptance_mask_x_pos = np.tile(acceptance_mask[:,None,None],([1,self.Net.max_len,2]))
        acceptance_mask_x_ang = np.tile(acceptance_mask[:,None,None],([1,self.Net.max_len,1]))
        x_pos = np.where(acceptance_mask_x_pos, new_x_pos, x_pos)
        x_ang = np.where(acceptance_mask_x_ang, new_x_ang, x_ang)
        num_accepted = np.count_nonzero(acceptance_mask)
        percent_accepted = 100*(num_accepted/self.num_chains)

        return x_pos, x_ang, percent_accepted
    
    def non_local_acc_prob_calculator(self, x_pos, x_ang, new_x_pos, new_x_ang, reactivity):
        '''
        Calculate log of acceptance probability of new_x_pos given x_pos, Pos_acc(x_pos, x_pos_new) in the paper.

        x_pos [numpy array]: Paths for all chains at the current iteration, {w_c(m)}
        new_x_pos [numpy array]: Proposal paths for each chains next iteration, {w_c'}
        reactivity [float]: What percentage of x_pos paths are reactive (i.e. target reaching)
        '''
        # Create array of zeros for all probabilities
        final_log_acc_prob = np.zeros(self.num_chains)
        
        # See what's reactive 
        new_reactivity = self.get_reac_mask(new_x_pos)
    
        # If new reactive and old unreactive, accept instantly - i.e. log prob acceptance stays 0
        instant_accept_mask = np.logical_and(new_reactivity, np.logical_not(reactivity))
        
        # If new reactive and old unreactive, reject instantly - i.e. prob becomes -inf
        instant_reject_mask = np.logical_and(np.logical_not(new_reactivity), reactivity)
        final_log_acc_prob[instant_reject_mask] = float('-inf')
        
        # For all other cases, calculate acceptance prob and save them in our array of log probs
        need_to_calculate_probs = np.logical_and(np.logical_not(instant_accept_mask), np.logical_not(instant_reject_mask))
        x_pos = x_pos[need_to_calculate_probs]
        x_ang = x_ang[need_to_calculate_probs]
        new_x_pos = new_x_pos[need_to_calculate_probs]
        new_x_ang = new_x_ang[need_to_calculate_probs]
                
        latent_old_pos, latent_old_ang, old_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict_on_batch([x_pos, x_ang])
        latent_new_pos, latent_new_ang, new_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict_on_batch([new_x_pos, new_x_ang])
        
        log_prob_old = Log_Target_Prob(self.Net, x_pos, x_ang)
        log_prob_latent_old = Log_Latent_Prob(self.Net, latent_old_pos, latent_old_ang)
        log_prob_new = Log_Target_Prob(self.Net, new_x_pos, new_x_ang)
        log_prob_latent_new = Log_Latent_Prob(self.Net, latent_new_pos, latent_new_ang)

        log_acc_prob_numerator = log_prob_latent_old + old_log_R_xz + log_prob_new
        log_acc_prob_denominator = log_prob_old + log_prob_latent_new + new_log_R_xz
        log_acc_prob = log_acc_prob_numerator - log_acc_prob_denominator
        log_acc_prob = np.where(log_acc_prob>0, 0, log_acc_prob)
        
        # Return overall acc prob and reactivity 
        final_log_acc_prob[need_to_calculate_probs] = log_acc_prob
        reactivity = np.logical_or(reactivity, new_reactivity)
        
        return final_log_acc_prob, reactivity 


    def get_reac_mask(self, Positions):
        '''
        Return a mask where only valid (reactive aka target reaching) trajs are True

        Positions [numpy array]: Array containing positions of paths. 
        '''
        energies = self.potential.energy(Positions, input_mode='Batch')
        energy_mask = energies < self.potential.params['target']
        right_mask = Positions[:,:,0] > 0
        r_e_mask = np.logical_and(energy_mask,right_mask)
        not_r_e_mask = np.logical_not(r_e_mask)
        unreactive_mask = np.all(not_r_e_mask, axis=1)
        reactive_mask = np.logical_not(unreactive_mask)
        return reactive_mask
    

    def Explore(self, iterations, epochs=1, return_chains=False, return_hists=False, hist_bins=None, save_to_file=False, save_net=False, file_loc=None, save_each=1):
        '''
        Combine local MCMC (Metropolis-Hastings) with training. This is the function that actually does the sampling.

        iterations [int]: Max number of iterations desired, m_max in the paper.  
        epochs [int]: Epochs of training per training step, just set to 1
        return_chains [bool]: Do you want the function to return the MCMC chains?
        save_to_file [bool]: Do you want to save the generated paths, network etc.?
        save_net [bool]: Do you want to save the network weights?
        file_loc [string]: Path to where you want to save the generated paths, network etc.
        save_hist_each [int]: How regularly we should save histograms of FlowRes distribution, i.e. save each n iterations. 
        '''
        # Stuff carried through each loop #
        indicies = np.arange(self.num_chains)
        x_pos = self.initial_chains[:,:,:2]
        x_ang = self.initial_chains[:,:,2:]
        if return_chains:
            chains = np.zeros((iterations+1, self.num_chains, self.Net.max_len, 3))
            chains[0] = np.concatenate([x_pos, x_ang], axis=2)
        if return_hists:
            num_bins = len(hist_bins)-1
            all_hists = np.zeros((iterations+1, num_bins, num_bins))
            all_hists[0] = Hist2D(clip_reac(x_pos, self.potential), hist_bins)
        reactivity = self.get_reac_mask(x_pos)    
        
        # Make starts #
        starts = np.tile([self.Net.start], (self.num_chains,1,1))
        # Set Up Generators #
        pos_basis = Gaussian_Generator(size=(self.num_chains, self.Net.max_len, 2))
        angle_basis = Angle_Walk_Generator(self.Net, size=(self.num_chains, self.Net.max_len, 1))
        
        # Data to be monitored #
        times = np.array([])
        ############################################
        
        for itr in range(iterations):
            print("Iteration", itr+1,"/",iterations)
            times = np.append(times, time())
            # Generate Non-Local Proposal
            new_z_pos = next(pos_basis)
            new_z_ang = next(angle_basis)
            new_x_pos, new_x_ang = self.Transform_ZtoX(new_z_pos, new_z_ang)
            # Calculate Acceptance Prob
            acc_prob, reactivity = self.non_local_acc_prob_calculator(x_pos, x_ang, new_x_pos, new_x_ang, reactivity)
            
            # Accept or Reject 
            x_pos, x_ang, percent_accepted = self.Acc_or_Rej(acc_prob, x_pos, x_ang, new_x_pos, new_x_ang)
            if return_chains:
                chains[itr+1] = np.concatenate([x_pos, x_ang], axis=2)
            if return_hists:
                all_hists[itr+1] = Hist2D(clip_reac(x_pos, self.potential), hist_bins)
                
            print("    accepted " + str(percent_accepted) + "% of non-local proposals")
            ######### Train the Network on the new samples ##########
            reactive_indicies = indicies[reactivity]
            np.random.shuffle(reactive_indicies)
            reactive_indicies = reactive_indicies[:int(np.floor(len(reactive_indicies)/self.batch_size)*self.batch_size)]
            self.Train(x_pos[reactive_indicies], x_ang[reactive_indicies], epochs)
            times = np.append(times, time())
            ##########################################################
            print("\n    network trained with non-local proposals")
            ##########################################################
            
            
            ###### Save Info ######
            if save_to_file:
                if ((itr+1) % save_each == 0) or ((itr+1) == iterations): # Save regularly so that even if training is interrupted, we have results stored
                    if return_chains:
                        np.save(file_loc + '_chains', chains[:itr+2].astype('float32'))
                    if return_hists:
                        np.save(file_loc + '_hists', all_hists[:its+2].astype('float32'))
                    # Times and Metrics# 
                    np.save(file_loc + '_times', times)
            #######################
        
        if save_net is True:
            self.Net.net_XtoZ.save_weights(file_loc + '_net.h5')
        
        outputs = []
        if return_chains:
            outputs.append(chains)
        if return_hists:
            outputs.append(all_hists)
        if outputs:
            return tuple(outputs)