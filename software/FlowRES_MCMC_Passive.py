import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
sys.path.append(r'/home/solomon.asghar/NF_TPS/software/')
from util import *
from time import time
from ABW_Passive_ProbEqs import *


def Gaussian_Generator(size):
    '''
    Generates a Gaussian walk to be used as a prior. 
    '''
    while True: # or we only generate once
        Gaussian = np.random.normal(size=size)
        yield Gaussian
        
def loss_ML_true(Net, starts, dX=0.001):
    '''
    Kullback-Leiber loss for target distribution of walks.
    '''
    ####
    # Angles = tf.concat((starts[:,:,2:], self.output_z_angles), axis=1)
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # ang_latent_log_prob = - basis_kappa * tf.reduce_sum(tf.cos(Ang_Increments-basis_mu))
    ####
    pos_latent_log_prob = 0.5 * tf.reduce_sum(Net.output_z_positions**2, axis=(1,2))

    # return pos_latent_log_prob + ang_latent_log_prob - tf.reduce_sum(self.log_Rxz)
    return pos_latent_log_prob - Net.log_Rxz


class Latent_MCMC():
    '''
    Does MCMC in latent space, generating an asymptotically accurate ensemble and training a network.
    '''
    def __init__(self, Net, num_chains, Sample_Buffer_Generator, potential):
        self.Net = Net
        self.num_chains = num_chains
        self.chains = Sample_Buffer_Generator(Net, num_chains)
        self.energy = potential.energy
        self.e_cutoff = potential.params['target']
        self.Build_Mixed_Model()
        self.Build_XtoZ_log_Rxz_Model()
        self.Build_ZtoX_log_Rzx_Model()

    def Build_Mixed_Model(self):
        '''
        Build a model with the right outputs for latent MCMC.
        '''
        # inputs = [[self.Net.input_x_positions, self.Net.input_x_angles], [self.Net.input_z_positions, self.Net.input_z_angles]]
        # outputs = [[self.Net.output_z_positions, self.Net.output_z_angles], [self.Net.output_x_positions, self.Net.output_x_angles]]
        inputs = [[self.Net.input_x_positions], [self.Net.input_z_positions]]
        outputs = [[self.Net.output_z_positions], [self.Net.output_x_positions]]

        Mixed_Model = tf.keras.Model(inputs, outputs)
        self.Net.Mixed_Model = Mixed_Model


    def Build_XtoZ_log_Rxz_Model(self):
        '''
        Build a model with the right outputs for latent MCMC.
        '''
        inputs_x = [self.Net.input_x_positions]
        outputs_z = [self.Net.output_z_positions, self.Net.log_Rxz]
        XtoZ_log_Rxz = tf.keras.Model(inputs_x, outputs_z)

        self.Net.transform_XtoZ_log_Rxz = XtoZ_log_Rxz

    def Build_ZtoX_log_Rzx_Model(self):
        '''
        Build a model with the right outputs for latent MCMC.
        '''
        inputs_z = [self.Net.input_z_positions]
        outputs_x = [self.Net.output_x_positions, self.Net.log_Rzx]
        ZtoX_log_Rzx = tf.keras.Model(inputs_z, outputs_x)

        self.Net.transform_ZtoX_log_Rzx = ZtoX_log_Rzx
    
    def Transform_ZtoX(self, z_pos):
        return self.Net.net_ZtoX.predict_on_batch([z_pos])
    
    def Compile_ML_Model(self, batch_size, lr):
        '''
        Compile the latent MCMC model so that it may be used for training.
        '''
        self.batch_size = batch_size
        starts = np.tile([self.Net.start], (batch_size,1,1)).astype('float32')
        
        ### Set up ML stuff ###
        starts = np.tile([self.Net.start], (batch_size,1,1)).astype('float32')
        def loss_ML(y_true, y_pred):
            return loss_ML_true(self.Net, starts)

        losses = loss_ML
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        self.Net.net_XtoZ.compile(optimizer=opt, loss=losses)

        size=(batch_size, self.Net.max_len, 2)
        self.Pos_Base = Gaussian_Generator(size)


    def Train(self, x_pos, epochs=1, verbose=1):
        '''
        Train the network via maximum liklihood
        '''
        def loss_ML(y_true, y_pred):
            return loss_ML_true(self, starts)
        
        history = self.Net.net_XtoZ.fit(x=x_pos,
                                        y=x_pos, # Not really the target, Keras requires realistic target
                                        batch_size=self.batch_size,
                                        epochs=epochs,
                                        verbose=verbose)
        return history


    def Acc_or_Rej(self, acc_prob, x_pos, new_x_pos):
        '''
        Accept or reject the new trajs based on the acc_prob
        '''
        # Accept or reject the new trajectories
        random_prob = np.log(np.random.uniform(size=np.shape(acc_prob)))
        acceptance_mask = acc_prob > random_prob

        acceptance_mask_x_pos = np.tile(acceptance_mask[:,None,None],([1,self.Net.max_len,2]))
        x_pos = np.where(acceptance_mask_x_pos, new_x_pos, x_pos)
        num_accepted = np.count_nonzero(acceptance_mask)
        percent_accepted = 100*(num_accepted/self.num_chains)

        return x_pos, percent_accepted

    
    def non_local_acc_prob_calculator(self, x_pos, new_x_pos, reactivity):
        '''
        Calculate log of non-local acceptance probability
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
        new_x_pos = new_x_pos[need_to_calculate_probs]       
                
        latent_x_old, old_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict_on_batch(x_pos)
        latent_x_new, new_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict_on_batch(new_x_pos)

        log_prob_old = Log_Target_Prob(self.Net, x_pos)
        log_prob_latent_old = Log_Latent_Prob(latent_x_old)
        log_prob_new = Log_Target_Prob(self.Net, new_x_pos)
        log_prob_latent_new = Log_Latent_Prob(latent_x_new)

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
        Return a mask where only valid trajs are True
        '''
        energies = self.energy(Positions, input_mode='Batch')
        energy_mask = energies < self.e_cutoff
        right_mask = Positions[:,:,0] > 0
        r_e_mask = np.logical_and(energy_mask,right_mask)
        not_r_e_mask = np.logical_not(r_e_mask)
        unreactive_mask = np.all(not_r_e_mask, axis=1)
        reactive_mask = np.logical_not(unreactive_mask)
        return reactive_mask
    

    def Explore(self, iterations, step_size, iter_Train, epochs=1, file_loc=None, save_each=1):
        '''
        Combine local MCMC (Metropolis-Hastings) with training
        '''
        # Stuff carried through each loop #
        x_pos = self.chains
        indicies = np.arange(self.num_chains)
        links = np.expand_dims(x_pos, axis=1)
        chains = links
        reactivity = self.get_reac_mask(x_pos)

        # Data to be monitored #
        times = np.array([])
        ############################################
        
        for iter in range(iterations):
            print("Iteration", iter+1,"/",iterations)
            times = np.append(times, time())
            # Generate Non-Local Proposal
            new_z_pos = np.random.normal(size=(self.num_chains, self.Net.max_len, 2))
            new_x_pos = self.Transform_ZtoX(new_z_pos)

            # Calculate acceptance probabilitiy
            acc_prob, reactivity = self.non_local_acc_prob_calculator(x_pos, new_x_pos, reactivity)

            # Accept or Reject
            x_pos, percent_accepted = self.Acc_or_Rej(acc_prob, x_pos, new_x_pos)
            links = np.expand_dims(x_pos, axis=1)
            chains = np.concatenate([chains, links], axis=1)
            print("    accepted " + str(percent_accepted) + "% of non-local proposals")

            ######### Train the Network on the new samples ##########
            if (iter+1) % (iter_Train) == 0:    # After iter_trian steps, retrain
                reactive_indicies = indicies[reactivity]
                self.Train(x_pos[reactive_indicies], epochs)
            ##########################################################

            times = np.append(times, time())
            print("\n    network trained with [reweighted] non-local samples")
            ####################################
            if ((iter+1) % save_each == 0) or ((iter+1) == iterations):
                ## Save Info ##
                # MCMC Info # 
                np.save(file_loc + '_MCMC_samples', chains.astype('float32'))
                # Times and Metrics# 
                np.save(file_loc + '_times', times)
            ####################################