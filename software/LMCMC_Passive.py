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
    pos_latent_log_prob = 0.5 * tf.reduce_sum(Net.output_z_positions**2)

    # return pos_latent_log_prob + ang_latent_log_prob - tf.reduce_sum(self.log_Rxz)
    return pos_latent_log_prob - tf.reduce_sum(Net.log_Rxz)

def loss_KL_Walks(Net, starts, dX=0.001):
    '''
    Kullback-Leiber loss for target distribution of walks.
    '''
    # Prep by appending start
    Positions = tf.concat((starts[:,:,:2], Net.output_x_positions), axis=1)   
    # Angles = tf.concat((starts[:,:,2:], Net.output_x_angles), axis=1)
    
    # Calculate log prob of the positions
    Pos_Increments = Positions[:,1:] - Positions[:,:-1]

    # Sin_Angles = tf.math.sin(Angles)
    # Cos_Angles = tf.math.cos(Angles)
    
    grad_U = Net.potential_grad(Positions[:,:-1], mode='TF')
    # Activity = tf.concat([Cos_Angles[:,:-1], Sin_Angles[:,:-1]], axis=2)
    
    Translational_Noise = Pos_Increments + Net.pot_coeff*grad_U #- act_coeff*Activity
    Translational_Noise = Translational_Noise/Net.tran_coeff
    pos_log_prob = 0.5 * tf.reduce_sum(Translational_Noise**2)
    ##########

    # Calculate log prob of the angles
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # ang_log_prob = - kappa * tf.reduce_sum(tf.cos(Ang_Increments-mu))

    # return pos_log_prob + ang_log_prob - tf.reduce_sum(Net.log_Rzx)
    return pos_log_prob - tf.reduce_sum(Net.log_Rzx)


class Latent_MCMC():
    '''
    Does MCMC in latent space, generating an asymptotically accurate ensemble and training a network.
    '''
    def __init__(self, Net, num_chains, Sample_Buffer_Generator, potential):
        self.Net = Net
        self.num_chains = num_chains
        self.chains = Sample_Buffer_Generator(Net, num_chains)
        self.potential = potential
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
    
    def Transform_ZtoX_Weights(self, z_pos):
        x_pos, log_Rzx = self.Net.transform_ZtoX_log_Rzx.predict([z_pos])
        starts = np.tile([self.Net.start], (len(z_pos),1,1)).astype('float32')

        # Prepare trajectories       
        Positions = np.concatenate((starts[:,:,:2], x_pos), axis=1)
        #
        Latent_Positions = z_pos

        # Calculate probs of latent vectors in generated latent distribution
        ## Prob of generated samples in target dist
        ### Calculate log prob of the positions
        Pos_Increments = Positions[:,1:] - Positions[:,:-1]
        grad_U = self.Net.potential_grad(Positions[:,:-1], mode='numpy')
        Translational_Noise = Pos_Increments + self.Net.pot_coeff*grad_U 
        Translational_Noise = Translational_Noise/self.Net.tran_coeff
        pos_log_prob = - 0.5 * np.sum(Translational_Noise**2, axis=(1,2))
        ###
        log_prob_gens_in_target = pos_log_prob
        ### Add Rzx to get log probs of latent in generated latent
        log_prob_latent_in_gen_latent = log_prob_gens_in_target + log_Rzx

        # Calculate probs of latent vectors in true latent distribution
        ## log prob of the latent positions
        pos_latent_log_prob = - 0.5 * np.sum(Latent_Positions**2, axis=(1,2))
        ###
        log_probs_latent_in_tru_latent = pos_latent_log_prob 
        ###
        
        # Calculate weights
        W_x = np.exp(log_prob_latent_in_gen_latent - log_probs_latent_in_tru_latent)

        return Positions[:,1:], W_x


    def Compile_Mixed_Model(self, batch_size, lr, w_ML=1, w_KL=1, patience=5, epochs=100):
        '''
        Compile the latent MCMC model so that it may be used for training.
        '''
        self.batch_size = batch_size
        starts = np.tile([self.Net.start], (batch_size,1,1)).astype('float32')
        
        ### Set up ML stuff ###
        starts = np.tile([self.Net.start], (batch_size,1,1)).astype('float32')
        def loss_ML(y_true, y_pred):
            return loss_ML_true(self.Net, starts)

        ### Set up KL stuff ###
        Pos_Base = Gaussian_Generator(size=(batch_size, self.Net.max_len, self.Net.dim-1))
        # Ang_Base = Angle_Walk_Generator(starts=starts, size=(batch_size, self.Net.max_len, 1))
        def loss_KL(y_true, y_pred):  
            return loss_KL_Walks(self.Net, starts)

        ###
        def junk_loss(y_true, y_pred):
            return 0.0
        ###

        losses = [loss_ML, loss_KL]
        w_losses = [w_ML, w_KL]
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        self.Net.Mixed_Model.compile(optimizer=opt, loss=losses, loss_weights=w_losses)

        size=(batch_size, self.Net.max_len, 2)
        starts = np.tile(self.Net.start, (size[0],1,1))
        self.Pos_Base = Gaussian_Generator(size)
        # self.Ang_Base = Angle_Walk_Generator(starts, size)

        self.Train_Patience = patience
        self.Train_Epochs = epochs


    def Train(self, x_pos, x_pos_val, weights_trn, weights_val):
        '''
        Train the model on the input samples
        '''
        num_training_samples = x_pos.shape[0]
        batches_per_epoch = math.floor(num_training_samples/self.batch_size)
        training_indices = np.arange(num_training_samples)
        validation_indices = np.arange(x_pos_val.shape[0])
        
        min = float('inf')
        best_weights = None
        steps_since_min = 0
        
        val_z_dummy_weights = z_dummy_weights = np.ones(self.batch_size) # defined here as it'll be a tiny bit quicker 
            
        for epoch in range(self.Train_Epochs):
            # New prog bar each epoch
            print("\nepoch {}/{}".format(epoch+1, self.Train_Epochs))
            progBar = tf.keras.utils.Progbar(num_training_samples)

            # shuffle
            np.random.shuffle(training_indices)
            x_pos = x_pos[training_indices]
            weights_trn = weights_trn [training_indices]
            np.random.shuffle(validation_indices) # because each epoch we validate on just one batch 
            x_pos_val = x_pos_val[validation_indices]
            weights_val = weights_val[validation_indices]
            
            # train on batches
            for batch in range(batches_per_epoch):
                x_pos_batch = x_pos[batch*self.batch_size:(batch+1)*self.batch_size]
                z_pos_batch = next(self.Pos_Base)
                x_weights_batch = weights_trn[batch*self.batch_size:(batch+1)*self.batch_size]
                t_history = self.Net.Mixed_Model.train_on_batch([x_pos_batch, z_pos_batch], sample_weight=[x_weights_batch, z_dummy_weights])
                # Update prog bar 
                values = [('Total', t_history[0]), ('ML', t_history[1]), ('KL', t_history[2])]
                progBar.update((batch)*self.batch_size, values=values)

            # test on validation data
            val_x_pos_batch = x_pos_val[:self.batch_size]
            val_z_pos_batch = next(self.Pos_Base)
            val_x_weights_batch = weights_val[:self.batch_size]
            v_history = self.Net.Mixed_Model.test_on_batch([val_x_pos_batch, val_z_pos_batch], sample_weight=[val_x_weights_batch, val_z_dummy_weights])

            # Finish progress bar
            values = [('Total', v_history[0]), ('ML', v_history[1]), ('KL', v_history[2])]
            progBar.update(num_training_samples, values=values, finalize=True)

            # Check if we can exit early
            if v_history[0] < min:
                min = v_history[0]
                best_weights = self.Net.Mixed_Model.get_weights()
                steps_since_min = 0
            else:
                steps_since_min += 1
                if steps_since_min == self.Train_Patience:
                    print('Early stopping triggered')
                    self.Net.Mixed_Model.set_weights(best_weights)
                    print('Best weights restored')
                    break


    def Acc_or_Rej(self, acc_prob, x_pos, new_x_pos, old_weights=None, new_weights=None):
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

        weights = old_weights
        # make sure, on avg, the contribution of reweighted samples is still 1
        acceptance_mask_index = np.where(acceptance_mask_x_pos)[0]
        weights_accepted = new_weights[acceptance_mask_index]
        weights_accepted = (weights_accepted/np.sum(weights_accepted)) * len(weights_accepted)      
        # scale the weights not being updated so that their contribution also averages to 1
        rejection_mask_index = np.where(np.invert(acceptance_mask_x_pos))[0]
        preserved_weights = weights[rejection_mask_index]
        preserved_weights = (preserved_weights/np.sum(preserved_weights)) * len(preserved_weights)      

        weights[acceptance_mask_index] = weights_accepted
        weights[rejection_mask_index] = preserved_weights

        return x_pos, weights, percent_accepted

    
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
                
        latent_x_old, old_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict(x_pos)
        latent_x_new, new_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict(new_x_pos)

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
        energies = self.potential(Positions, input_mode='Batch')
        energy_mask = energies < 2
        right_mask = Positions[:,:,0] > 0
        r_e_mask = np.logical_and(energy_mask,right_mask)
        not_r_e_mask = np.logical_not(r_e_mask)
        unreactive_mask = np.all(not_r_e_mask, axis=1)
        reactive_mask = np.logical_not(unreactive_mask)
        return reactive_mask
    

    def Explore(self, iterations, step_size, iter_Lang, training_split=0.8, file_loc=None, save_each=1):
        '''
        Combine local MCMC (Metropolis-Hastings) with training
        '''
        # Stuff carried through each loop #
        x_pos = self.chains
        indicies = np.arange(self.num_chains)
        links = np.expand_dims(x_pos, axis=1)
        chains = links
        weights = np.ones(self.num_chains)
        link_weights_chain = np.expand_dims(weights, axis=1)  
        reactivity = self.get_reac_mask(x_pos)

        # Data to be monitored #
        times = np.array([time()])
        ### Save what untrained network produces ###
        new_z_pos = np.random.normal(size=(self.num_chains, 32, 2))
        new_x_pos, new_x_weights = self.Transform_ZtoX_Weights(new_z_pos)
        NETWORK_links_chain = np.expand_dims(new_x_pos, axis=1)
        NETWORK_weights_chain = np.expand_dims(new_x_weights, axis=1)
        ############################################
        
        for iter in range(iterations):
            print("Iteration", iter+1,"/",iterations)
            if (iter+1) % (iter_Lang+1) == 0:    # After inter_Lang steps, resample
                # Generate Non-Local Proposal
                new_z_pos = np.random.normal(size=(self.num_chains, 32, 2))
                new_x_pos, new_x_weights = self.Transform_ZtoX_Weights(new_z_pos)
                # Save proposals to monitor network training
                NETWORK_links_chain = np.concatenate([NETWORK_links_chain, np.expand_dims(new_x_pos, axis=1)], axis=1)
                NETWORK_weights_chain = np.concatenate([NETWORK_weights_chain, np.expand_dims(new_x_weights, axis=1)], axis=1)

                # Calculate acceptance probabilitiy
                acc_prob, reactivity = self.non_local_acc_prob_calculator(x_pos, new_x_pos, reactivity)

                # Accept or Reject
                x_pos, weights, percent_accepted = self.Acc_or_Rej(acc_prob, x_pos, new_x_pos, weights, new_x_weights)
                links = np.expand_dims(x_pos, axis=1)
                chains = np.concatenate([chains, links], axis=1)
                link_weights_chain = np.concatenate([link_weights_chain, np.expand_dims(weights, axis=1)], axis=1)
                print("    accepted " + str(percent_accepted) + "% of non-local proposals")

                ######### Train the Network on the new samples ##########
                ### Adjust here to train on only reactive new samples ### 
                reactive_indicies = indicies[reactivity]
                np.random.shuffle(reactive_indicies)
                num_reactive = np.count_nonzero(reactivity)
                trai_indicies = indicies[:int(num_reactive*training_split)]
                vali_indicies = indicies[int(num_reactive*training_split):] 
                ##########################################################
                
                self.Train(x_pos[trai_indicies], x_pos[vali_indicies], 
                           weights_trn=weights[trai_indicies], weights_val=weights[vali_indicies])

                print("\n    network trained with [reweighted] non-local samples")

                times = np.append(times, time())

                ####################################
                if ((iter+1) % save_each == 0) or ((iter+1) == iterations):
                    ## Save Info ##
                    # MCMC Info # 
                    np.save(file_loc + '_MCMC_samples', chains)
                    np.save(file_loc + '_MCMC_weights', link_weights_chain)
                    # Network Info # 
                    np.save(file_loc + '_Net_samples', NETWORK_links_chain)
                    np.save(file_loc + '_Net_weights', NETWORK_weights_chain)
                    # Times and Metrics# 
                    np.save(file_loc + '_times', times)
                ####################################
                
        self.Net.net_XtoZ.save_weights(file_loc + '_net' + ".h5")
