import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
sys.path.append(r'/home/solomon.asghar/NF_TPS/software/')
from util import *
from time import time
from ABW_ProbEqs import *


def Gaussian_Generator(size):
    '''
    Generates a Gaussian walk to be used as a prior. 
    '''
    while True: # or we only generate once
        Gaussian = np.random.normal(size=size)
        yield Gaussian
        
def Angle_Walk_Generator(Net, starts, size):
    '''
    Generates a Gaussian walk to be used as a prior. 
    '''
    while True: # or we only generate once
        # Create rotational walk
        Ang_Increments = np.random.vonmises(0, 1/(Net.rot_coeff**2), size=(size[0], size[1], 1))
        Ang_Increments = np.concatenate((starts[:,:,2:], Ang_Increments), axis=1)
        Ang_Walk = np.cumsum(Ang_Increments, axis=1)
        Ang_Walk = PM_pi(Ang_Walk)

        # Combine to yeild walk
        yield Ang_Walk[:,1:]

def loss_ML_true(Net, starts, dX=0.001):
    '''
    Kullback-Leiber loss for target distribution of walks.
    '''
    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    # Angles = tf.concat((starts[:,:,2:], Net.output_z_angles), axis=1)
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # basis_kappa = 1/(Net.rot_coeff**2)
    # ang_latent_log_prob = basis_kappa * tf.reduce_sum(tf.cos(Ang_Increments))
    ang_latent_log_prob = 0
    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    
    pos_latent_log_prob = 0.5 * tf.reduce_sum(Net.output_z_positions**2)

    return pos_latent_log_prob + ang_latent_log_prob - tf.reduce_sum(Net.log_Rxz)

def loss_KL_Walks(Net, starts, dX=0.001):
    '''
    Kullback-Leiber loss for target distribution of walks.
    '''
    # Prep by appending start
    Positions = tf.concat((starts[:,:,:2], Net.output_x_positions), axis=1)   
    Angles = tf.concat((starts[:,:,2:], Net.output_x_angles), axis=1)
    
    # Calculate log prob of the positions
    Pos_Increments = Positions[:,1:] - Positions[:,:-1]

    Sin_Angles = tf.math.sin(Angles)
    Cos_Angles = tf.math.cos(Angles)
    
    grad_U = Net.potential_grad(Positions[:,:-1], mode='TF')
    Activity = tf.concat([Cos_Angles[:,:-1], Sin_Angles[:,:-1]], axis=2)
    
    Translational_Noise = Pos_Increments + Net.pot_coeff*grad_U - Net.act_coeff*Activity
    Translational_Noise = Translational_Noise/Net.tran_coeff
    pos_log_prob = 0.5 * tf.reduce_sum(Translational_Noise**2)
    ##########

    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    # # Calculate log prob of the angles
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # kappa = 1/(Net.rot_coeff**2)
    # ang_log_prob = - kappa * tf.reduce_sum(tf.cos(Ang_Increments))
    ang_log_prob = 0
    ### SIMPLIFY BY OVERLOOKING ANGLES ###

    return pos_log_prob + ang_log_prob - tf.reduce_sum(Net.log_Rzx)


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
        inputs = [[self.Net.input_x_positions, self.Net.input_x_angles], [self.Net.input_z_positions, self.Net.input_z_angles]]
        outputs = [[self.Net.output_z_positions, self.Net.output_z_angles], [self.Net.output_x_positions, self.Net.output_x_angles]]
        self.Net.Mixed_Model = tf.keras.Model(inputs, outputs)


    def Build_XtoZ_log_Rxz_Model(self):
        '''
        Build a model with the right outputs for latent MCMC.
        '''
        inputs_x = [self.Net.input_x_positions, self.Net.input_x_angles]
        outputs_z = [self.Net.output_z_positions, self.Net.output_z_angles, self.Net.log_Rxz]
        self.Net.transform_XtoZ_log_Rxz = tf.keras.Model(inputs_x, outputs_z)

    def Build_ZtoX_log_Rzx_Model(self):
        '''
        Build a model with the right outputs for latent MCMC.
        '''
        inputs_z = [self.Net.input_z_positions, self.Net.input_z_angles]
        outputs_x = [self.Net.output_x_positions, self.Net.output_x_angles, self.Net.log_Rzx]
        self.Net.net_ZtoX_log_Rzx = tf.keras.Model(inputs_z, outputs_x)
    
    def Transform_ZtoX_Weights(self, z_pos, z_ang):
        x_pos, x_ang, log_Rzx = self.Net.net_ZtoX_log_Rzx.predict([z_pos, z_ang])
        starts = np.tile([self.Net.start], (len(z_pos),1,1)).astype('float32')

        # Prepare trajectories       
        Positions = np.concatenate((starts[:,:,:2], x_pos), axis=1)
        Angles = np.concatenate((starts[:,:,2:], x_ang), axis=1)
        #
        Latent_Positions = z_pos
        Latent_Angles = np.concatenate((starts[:,:,2:], z_ang), axis=1)

        # Calculate probs of latent vectors in generated latent distribution
        ## Prob of generated samples in target dist
        ### Calculate log prob of the positions
        Pos_Increments = Positions[:,1:] - Positions[:,:-1]
        Sin_Angles = np.sin(Angles)
        Cos_Angles = np.cos(Angles)
        grad_U = self.Net.potential_grad(Positions[:,:-1], mode='numpy')
        Activity_Vector = np.concatenate([Cos_Angles[:,:-1], Sin_Angles[:,:-1]], axis=2)
        Translational_Noise = Pos_Increments + self.Net.pot_coeff*grad_U - self.Net.act_coeff*Activity_Vector
        Translational_Noise = Translational_Noise/self.Net.tran_coeff
        pos_log_prob = - 0.5 * np.sum(Translational_Noise**2, axis=(1,2))

        ### SIMPLIFY BY OVERLOOKING ANGLES ###
        # ### Calculate log prob of the angles
        # kappa = np.array([1/(self.Net.rot_coeff**2)])
        # basis_kappa = np.array([1/(self.Net.basis_rot_coeff**2)])
        # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
        # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
        # ang_log_prob = kappa * np.sum(np.cos(Ang_Increments), axis=(1,2))
        ang_log_prob = 0
        ### SIMPLIFY BY OVERLOOKING ANGLES ###

        ###
        log_prob_gens_in_target = pos_log_prob + ang_log_prob
        ### Add Rzx to get log probs of latent in generated latent
        log_prob_latent_in_gen_latent = log_prob_gens_in_target + log_Rzx

        # Calculate probs of latent vectors in true latent distribution
        ## log prob of the latent positions
        pos_latent_log_prob = - 0.5 * np.sum(Latent_Positions**2, axis=(1,2))

        ### SIMPLIFY BY OVERLOOKING ANGLES ###
        # ## log prob of the latent angles
        # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
        # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
        # ang_latent_log_prob = basis_kappa * np.sum(np.cos(Ang_Increments), axis=(1,2))
        ang_latent_log_prob = 0
        ### SIMPLIFY BY OVERLOOKING ANGLES ###

        ###
        log_probs_latent_in_tru_latent = pos_latent_log_prob + ang_latent_log_prob
        ###
        
        # Calculate weights
        W_x = np.exp(log_prob_latent_in_gen_latent - log_probs_latent_in_tru_latent)
        
        return Positions[:,1:], Angles[:,1:], W_x


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
        self.Pos_Base = Gaussian_Generator(size=(batch_size, self.Net.max_len, self.Net.dim-1))
        self.Ang_Base = Angle_Walk_Generator(self.Net, starts=starts, size=(batch_size, self.Net.max_len, 1))
        def loss_KL(y_true, y_pred):  
            return loss_KL_Walks(self.Net, starts)

        ###
        def junk_loss(y_true, y_pred):
            return 0.0
        ###

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        losses = [loss_ML, loss_KL,  junk_loss,  junk_loss]
        w_losses = [w_ML, w_KL, 0, 0]

        self.Net.Mixed_Model.compile(optimizer=opt, loss=losses, loss_weights=w_losses)

        self.Train_Patience = patience
        self.Train_Epochs = epochs


    def Train(self, x_pos, x_ang, x_pos_val, x_ang_val, weights_trn, weights_val):
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
            x_ang = x_ang[training_indices]
            weights_trn = weights_trn [training_indices]
            np.random.shuffle(validation_indices) # because each epoch we validate on just one batch 
            x_pos_val = x_pos_val[validation_indices]
            x_ang_val = x_ang_val[validation_indices]
            weights_val = weights_val[validation_indices]
            
            # train on batches
            for batch in range(batches_per_epoch):
                x_pos_batch = x_pos[batch*self.batch_size:(batch+1)*self.batch_size]
                x_ang_batch = x_ang[batch*self.batch_size:(batch+1)*self.batch_size]
                trn_batch_weights = weights_trn[batch*self.batch_size:(batch+1)*self.batch_size]
                z_pos_batch = next(self.Pos_Base)
                z_ang_batch = next(self.Ang_Base)
                t_history = self.Net.Mixed_Model.train_on_batch([x_pos_batch, x_ang_batch, z_pos_batch, z_ang_batch],
                                                                sample_weight=[trn_batch_weights, trn_batch_weights, z_dummy_weights, z_dummy_weights])
                # Update prog bar 
                values = [('Total', t_history[0]), ('ML', t_history[1]), ('KL', t_history[2])]
                progBar.update((batch)*self.batch_size, values=values)

            # test on validation data
            val_x_pos_batch = x_pos_val[:self.batch_size]
            val_x_ang_batch = x_ang_val[:self.batch_size]
            val_batch_weights = weights_val[:self.batch_size]
            val_z_pos_batch = next(self.Pos_Base)
            val_z_ang_batch = next(self.Ang_Base)
            v_history = self.Net.Mixed_Model.test_on_batch([val_x_pos_batch, val_x_ang_batch, val_z_pos_batch, val_z_ang_batch],
                                                           sample_weight=[val_batch_weights, val_batch_weights, val_z_dummy_weights, val_z_dummy_weights])

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
                    
    def Train_weightless(self, x_pos, x_ang, x_pos_val, x_ang_val):
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
    
        for epoch in range(self.Train_Epochs):
            # New prog bar each epoch
            print("\nepoch {}/{}".format(epoch+1, self.Train_Epochs))
            progBar = tf.keras.utils.Progbar(num_training_samples)

            # shuffle
            np.random.shuffle(training_indices)
            x_pos = x_pos[training_indices]
            x_ang = x_ang[training_indices]
            np.random.shuffle(validation_indices) # because each epoch we validate on just one batch 
            x_pos_val = x_pos_val[validation_indices]
            x_ang_val = x_ang_val[validation_indices]
            
            # train on batches
            for batch in range(batches_per_epoch):
                x_pos_batch = x_pos[batch*self.batch_size:(batch+1)*self.batch_size]
                x_ang_batch = x_ang[batch*self.batch_size:(batch+1)*self.batch_size]
                z_pos_batch = next(self.Pos_Base)
                z_ang_batch = next(self.Ang_Base)
                t_history = self.Net.Mixed_Model.train_on_batch([x_pos_batch, x_ang_batch, z_pos_batch, z_ang_batch])
                # Update prog bar 
                values = [('Total', t_history[0]), ('ML', t_history[1]), ('KL', t_history[2])]
                progBar.update((batch)*self.batch_size, values=values)

            # test on validation data
            val_x_pos_batch = x_pos_val[:self.batch_size]
            val_x_ang_batch = x_ang_val[:self.batch_size]
            val_z_pos_batch = next(self.Pos_Base)
            val_z_ang_batch = next(self.Ang_Base)
            v_history = self.Net.Mixed_Model.test_on_batch([val_x_pos_batch, val_x_ang_batch, val_z_pos_batch, val_z_ang_batch])

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


    def Acc_or_Rej(self, acc_prob, x_pos, x_ang, new_x_pos, new_x_ang, old_weights=None, new_weights=None):
        '''
        Accept or reject the new trajs based on the acc_prob
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

        if new_weights is None:
            return x_pos, x_ang, percent_accepted
        else:
            weights = old_weights
            acceptance_mask_index = np.where(acceptance_mask_x_pos)[0]
            # make sure, on avg, the contribution of reweighted samples is still 1
            weights_accepted = new_weights[acceptance_mask_index]
            weights_accepted = (weights_accepted/np.sum(weights_accepted)) * len(weights_accepted)      
            # scale the weights not being updated so that their contribution also averages to 1
            rejection_mask_index = np.where(np.invert(acceptance_mask_x_pos))[0]
            preserved_weights = weights[rejection_mask_index]
            preserved_weights = (preserved_weights/np.sum(preserved_weights)) * len(preserved_weights)      
            
            weights[acceptance_mask_index] = weights_accepted
            weights[rejection_mask_index] = preserved_weights

            return x_pos, x_ang, weights, percent_accepted

    
    def non_local_acc_prob_calculator(self, x_pos, x_ang, new_x_pos, new_x_ang, reactivity):
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
        x_ang = x_ang[need_to_calculate_probs]
        new_x_pos = new_x_pos[need_to_calculate_probs]
        new_x_ang = new_x_ang[need_to_calculate_probs]
                
        latent_old_pos, latent_old_ang, old_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict([x_pos, x_ang])
        latent_new_pos, latent_new_ang, new_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict([new_x_pos, new_x_ang])
        
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
    
    
    def non_local_acc_prob_calculator_all(self, x_pos, x_ang, new_x_pos, new_x_ang):
        '''
        Calculate log of non-local acceptance probability
        '''
        # Create array of zeros for all probabilities
        latent_old_pos, latent_old_ang, old_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict([x_pos, x_ang])
        latent_new_pos, latent_new_ang, new_log_R_xz = self.Net.transform_XtoZ_log_Rxz.predict([new_x_pos, new_x_ang])
        
        log_prob_old = Log_Target_Prob(self.Net, x_pos, x_ang)
        log_prob_latent_old = Log_Latent_Prob(self.Net, latent_old_pos, latent_old_ang)
        log_prob_new = Log_Target_Prob(self.Net, new_x_pos, new_x_ang)
        log_prob_latent_new = Log_Latent_Prob(self.Net, latent_new_pos, latent_new_ang)

        log_acc_prob_numerator = log_prob_latent_old + old_log_R_xz + log_prob_new
        log_acc_prob_denominator = log_prob_old + log_prob_latent_new + new_log_R_xz
        log_acc_prob = log_acc_prob_numerator - log_acc_prob_denominator
        log_acc_prob = np.where(log_acc_prob>0, 0, log_acc_prob)
        
        return log_acc_prob 


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
    

    def Explore(self, iterations, step_size, iter_Lang, training_split=0.8, file_loc=None, save_each=1, constrain_to_reactive=True, use_weights=True):
        '''
        Combine local MCMC (Metropolis-Hastings) with training
        '''
        # Stuff carried through each loop #
        x_pos = self.chains[:,:,:2]
        x_ang = self.chains[:,:,2:]
        indicies = np.arange(self.num_chains)
        links = np.concatenate([x_pos,x_ang], axis=2)
        chains = np.expand_dims(links, axis=1)
        weights = np.ones(self.num_chains)
        link_weights_chain = np.expand_dims(weights, axis=1)  
        reactivity = self.get_reac_mask(x_pos)

        # Make starts #
        starts = np.tile([self.Net.start], (self.num_chains,1,1))
        # Data to be monitored #
        times = np.array([time()])
        ### Save what untrained network produces ###
        new_z_pos = np.random.normal(size=(self.num_chains, self.Net.max_len, 2))
        ####
        Ang_Increments = np.random.vonmises(0, 1/(self.Net.rot_coeff**2), size=(self.num_chains, self.Net.max_len, 1))
        Ang_Increments = np.concatenate((starts[:,:,2:], Ang_Increments), axis=1)
        Ang_Walk = np.cumsum(Ang_Increments,axis=1)
        new_z_ang = PM_pi(Ang_Walk)[:,1:]
        ####
        new_x_pos, new_x_ang, new_x_weights = self.Transform_ZtoX_Weights(new_z_pos, new_z_ang)        
        NETWORK_links_chain = np.expand_dims(np.concatenate([new_x_pos, new_x_ang], axis=2), axis=1)
        NETWORK_weights_chain = np.expand_dims(new_x_weights, axis=1)
        ############################################
        
        starts = np.tile([self.Net.start], (self.num_chains,1,1))
        
        for iter in range(iterations):
            print("Iteration", iter+1,"/",iterations)
            if (iter+1) % (iter_Lang+1) == 0:    # After inter_Lang steps, resample
                # Generate Latent Proposal
                new_z_pos = np.random.normal(size=(self.num_chains, self.Net.max_len, 2))
                Ang_Increments = np.random.vonmises(0, 1/(self.Net.rot_coeff**2), size=(self.num_chains, self.Net.max_len, 1))
                Ang_Increments = np.concatenate((starts[:,:,2:], Ang_Increments), axis=1)
                Ang_Walk = np.cumsum(Ang_Increments,axis=1)
                new_z_ang = PM_pi(Ang_Walk)[:,1:]
                # Project to configuration space
                new_x_pos, new_x_ang, new_x_weights = self.Transform_ZtoX_Weights(new_z_pos, new_z_ang)
                # Save proposals to monitor network training
                NETWORK_links = np.expand_dims(np.concatenate([new_x_pos, new_x_ang], axis=2), axis=1)
                NETWORK_links_chain = np.concatenate([NETWORK_links_chain, NETWORK_links], axis=1)
                NETWORK_weights_chain = np.concatenate([NETWORK_weights_chain, np.expand_dims(new_x_weights, axis=1)], axis=1)

                # Calculate acceptance probabilitiy
                if constrain_to_reactive is True:
                    acc_prob, reactivity = self.non_local_acc_prob_calculator(x_pos, x_ang, new_x_pos, new_x_ang, reactivity)
                elif constrain_to_reactive is False:
                    acc_prob = self.non_local_acc_prob_calculator_all(x_pos, x_ang, new_x_pos, new_x_ang)
                
                # Accept or Reject
                x_pos, x_ang, weights, percent_accepted = self.Acc_or_Rej(acc_prob, x_pos, x_ang, new_x_pos, new_x_ang, weights, new_x_weights)
                links = np.expand_dims(np.concatenate([x_pos, x_ang], axis=2), axis=1)
                chains = np.concatenate([chains, links], axis=1)
                link_weights_chain = np.concatenate([link_weights_chain, np.expand_dims(weights, axis=1)], axis=1)
                print("    accepted " + str(percent_accepted) + "% of non-local proposals")

                ######### Train the Network on the new samples ##########
                if constrain_to_reactive is True:
                    reactive_indicies = indicies[reactivity]
                    np.random.shuffle(reactive_indicies)
                    num_reactive = np.count_nonzero(reactivity)
                    trai_indicies = indicies[:int(num_reactive*training_split)]
                    vali_indicies = indicies[int(num_reactive*training_split):]                
                elif constrain_to_reactive is False:
                    reactive_indicies = indicies
                    np.random.shuffle(reactive_indicies)
                    num_reactive = np.count_nonzero(reactivity)
                    trai_indicies = indicies[:int(num_reactive*training_split)]
                    vali_indicies = indicies[int(num_reactive*training_split):]                
                ##########################################################
                
                if use_weights is True:
                    self.Train(x_pos[trai_indicies], x_ang[trai_indicies], x_pos[vali_indicies], x_ang[vali_indicies],
                               weights_trn=weights[trai_indicies], weights_val=weights[vali_indicies])
                elif use_weights is False:
                    self.Train_weightless(x_pos[trai_indicies], x_ang[trai_indicies], x_pos[vali_indicies], x_ang[vali_indicies])
                
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
