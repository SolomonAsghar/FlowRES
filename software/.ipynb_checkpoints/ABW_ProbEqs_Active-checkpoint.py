'''
Contains functions to work out probability equations required for FlowRes training and for calculating acceptance probability.
'''
import numpy as np
import math

def Target_Prob(Net, pos, ang):
    '''
    Probability of the target distribution, i.e. pdf of the transition path distribution.

    Net [FlowNet]: Network used to generate the paths.
    pos [numpy array]: Array containing the positions we want to work out prob of. 
    ang [numpy array]: Array containing the angles we want to work out prob of. 
    '''
    ### Append Starts
    starts_pos = np.tile(Net.start[:2], (len(pos),1,1))
    Positions = np.concatenate((starts_pos, pos), axis=1)
    starts_ang = ang[:,:1] - np.random.vonmises(0, 1/(Net.rot_coeff**2), size=(len(ang), 1, 1))
    Angles = np.concatenate([starts_ang, ang], axis=1)
    
    # Calculate log prob of the positions
    Pos_Increments = Positions[:,1:] - Positions[:,:-1]

    Sin_Angles = np.sin(Angles)
    Cos_Angles = np.cos(Angles)
    
    grad_U = Net.potential_grad(Positions[:,:-1], mode='numpy')
    Activity = np.concatenate([Cos_Angles[:,:-1], Sin_Angles[:,:-1]], axis=2)
    
    Translational_Noise = Pos_Increments + Net.pot_coeff*grad_U - Net.act_coeff*Activity
    Translational_Noise = Translational_Noise/Net.tran_coeff
    pos_log_prob = - 0.5 * np.sum(Translational_Noise**2, axis=(1,2))
    ##########

    # Calculate log prob of the angles
    Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    ang_log_prob = 1/(Net.rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
    ### SIMPLIFY BY OVERLOOKING ANGLES ###
   
    return np.exp(pos_log_prob + ang_log_prob)

def Log_Target_Prob(Net, pos, ang):
    '''
    Log of probability of the target distribution, i.e. pdf of the transition path distribution.

    Net [FlowNet]: Network used to generate the paths.
    pos [numpy array]: Array containing the positions we want to work out prob of. 
    ang [numpy array]: Array containing the angles we want to work out prob of. 
    ''' 
    ### Append Starts
    starts_pos = np.tile(Net.start[:2], (len(pos),1,1))
    Positions = np.concatenate((starts_pos, pos), axis=1)
    starts_ang = ang[:,:1] - np.random.vonmises(0, 1/(Net.rot_coeff**2), size=(len(ang), 1, 1))
    Angles = np.concatenate([starts_ang, ang], axis=1)
    
    # Calculate log prob of the positions
    Pos_Increments = Positions[:,1:] - Positions[:,:-1]

    Sin_Angles = np.sin(Angles)
    Cos_Angles = np.cos(Angles)
    
    grad_U = Net.potential_grad(Positions[:,:-1], mode='numpy')
    Activity = np.concatenate([Cos_Angles[:,:-1], Sin_Angles[:,:-1]], axis=2)
    
    Translational_Noise = Pos_Increments + Net.pot_coeff*grad_U - Net.act_coeff*Activity
    Translational_Noise = Translational_Noise/Net.tran_coeff
    pos_log_prob = - 0.5 * np.sum(Translational_Noise**2, axis=(1,2))
    ##########

    # Calculate log prob of the angles
    Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    ang_log_prob = 1/(Net.rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
    ### SIMPLIFY BY OVERLOOKING ANGLES ###

    return pos_log_prob + ang_log_prob

def Latent_Prob(Net, pos, ang):
    '''
    Probability of samples from the base distribution.

    pos [numpy array]: Positions from base path that we want to work out the probability of.
    ang [numpy array]: Array containing the angles we want to work out prob of. 
    '''
    ### Append Starts
    starts_pos = np.tile(Net.start[:2], (len(pos),1,1))
    Positions = np.concatenate((starts_pos, pos), axis=1)
    
    # Angles
    Angles = ang
    Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    ang_latent_log_prob = 1/(Net.rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))

    # Pos
    pos_latent_log_prob = - 0.5 * np.sum(pos**2, axis=(1,2))

    return np.exp(ang_latent_log_prob + pos_latent_log_prob)

def Log_Latent_Prob(Net, pos, ang):
    '''
    Log of probability of samples from the base distribution.
    pos [numpy array]: Positions from base path that we want to work out the probability of.
    ang [numpy array]: Array containing the angles we want to work out prob of. 
    '''
    ### Append Starts
    starts_pos = np.tile(Net.start[:2], (len(pos),1,1))
    Positions = np.concatenate((starts_pos, pos), axis=1)
    
    # Angles
    Angles = ang
    Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    ang_latent_log_prob = 1/(Net.rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))

    # Pos
    pos_latent_log_prob = - 0.5 * np.sum(pos**2, axis=(1,2))

    return ang_latent_log_prob + pos_latent_log_prob