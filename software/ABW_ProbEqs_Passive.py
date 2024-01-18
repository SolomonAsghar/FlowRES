'''
Contains functions to work out probability equations required for FlowRes training and for calculating acceptance probability.
'''
import numpy as np
import math

def Target_Prob(Net, pos):
    '''
    Probability of the target distribution, i.e. pdf of the transition path distribution.

    Net [FlowNet]: Network used to generate the paths.
    pos [numpy array]: Array containing the positions we want to work out prob of. 
    '''
    ### Append Starts
    starts = np.tile([Net.start], (len(pos),1,1))
    Positions = np.concatenate((starts[:,:,:2], pos), axis=1)

    ### Calculate log prob of the positions
    Pos_Increments = Positions[:,1:] - Positions[:,:-1]
    grad_U = Net.potential_grad(Positions[:,:-1], mode='numpy')
    Translational_Noise = Pos_Increments + Net.pot_coeff*grad_U 
    Translational_Noise = Translational_Noise/Net.tran_coeff
    return np.exp(- 0.5 * np.sum(Translational_Noise**2, axis=(1,2)))

def Log_Target_Prob(Net, pos):
    '''
    Log of probability of the target distribution, i.e. pdf of the transition path distribution.

    Net [FlowNet]: Network used to generate the paths.
    pos [numpy array]: Array containing the positions we want to work out prob of. 
    '''    
    ### Append Starts
    starts = np.tile([Net.start], (len(pos),1,1))
    Positions = np.concatenate((starts[:,:,:2], pos), axis=1)

    ### Calculate log prob of the positions
    Pos_Increments = Positions[:,1:] - Positions[:,:-1]
    grad_U = Net.potential_grad(Positions[:,:-1], mode='numpy')
    Translational_Noise = Pos_Increments + Net.pot_coeff*grad_U 
    Translational_Noise = Translational_Noise/Net.tran_coeff
    return - 0.5 * np.sum(Translational_Noise**2, axis=(1,2))

def Latent_Prob(pos):
    '''
    Probability of samples from the base distribution, in the passive case this is simply the Gaussian pdf. 

    pos [numpy array]: Positions from base path that we want to work out the probability of.
    '''
    return np.exp(- 0.5 * np.sum(pos**2, axis=(1,2)))

def Log_Latent_Prob(pos):
    '''
    Log of probability of samples from the base distribution, in the passive case this is simply the Gaussian pdf. 

    pos [numpy array]: Positions from base path that we want to work out the probability of.
    '''
    return - 0.5 * np.sum(pos**2, axis=(1,2))
