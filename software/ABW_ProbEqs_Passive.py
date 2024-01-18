import numpy as np
import math

def Target_Prob(Net, pos):
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
    return np.exp(- 0.5 * np.sum(pos**2, axis=(1,2)))

def Log_Latent_Prob(pos):
    return - 0.5 * np.sum(pos**2, axis=(1,2))
