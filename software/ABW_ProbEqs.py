import numpy as np
import math

def Target_Prob(Net, pos, ang):
    ### Append Starts
    starts = np.tile([Net.start], (len(pos),1,1))
    Positions = np.concatenate((starts[:,:,:2], pos), axis=1)   
    Angles = np.concatenate((starts[:,:,2:], ang), axis=1)
    
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

    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    # # Calculate log prob of the angles
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # ang_log_prob = 1/(Net.rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
    ang_log_prob = 0
    ### SIMPLIFY BY OVERLOOKING ANGLES ###
   
    return np.exp(pos_log_prob + ang_log_prob)

def Log_Target_Prob(Net, pos, ang):
    ### Append Starts
    starts = np.tile([Net.start], (len(pos),1,1))
    Positions = np.concatenate((starts[:,:,:2], pos), axis=1)   
    Angles = np.concatenate((starts[:,:,2:], ang), axis=1)
    
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

    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    # # Calculate log prob of the angles
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # ang_log_prob = 1/(Net.rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
    ang_log_prob = 0
    ### SIMPLIFY BY OVERLOOKING ANGLES ###

    return pos_log_prob + ang_log_prob

def Latent_Prob(Net, pos, ang):
    
    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    # # Angles
    # starts = np.tile([Net.start], (len(pos),1,1))
    # Angles = np.concatenate((starts[:,:,2:], ang), axis=1)
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # ang_latent_log_prob = 1/(Net.basis_rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
    ang_latent_log_prob = 0
    ### SIMPLIFY BY OVERLOOKING ANGLES ###

    # Pos
    pos_latent_log_prob = - 0.5 * np.sum(pos**2, axis=(1,2))

    return np.exp(ang_latent_log_prob + pos_latent_log_prob)

def Log_Latent_Prob(Net, pos, ang):
    ### SIMPLIFY BY OVERLOOKING ANGLES ###
    # # Angles
    # starts = np.tile([Net.start], (len(pos),1,1))
    # Angles = np.concatenate((starts[:,:,2:], ang), axis=1)
    # Ang_Increments = Angles[:,1:] - Angles[:,:-1]
    # Ang_Increments = (Ang_Increments +math.pi)%(2*math.pi) -math.pi
    # ang_latent_log_prob = 1/(Net.basis_rot_coeff**2) * np.sum(np.cos(Ang_Increments), axis=(1,2))
    ang_latent_log_prob = 0
    ### SIMPLIFY BY OVERLOOKING ANGLES ###

    # Pos
    pos_latent_log_prob = - 0.5 * np.sum(pos**2, axis=(1,2))

    return ang_latent_log_prob + pos_latent_log_prob