'''
Small functions and simple layers used extensively by other classes in this directory
'''
import math
import numpy as np
import tensorflow as tf

def AngleToPeriodic(x):
    ''' Returns a tensor that is [sin(x), cos(x)] of the input x '''
    sin_x = tf.math.sin(x)
    cos_x = tf.math.cos(x)
    periodic_encoding = tf.concat([cos_x, sin_x], axis=2)
    return periodic_encoding


def conv_unit(n_type, dim, neurons=50):
    '''
    Small network that uses 1D convolutions. Used for S and T transformations that form a RealNVP block.
    
    n_type (char): Network type, Scaling network (s) or a Translation network (t)
    dim (int): Dimensionality of the input tensor
    neurons (int): neurons used in final few densely connected layers of the conv_unit
    '''
    n_type = n_type.lower()
    
    if n_type == 's':
        activation = 'tanh'
    elif n_type == 't':
        activation = 'relu'
    else:
        raise Exception("Only transformation_type 's' and 't' are implemented")
    
    conv_base = [tf.keras.layers.Conv1D(16, kernel_size=3, activation=activation, padding='same'),
                 tf.keras.layers.Conv1D(16, kernel_size=3, activation=activation, padding='same')]
                
    dense = [tf.keras.layers.Dense(neurons, activation=activation),
            tf.keras.layers.Dense(neurons, activation=activation)]
    
    if n_type == 't':
        final_layer = [tf.keras.layers.Dense(dim, activation='linear')]
    elif n_type == 's':
        final_layer = [tf.keras.layers.Dense(dim, activation='linear',
                                         kernel_initializer=tf.keras.initializers.Zeros(),
                                         bias_initializer=tf.keras.initializers.Zeros())]
    
    return conv_base + dense + final_layer
    
    
def connect(input_layer, layers):
    '''
    Connects a given sequence of layers.
    '''
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer


def PM_pi(Traj):
    '''
    Converts an input sequence of angles to be in the interval [-pi, +pi]
    '''
    return (Traj +math.pi)%(2*math.pi) -math.pi
