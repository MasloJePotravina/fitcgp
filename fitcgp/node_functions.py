import numpy as np

#Default functions
def logical_and(inputs):
    return inputs[0] and inputs[1]

def logical_or(inputs):
    return inputs[0] or inputs[1]

def logical_not(inputs):
    return not inputs[0]

def logical_xor(inputs):
    return inputs[0] != inputs[1]



#NumPy functions
def logical_and_np(inputs):
    return np.logical_and(inputs[0], inputs[1])

def logical_or_np(inputs):
    return np.logical_or(inputs[0], inputs[1])

def logical_not_np(inputs):
    return np.logical_not(inputs[0])

def logical_xor_np(inputs):
    return np.logical_xor(inputs[0], inputs[1])


#NumPy bitwise functions
def logical_and_bitwise_np(inputs):
    return np.bitwise_and(inputs[0], inputs[1])

def logical_or_bitwise_np(inputs):
    return np.bitwise_or(inputs[0], inputs[1])

def logical_not_bitwise_np(inputs):
    return np.bitwise_not(inputs[0])

def logical_xor_bitwise_np(inputs):
    return np.bitwise_xor(inputs[0], inputs[1])