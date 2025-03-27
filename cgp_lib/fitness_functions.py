import numpy as np




def bitwise_fitness(outputs, expected_outputs):
    """Calculate fitness based on bitwise correctness for a 2-bit adder."""
    total_bits = len(outputs) * len(outputs[0])  # Total number of bits in all outputs
    correct_bits = sum(
        1 for i in range(len(outputs)) 
        for j in range(len(outputs[i])) if outputs[i][j] == expected_outputs[i][j]
    )
    return correct_bits / total_bits  # Fitness between 0.0 and 1.0


def bitwise_fitness_np(outputs, expected_outputs):
    return np.count_nonzero(outputs == expected_outputs) / outputs.size

def bitwise_fitness_bitwise_np(outputs, expected_outputs):
    mismatched_bits = np.bitwise_xor(outputs, expected_outputs)
    
    # Convert the result into individual bits (we use np.unpackbits and view as np.uint8)
    #mismatched_bit_count = np.sum(np.unpackbits(mismatched_bits.view(np.uint8)))
    mismatched_bit_count = np.sum(np.bitwise_count(mismatched_bits))


    # Divide by the total number of bits (outputs.size * 64 bits per uint64)
    total_bits = outputs.size * 64

    return 1 - mismatched_bit_count / total_bits

#TODO:try to optimize
def mse_fitness_bitwise_np(outputs, expected_outputs):

    #get number of outputs
    num_outputs = len(outputs)

    # Step 1: Unpack to get individual bits (reshaping to maintain structure)
    unpacked_outputs = np.unpackbits(outputs.view(np.uint8), axis=-1).T
    unpacked_expected = np.unpackbits(expected_outputs.view(np.uint8), axis=-1).T
    

    # Step 2: Convert from binary to decimal values
    decimal_outputs = unpacked_outputs.dot(1 << np.arange(num_outputs)[::-1])  # Convert each row of num_outputs bits to decimal
    decimal_expected = unpacked_expected.dot(1 << np.arange(num_outputs)[::-1])  


    # Step 3: Compute MSE
    mse = np.mean((decimal_outputs - decimal_expected) ** 2)
    
    return mse


#Error rate
def error_rate(outputs, expected_outputs):
    """Calculate the error rate of the individual (percentage of incorrect outputs compared to expected outputs)."""
    print(outputs[0])
    print(expected_outputs[0])
    return sum([1 for i in range(len(outputs)) if outputs[i] != expected_outputs[i]]) / len(outputs)

def error_rate_np(outputs, expected_outputs):
    outputs_transposed = outputs.T
    expected_outputs_transposed = expected_outputs.T
    incorrect_cases = np.any(outputs_transposed != expected_outputs_transposed, axis=1)
    error_rate = np.sum(incorrect_cases) / incorrect_cases.size
    return error_rate


def error_rate_bitwise_np(outputs, expected_outputs):
    unpacked_outputs = np.unpackbits(outputs.view(np.uint8), axis=-1).T
    unpacked_expected = np.unpackbits(expected_outputs.view(np.uint8), axis=-1).T

    incorrect_cases = np.any(unpacked_outputs != unpacked_expected, axis=1)

    error_rate = np.sum(incorrect_cases) / incorrect_cases.size

    return error_rate