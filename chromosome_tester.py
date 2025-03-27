import cgp_lib
import numpy as np






def convert_gt_table_to_np(gt_table, data_type, bitwise_parallelization):
        inputs, outputs = map(lambda x: np.array(x, dtype=data_type), gt_table)
        inputs= inputs.T
        outputs = outputs.T
        if bitwise_parallelization:
            inputs = np.packbits(inputs.reshape(16, -1).copy(), axis=-1).view(np.uint64)
            outputs = np.packbits(outputs.reshape(9, -1).copy(), axis=-1).view(np.uint64)
        return inputs, outputs



def load_chromosome_from_file(file_name):
    with open(file_name, "r") as file:
        chromosome = [int(x) for x in file.read().strip().split(",")]
    return chromosome




def get_active_nodes(individual):
    active_nodes = set()
    to_be_processed = list(individual.chromosome[-outputs:])


    while to_be_processed:
        processed_node_id = to_be_processed.pop()
        if processed_node_id < inputs or processed_node_id in active_nodes:
            continue
        active_nodes.add(processed_node_id)

        to_be_processed.extend(
        gene for i in range(arity)
        if (gene := individual.chromosome[(processed_node_id - inputs) * (arity + 1) + i]) >= inputs
    )

    return list(active_nodes)

def get_function_gene(individual, node_id, arity, inputs):
    """
    Get the function gene of a node
    """
    return individual.chromosome[(node_id - inputs) * (arity + 1) + arity]

def get_input_genes(individual, node_id, arity, inputs):
    """
    Get the input genes of a node
    """
    start_idx = (node_id - inputs) * (arity + 1)
    return individual.chromosome[start_idx : start_idx + arity]

def execute_individual(individual, np_workspace=None):

    active_nodes = get_active_nodes(individual)
    
    
    individual_outputs = [None] * len(gt_table[0])

    function_genes = {}
    input_genes = {}

    for node_id in active_nodes:
        function_genes[node_id] = get_function_gene(individual, node_id, arity, inputs)
        input_genes[node_id] = get_input_genes(individual, node_id, arity, inputs)

    #print(function_genes)
    
    #pre allocate a 2d array to store node outputs, second dimension is the same as the first dimension of the ground truth table, first dimension is the number of nodes and inputs 
    node_outputs = np_workspace
    #copy the input values to the first columns of the node_outputs array
    node_outputs[:inputs, :] = gt_table[0]
    
    for node_id in active_nodes:
        function_gene = function_genes[node_id]
        input_genes_list = input_genes[node_id]
        node_inputs = node_outputs[input_genes_list]
        node_outputs[node_id, :] = node_functions_bitwise_np[function_gene](node_inputs)
    output_genes = individual.chromosome[-outputs:]
    individual_outputs = node_outputs[output_genes]
    return individual_outputs

def numpy_64_to_decimal(binary_input, rows):
    binary_input = np.unpackbits(binary_input.view(np.uint8), axis=-1).T
    print(binary_input)
    #convert the binary rows bits to decimal
    powers_of_two = 2 ** np.arange(rows)[::-1]  # [256, 128, 64, ..., 2, 1]
    decimal_outputs = binary_input @ powers_of_two  # Matrix multiplication for fast conversion
    print(decimal_outputs)
    return decimal_outputs



arity = 2
inputs = 16
outputs = 9
node_functions_bitwise_np = [cgp_lib.bitwise_and_np, cgp_lib.bitwise_or_np, cgp_lib.bitwise_xor_np]
rows = 1
columns = 1000
gt_table = cgp_lib.load_gt_from_file("8bit_adder.txt")
gt_table = convert_gt_table_to_np(gt_table, np.uint64, True)
chromosome = load_chromosome_from_file("chromosome.txt")
individual = cgp_lib.Individual(chromosome, 0.5)
np_wokspace = np.zeros((rows*columns + inputs, len(gt_table[0][0])), dtype=np.uint64)
individual_outputs = execute_individual(individual, np_wokspace)
#print(gt_table[0][:8].shape)
#print(gt_table[0][:8])
first_operand = numpy_64_to_decimal(gt_table[0][:8],8)
#print(first_operand.shape)
second_operand = numpy_64_to_decimal(gt_table[0][8:],8)
individual_outputs = numpy_64_to_decimal(individual_outputs,9)
gt_table_outputs = numpy_64_to_decimal(gt_table[1],9)
#print(first_operand)
#print(second_operand)
#print(gt_table[0].shape)
#print(gt_table_outputs)
#print(individual_outputs)

for i in range(1000):
    print(f"{first_operand[i]} + {second_operand[i]} = {individual_outputs[i]} / {gt_table_outputs[i]}")

#print correct amount of outputs
correct_cnt = np.count_nonzero(individual_outputs == gt_table_outputs)
print(f"Correct outputs: {correct_cnt}")

    
