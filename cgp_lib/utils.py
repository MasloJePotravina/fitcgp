import numpy as np
from .node_functions import *
from .fitness_functions import *

def is_function_gene(gene_num, arity):
    """
    Check if the gene is a function gene
    """
    return (gene_num + 1) % (arity + 1) == 0

def get_node_id(gene_num, arity, inputs):
    """
    Get the node id of the gene
    """
    return (gene_num // (arity + 1)) + inputs

def get_node_column(node_id, rows, inputs):
    """
    Get the column of the node
    """
    return (node_id - inputs) // rows

def get_possible_input_nodes(gene_num, rows, levels_back, arity, inputs):
    """
    Get the possible input genes for a given gene
    """
    node_id = get_node_id(gene_num, arity, inputs)
    node_column = get_node_column(node_id, rows, inputs)
    smallest_column = max(0, node_column - levels_back)

    lower_bound = inputs + smallest_column * rows
    upper_bound = inputs + node_column * rows


    return list(range(inputs)) + list(range(lower_bound, upper_bound))

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

def convert_gt_to_np(gt_inputs, gt_outputs, data_type, bitwise_parallelization, inputs, outputs):
        #inputs, outputs = map(lambda x: np.array(x, dtype=data_type), gt_table)
        inputs_arr = np.array(gt_inputs, dtype=data_type).T
        outputs_arr = np.array(gt_outputs, dtype=data_type).T
        if bitwise_parallelization:
            inputs_arr = np.packbits(inputs_arr.reshape(inputs, -1).copy(), axis=-1).view(np.uint64)
            outputs_arr = np.packbits(outputs_arr.reshape(outputs, -1).copy(), axis=-1).view(np.uint64)
        return inputs_arr, outputs_arr

def load_gt_from_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    gt_table = [[],[]]
    for line in lines:
        gt_table[0].append([int(a) for a in line.split(":")[0]])
        gt_table[1].append([int(a) for a in line.split(":")[1].strip()])
    return gt_table[0], gt_table[1]

#Switch default functions for numpy or numpy bitwise functions based on execution mode
def convert_function_to_execution_mode(function, execution_mode):
    function_name = function.__name__
    if execution_mode == "numpy":
        function_name += "_np"
        if function_name in globals():
            return globals()[function_name]
    elif execution_mode == "numpy_bitwise":
        function_name += "_bitwise_np"
        if function_name in globals():
            return globals()[function_name]
    
    return function