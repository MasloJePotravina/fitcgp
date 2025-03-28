import numpy as np
from .node_functions import *
from .fitness_functions import *
import json
from .cgp_config import IndividualConfig, AlgorithmConfig, AdvancedConfig
from .cgp_individual import Individual
import re


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
    if function is None:
        print(
        "NOTE: If you are loading a checkpoint with custom node functions, you have to define them and add them to the node_functions list in IndividualConfig after loading the checkpoint. Make sure to insert them onto their correct indexes.\n"
        "NOTE: If you are loading a checkpoint with a custom fitness function, you have to define it and add it to the fitness_function in IndividualConfig after loading the checkpoint.")
        raise ValueError("Node function cannot be None")
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

def get_function_from_name(function_name):
    return globals().get(function_name, None)

def print_report(generation, best_fitness):
    """
    Print the report of the current generation
    """
    print(f"Generation: {generation}, Best fitness: {best_fitness}")



def prepare_function_name_for_saving(function_name):
    return function_name.removesuffix("_bitwise_np").removesuffix("_np")

def load_json_checkpoint(checkpoint_path):
    """
    Load a checkpoint from a json file
    """
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)

    individual_config = IndividualConfig(
        inputs=checkpoint_data["individual_config"]["inputs"],
        outputs=checkpoint_data["individual_config"]["outputs"],
        columns=checkpoint_data["individual_config"]["columns"],
        rows=checkpoint_data["individual_config"]["rows"],
        levels_back=checkpoint_data["individual_config"]["levels_back"],
        node_functions=[get_function_from_name(func) for func in checkpoint_data["individual_config"]["node_functions"]],
        arity=checkpoint_data["individual_config"]["arity"]
    )

    algorithm_config = AlgorithmConfig(
        fitness_function=get_function_from_name(checkpoint_data["algorithm_config"]["fitness_function"]),
        fitness_maximization=checkpoint_data["algorithm_config"]["fitness_maximization"],
        target_fitness=checkpoint_data["algorithm_config"]["target_fitness"],
        population_size=checkpoint_data["algorithm_config"]["population_size"],
        generations=checkpoint_data["algorithm_config"]["generations"],
        mutation_rate=checkpoint_data["algorithm_config"]["mutation_rate"],
        multiprocessing=checkpoint_data["algorithm_config"]["multiprocessing"],
        mode=checkpoint_data["algorithm_config"]["mode"]
    )

    advanced_config = AdvancedConfig(
        report_interval=checkpoint_data["advanced_config"]["report_interval"],
        checkpoint_interval=checkpoint_data["advanced_config"]["checkpoint_interval"],
        checkpoint_path=checkpoint_path
    )



    best_individual = Individual(
        chromosome=checkpoint_data["best_individual"],
        fitness=checkpoint_data["best_fitness"]
    )
    return individual_config, algorithm_config, advanced_config, best_individual


def save_json_checkpoint(individual_config, algorithm_config, advanced_config, best_individual):
    """
    save a checkpoint of the current state of the algorithm
    """
    checkpoint_data = {
        "individual_config": {
            "inputs": individual_config.inputs,
            "outputs": individual_config.outputs,
            "columns": individual_config.columns,
            "rows": individual_config.rows,
            "levels_back": individual_config.levels_back,
            "node_functions": [prepare_function_name_for_saving(func.__name__) for func in individual_config.node_functions],
            "arity": individual_config.arity
        },
        "algorithm_config": {
            "fitness_function": prepare_function_name_for_saving(algorithm_config.fitness_function.__name__),
            "fitness_maximization": algorithm_config.fitness_maximization,
            "target_fitness": algorithm_config.target_fitness,
            "population_size": algorithm_config.population_size,
            "generations": algorithm_config.generations,
            "mutation_rate": algorithm_config.mutation_rate,
            "multiprocessing": algorithm_config.multiprocessing,
            "mode": algorithm_config.mode
        },
        "advanced_config": {
            "report_interval": advanced_config.report_interval,
            "checkpoint_interval": advanced_config.checkpoint_interval,
            "checkpoint_path": advanced_config.checkpoint_path
        },
        "best_fitness": best_individual.fitness,
        "best_individual": best_individual.chromosome
    }

    with open(advanced_config.checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    
    with open(advanced_config.checkpoint_path, 'r+') as f:
        content = f.read()

        # Use regex to find the "best_individual" list and remove the newlines
        content = re.sub(r'("best_individual": \[)([\s\S]*?)(\])', 
                 lambda m: m.group(1) + ''.join(m.group(2).splitlines()).replace(" ", "") + m.group(3), 
                 content)
        f.seek(0)
        f.write(content)
        f.truncate()

def save_checkpoint(individual_config, algorithm_config, advanced_config, best_individual):
    #if path ends with .json
    if advanced_config.checkpoint_path.endswith(".json"):
        save_json_checkpoint(individual_config, algorithm_config, advanced_config, best_individual)


