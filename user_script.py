import fitcgp
import numpy as np

def myown_xor(inputs):
    return np.bitwise_xor(inputs[0], inputs[1])

#node_functions = [fitcgp.logical_and, fitcgp.logical_or, fitcgp.logical_not, myown_xor]

X, y = fitcgp.load_gt_from_file("8bit_adder.txt")

"""individual_config = fitcgp.IndividualConfig(inputs = 16,
                                             outputs = 9,
                                             columns = 1000,
                                             rows = 1,
                                             levels_back = 1000,
                                             node_functions = node_functions,
                                             arity = 2)

algo_config = fitcgp.AlgorithmConfig(fitness_function = fitcgp.error_rate,
                                      target_fitness = 0,
                                      fitness_maximization = False,
                                      population_size = 5,
                                      generations = 10000, 
                                      multiprocessing = False, 
                                      mode = "numpy_bitwise", 
                                      np_dtype=np.uint64)

advanced_config = fitcgp.AdvancedConfig(report_interval = 100,
                                          checkpoint_interval = 1000,
                                          checkpoint_path = "./checkpoint.json")

#NOTE: nto required, this is just for ease of testing
best_individual = None"""

individual_config, algo_config, advanced_config, best_individual = fitcgp.load_json_checkpoint("./checkpoint.json")

individual_config.node_functions[3] = myown_xor

# Create an algorithm object
algorithm = fitcgp.CGPAlgorithm(individual_config = individual_config, 
                                 algorithm_config = algo_config,
                                 advanced_config = advanced_config,
                                 best_individual = best_individual)


result = algorithm.fit(X, y)

print(result)