import cgp_lib
import numpy as np



node_functions = [cgp_lib.logical_and, cgp_lib.logical_or, cgp_lib.logical_not, cgp_lib.logical_xor]

X, y = cgp_lib.load_gt_from_file("10bit_adder.txt")

individual_config = cgp_lib.IndividualConfig(inputs = 20,
                                             outputs = 11,
                                             columns = 1000,
                                             rows = 1,
                                             levels_back = 1000,
                                             node_functions = node_functions,
                                             arity = 2)

algo_config = cgp_lib.AlgorithmConfig(fitness_function = cgp_lib.error_rate,
                                      target_fitness = 0,
                                      fitness_maximization = False,
                                      population_size = 5,
                                      generations = 10000, 
                                      multiprocessing = True, 
                                      mode = "numpy_bitwise", 
                                      np_dtype=np.uint64)


# Create an algorithm object
algorithm = cgp_lib.CGPAlgorithm(individual_config = individual_config, 
                                 algorithm_config = algo_config)


result = algorithm.fit(X, y)

print(result)