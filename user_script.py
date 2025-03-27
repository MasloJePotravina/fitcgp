import fitcgp
import numpy as np



node_functions = [fitcgp.logical_and, fitcgp.logical_or, fitcgp.logical_not, fitcgp.logical_xor]

X, y = fitcgp.load_gt_from_file("10bit_adder.txt")

individual_config = fitcgp.IndividualConfig(inputs = 20,
                                             outputs = 11,
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
                                      multiprocessing = True, 
                                      mode = "numpy_bitwise", 
                                      np_dtype=np.uint64)


# Create an algorithm object
algorithm = fitcgp.CGPAlgorithm(individual_config = individual_config, 
                                 algorithm_config = algo_config)


result = algorithm.fit(X, y)

print(result)