import random
import cProfile
import numpy as np
import time

from multiprocessing import Pool, shared_memory, current_process

from .cgp_individual import Individual
from .utils import is_function_gene, get_possible_input_nodes, get_function_gene, get_input_genes, convert_gt_to_np, convert_function_to_execution_mode
from .cgp_mutation import Mutation

#taken from: https://stackoverflow.com/questions/31704081/shared-arrays-in-multiprocessing-python
class SharedNumpyArray:
    def __init__(self, arr):
        self._shape = arr.shape
        self._dtype = arr.dtype
        self._shm = None
        self._creator = current_process().pid

        # Initialize shared memory
        self._acquired_shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self._name = self._acquired_shm.name
        _arr = np.ndarray(shape=self._shape, dtype=self._dtype, buffer=self._acquired_shm.buf)
        _arr[:] = arr[:]

    def __getstate__(self):
        # If pickle is being used to serialize this instance to another process,
        # then we do not need to include attribute _acquired_shm.

        if '_acquired_shm' in self.__dict__:
            state = self.__dict__.copy()
            del state['_acquired_shm']
        else:
            state = self.__dict__
        return state

    def arr(self):
        self._shm = shared_memory.SharedMemory(name=self._name)
        return np.ndarray(shape=self._shape, dtype=self._dtype, buffer=self._shm.buf)

    def __del__(self):
        if self._shm:
            self._shm.close()


    def close_and_unlink(self):
        """Called only by the process that created this instance."""

        if current_process().pid != self._creator:
            raise RuntimeError('Only the creating process may call close_and_unlink')

        if self._shm:
            self._shm.close()
            # Prevent __del__ from trying to close _shm again
            self._shm = None

        if self._acquired_shm:
            self._acquired_shm.close()
            self._acquired_shm.unlink()
            # Make additional call to this method a no-op:
            self._acquired_shm = None

class CGPAlgorithm:
    def __init__(self, individual_config, algorithm_config):
        self.individual_config = individual_config
        self.algorithm_config = algorithm_config
        self.population = []
        self.best_individual = None
        self.node_cnt = individual_config.columns * individual_config.rows #store number of nodes in each individual to avoid recomputation
        self.chromosome_length = self.node_cnt * (individual_config.arity + 1) + individual_config.outputs #store length of chromosome to avoid recomputation
        self.np_workspace = None #numpy workspace for numpy without multiprocessing
        self.shm_name = None #shared memory name for numpy with multiprocessing
        self.individual_config.node_functions = [convert_function_to_execution_mode(func, algorithm_config.mode) for func in individual_config.node_functions]
        self.algorithm_config.fitness_function = convert_function_to_execution_mode(algorithm_config.fitness_function, algorithm_config.mode)

    def generate_random_individual(self):
        """
        Create a random individual
        """
        chromosome = []
        for i in range(self.individual_config.columns * self.individual_config.rows):
            for j in range(self.individual_config.arity + 1):
                if is_function_gene(i * (self.individual_config.arity + 1) + j, self.individual_config.arity):
                    chromosome.append(random.randint(0, len(self.individual_config.node_functions) - 1))
                else:
                    possible_input_nodes = get_possible_input_nodes(i * (self.individual_config.arity + 1) + j, self.individual_config.rows, self.individual_config.levels_back, self.individual_config.arity, self.individual_config.inputs)
                    chromosome.append(random.choice(possible_input_nodes))
        for _ in range(self.individual_config.outputs):
            chromosome.append(random.randint(0, (self.node_cnt - 1 + self.individual_config.inputs)))
        return Individual(chromosome, 0)

    def init_population(self):
        for _ in range(self.algorithm_config.population_size):
            self.population.append(self.generate_random_individual())

    def get_active_nodes(self, individual):
        active_nodes = set()
        to_be_processed = list(individual.chromosome[-self.individual_config.outputs:])


        while to_be_processed:
            processed_node_id = to_be_processed.pop()
            if processed_node_id < self.individual_config.inputs or processed_node_id in active_nodes:
                continue
            active_nodes.add(processed_node_id)

            to_be_processed.extend(
            gene for i in range(self.individual_config.arity)
            if (gene := individual.chromosome[(processed_node_id - self.individual_config.inputs) * (self.individual_config.arity + 1) + i]) >= self.individual_config.inputs
        )

        return list(active_nodes)
    
    def execute_individual(self, individual, gt_inputs, gt_outputs, np_workspace):
        active_nodes = self.get_active_nodes(individual)
        
        
        individual_outputs = [None] * len(gt_outputs)

        function_genes = {}
        input_genes = {}

        for node_id in active_nodes:
            function_genes[node_id] = get_function_gene(individual, node_id, self.individual_config.arity, self.individual_config.inputs)
            input_genes[node_id] = get_input_genes(individual, node_id, self.individual_config.arity, self.individual_config.inputs)

        #print(function_genes)
        
        if self.algorithm_config.mode == "numpy" or self.algorithm_config.mode == "numpy_bitwise":
            #pre allocate a 2d array to store node outputs, second dimension is the same as the first dimension of the ground truth table, first dimension is the number of nodes and inputs 
            node_outputs = np_workspace
            #copy the input values to the first columns of the node_outputs array
            node_outputs[:self.individual_config.inputs, :] = gt_inputs
            
            for node_id in active_nodes:
                function_gene = function_genes[node_id]
                input_genes_list = input_genes[node_id]
                node_inputs = node_outputs[input_genes_list]
                node_outputs[node_id, :] = self.individual_config.node_functions[function_gene](node_inputs)
            output_genes = individual.chromosome[-self.individual_config.outputs:]
            individual_outputs = node_outputs[output_genes]
            return individual_outputs
        
        if self.algorithm_config.mode == "default":
        
            node_outputs = gt_inputs[0][:self.individual_config.inputs]

            # Pre-allocate the list size for node outputs to avoid appending
            node_outputs.extend([0] * self.node_cnt)


            for i in range(len(gt_inputs)):
                
                node_outputs[:self.individual_config.inputs] = gt_inputs[i][:self.individual_config.inputs]

                for node_id in range(self.individual_config.inputs, self.individual_config.inputs + self.node_cnt):
                    if node_id in active_nodes:
                        function_gene = function_genes[node_id]
                        input_genes_list = input_genes[node_id]

                        node_inputs = [node_outputs[input_gene] for input_gene in input_genes_list]
                        node_outputs[node_id] = self.individual_config.node_functions[function_gene](node_inputs)
                    else:
                        node_outputs[node_id] = 0

                output_genes = individual.chromosome[-self.individual_config.outputs:]
                individual_output = []
                for j in range(self.individual_config.outputs):
                    individual_output.append(node_outputs[output_genes[j]])
                individual_outputs[i] = individual_output

            return individual_outputs
        
    
    def evaluate_individual(self, individual, gt_inputs, gt_outputs, idx=None):
        if idx is not None:
            workspace = self.np_workspace.arr[idx]
            print(workspace)
            individual_outputs = self.execute_individual(individual, gt_inputs, gt_outputs, workspace)
        else:
            individual_outputs = self.execute_individual(individual, gt_inputs, gt_outputs, self.np_workspace)
        #NOTE: the line below has no effect when this function is used in multiprocessing, because it only affects the copy sent to the worker process
        individual.fitness = self.algorithm_config.fitness_function(individual_outputs, gt_outputs)
        return individual.fitness
    
    
    def fit(self, gt_inputs, gt_outputs):

        #Initialize population
        self.init_population()

        #Set comparison function for fitness based on whether we are maximizing or minimizing
        fitness_compare = max if self.algorithm_config.fitness_maximization else min

        #Create mutation object, that will handle mutation of individuals
        mutation = Mutation(self.individual_config)

        #If the user is working with numpy arrays (with or without bitwise parallelism), convert the ground truth to numpy arrays
        if self.algorithm_config.mode == "numpy" or self.algorithm_config.mode == "numpy_bitwise":
            gt_inputs, gt_outputs = convert_gt_to_np(gt_inputs, gt_outputs, self.algorithm_config.np_dtype, (True if self.algorithm_config.mode == "numpy_bitwise" else False), self.individual_config.inputs, self.individual_config.outputs)

            if self.algorithm_config.multiprocessing:
                np_workspace_tmp = np.zeros((self.algorithm_config.population_size - 1, self.node_cnt + self.individual_config.inputs, len(gt_inputs[0])), dtype=self.algorithm_config.np_dtype)
                self.np_workspace = SharedNumpyArray(np_workspace_tmp)
            else:
                self.np_workspace = np.zeros((self.node_cnt + self.individual_config.inputs, len(gt_inputs[0])), dtype=self.algorithm_config.np_dtype)

        self.best_individual = self.population[0]

        
        if(self.algorithm_config.multiprocessing) and ((self.algorithm_config.mode == "numpy") or (self.algorithm_config.mode == "numpy_bitwise")):
            self.best_individual.fitness = self.evaluate_individual(self.best_individual, gt_inputs, gt_outputs, 0)
        else:
            self.best_individual.fitness = self.evaluate_individual(self.best_individual, gt_inputs, gt_outputs)

        
        
        start_time = time.time()
        for i in range(self.algorithm_config.generations):
            #if i == 5:  # Start profiling only for Generation 5
            #    profiler = cProfile.Profile()  # Create a profiler instance
            #    profiler.enable()  # Start profiling
            
            fitness_list = [self.best_individual.fitness]
            fitness_list_tmp = []
            #-1 because parent left over from previous generation was already evaluated
            if(self.algorithm_config.multiprocessing):
                with Pool(self.algorithm_config.population_size-1) as p:
                    fitness_list_tmp = p.starmap(self.evaluate_individual, [(individual, gt_inputs, gt_outputs, idx) for idx, individual in enumerate(self.population[1:])])
                #This needs to be here, see note in evaluate_individual    
                for j, fitness in enumerate(fitness_list_tmp):
                    self.population[j+1].fitness = fitness
            else:
                fitness_list_tmp = [self.evaluate_individual(individual, gt_inputs, gt_outputs) for individual in self.population[1:]]
            
            fitness_list.extend(fitness_list_tmp)
            best_individual_index = fitness_list.index(fitness_compare(fitness_list))
            self.best_individual = self.population[best_individual_index]
            new_population = [self.best_individual]
            for _ in range(self.algorithm_config.population_size - 1):
                parent = self.best_individual
                new_individual = mutation.uniform_mutation(parent, self.individual_config)
                #new_individual = self.config.mutation_function(individual = parent, cgp_config = self.config)
                new_population.append(new_individual)
            self.population = new_population
            print(f"Generation {i}, best fitness: {self.best_individual.fitness}")

            #if i == 5:  # Stop profiling after Generation 5
            #    profiler.disable()  # Stop profiling
            #    profiler.print_stats()  # Print profiling results
            #    break
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        self.np_workspace.close_and_unlink()
        return self.best_individual