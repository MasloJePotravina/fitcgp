import copy
import random
from .cgp_individual import Individual

from .utils import is_function_gene, get_possible_input_nodes

import numpy as np

class Mutation:
    def __init__(self, individual_config, mutation_rate, randomness_level):
        print("Pre-generating randomness...")
        #self.mutation_function = config.mutation_function
        self.mutation_rate = mutation_rate

        self.chromosome_length = individual_config.columns * individual_config.rows * (individual_config.arity + 1) + individual_config.outputs

        #arbitrary large number to avoid recomputation, ideally should be longer than the chromosome
        self.randomness_level = randomness_level

        #Generates a list of mutation masks, individual selects one of these masks to mutate their genes
        #self.mutation_mask_list = [[] for _ in range(self.randomness_level)]
        #for i in range(self.randomness_level):
        #    self.mutation_mask_list[i] = [random.random() < self.mutation_rate for _ in range(self.chromosome_length)]

        self.mutation_mask_list = np.random.rand(self.randomness_level, self.chromosome_length) < self.mutation_rate


        #Generates a list of random node functions, individual is selecting one of these functions to mutate their function gene (if it was selected by the mask)
        self.node_function_random_list = [random.randint(0, len(individual_config.node_functions) - 1) for _ in range(self.randomness_level)]

        #Stores a list of possible input nodes for each node in the chromosome (output genes excluded, as they can connect to any node)
        possible_input_nodes_table = [get_possible_input_nodes(i, individual_config.rows, individual_config.levels_back, individual_config.arity, individual_config.inputs) for i in range(self.chromosome_length - individual_config.outputs)]

        #Generates a list of random input nodes, individual is selecting one of these nodes to mutate their input gene (if it was selected by the mask)
        self.node_inputs_random_lists = [[] for _ in range(self.chromosome_length - individual_config.outputs)]
        for i in range(self.chromosome_length - individual_config.outputs):
            self.node_inputs_random_lists[i] = [random.choice(possible_input_nodes_table[i]) for _ in range(self.randomness_level)]
        
        #Generates a list of random output genes, individual is selecting one of these genes to mutate their output gene (if it was selected by the mask)
        self.output_genes_random_list = [random.randint(0, (individual_config.columns * individual_config.rows - 1 + individual_config.inputs)) for _ in range(self.randomness_level)]

        #Mask of output genes to avoid calling is_function_gene for each gene
        self.function_gene_mask = [is_function_gene(i, individual_config.arity) for i in range(self.chromosome_length - individual_config.outputs)]

        print("Randomness pre-generated")
    
    #Method to increase the index of the random lists, wraps around if the index is at the end of the list
    def increase_index(self, index):
        return (index + 1) % self.randomness_level

    def uniform_mutation(self, individual, individual_config, mut_chance = 0.1):
        """
        Mutate an individual by changing the function or input of a node
        """
        mutated_individual = Individual(
            chromosome=individual.chromosome[:],
            fitness=None
        )

        #Generate random index for mutation mask and random start indexes for node function, node inputs and output genes
        mutation_mask_lists_idx = random.randint(0, self.randomness_level - 1)
        node_function_random_list_idx = random.randint(0, self.randomness_level - 1)
        node_inputs_random_lists_idx = random.randint(0, self.randomness_level - 1)
        output_genes_random_list_idx = random.randint(0, self.randomness_level - 1)

        mutation_mask = self.mutation_mask_list[mutation_mask_lists_idx]

        for i in range(self.chromosome_length - individual_config.outputs):
            if mutation_mask[i]:
                #Function gene branch
                if self.function_gene_mask[i]:
                    mutated_individual.chromosome[i] = self.node_function_random_list[node_function_random_list_idx]
                    node_function_random_list_idx = self.increase_index(node_function_random_list_idx)
                #Input gene branch
                else:
                    mutated_individual.chromosome[i] = self.node_inputs_random_lists[i][node_inputs_random_lists_idx]
                    node_inputs_random_lists_idx = self.increase_index(node_inputs_random_lists_idx)

        #Mutate outputs
        for i in range(self.chromosome_length - individual_config.outputs, self.chromosome_length):
            if mutation_mask[i]:
                mutated_individual.chromosome[i] = self.output_genes_random_list[output_genes_random_list_idx]
                output_genes_random_list_idx = self.increase_index(output_genes_random_list_idx)

        return mutated_individual
    