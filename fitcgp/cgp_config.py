import numpy as np

class IndividualConfig:
    """
    Configuration of parameters of CGP individuals.

    :param inputs: Number of inputs of the CGP individual.
    :type inputs: int
    :param outputs: Number of outputs of the CGP individual.
    :type outputs: int
    :param columns: Number of columns of the CGP individual.
    :type columns: int
    :param rows: Number of rows of the CGP individual.
    :type rows: int
    :param levels_back: Number of levels back the CGP individual can access.
    :type levels_back: int
    :param node_functions: List of functions that can be used in CGP nodes.
    :type node_functions: List[Callable]
    :param arity: Arity of the CGP nodes.
    :type arity: int
    """
    def __init__(self, 
                 inputs: int, 
                 outputs: int,
                 columns: int,
                 rows: int,
                 levels_back: int, 
                 node_functions : list[str],
                 arity: int):
        self.inputs = inputs
        self.outputs = outputs
        self.columns = columns
        self.rows = rows
        self.levels_back = levels_back
        self.node_functions = node_functions
        self.arity = arity

class AlgorithmConfig:
    valid_modes = {"default", "numpy", "numpy_bitwise"}
    def __init__(self,
                 fitness_function,
                 fitness_maximization: bool, 
                 target_fitness: float,
                 population_size: int, 
                 generations: int,
                 mutation_rate: float = 0.05,
                 multiprocessing: bool = False,
                 mode: str = "default",
                 np_dtype = np.int64):
        self.fitness_function = fitness_function
        self.fitness_maximization = fitness_maximization
        self.target_fitness = target_fitness
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.multiprocessing = multiprocessing
        self.mode = mode
        if mode not in self.valid_modes:
            raise ValueError(f"Invalid mode {mode}. Valid modes are {self.valid_modes}")
        self.np_dtype = np_dtype

class AdvancedConfig:
    def __init__(self,
                 report_interval: int = 100,
                 checkpoint_interval: int = 1000,
                 checkpoint_path: str = "./checkpoint.json",
                 randomness_level: int = 1000):
        self.report_interval = report_interval #How many generations between printed reports
        self.checkpoint_interval = checkpoint_interval #How many generations between file checkpoints
        self.checkpoint_path = checkpoint_path #Name of the checkpoint file
        self.randomness_level = randomness_level




