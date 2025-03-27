class Individual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

    def __str__(self):
        return f"chromosome: {self.chromosome}, Fitness: {self.fitness}"