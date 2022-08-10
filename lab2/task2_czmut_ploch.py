'''
Laura Ploch, 300176, 01143517@pw.edu.pl
Julia Czmut, 300168, 01143509@pw.edu.pl

Lab 2. - finding maximum of a function using genetic algorithm
'''

import numpy as np
import sys
from input import *
import random


class Function:
    "Multidimensional quadratic function"
    def __init__(self, A, b, c) -> None:
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = float(c)

    def compute(self, x):
        "Return the value of the function for a given x"
        return int(x.transpose().dot(self.A).dot(x) + self.b.transpose().dot(x) + self.c)

class Genotype:
    "Class holding genotype (array of values) of a particular individual"
    def __init__(self, values, dim) -> None:
        self.genotype = []
        for i in range(dim):
            self.genotype.append(values[i])
    
    def getArray(self):
        return self.genotype

    def getSize(self):
        return len(self.genotype)

def maximizeF(F, population_size, cp, mp, max_iter, dim, d):
    "Calculate maximal value of a function using genetic algorithm"

    # initialize the population
    population = []
    for i in range(population_size):
       population.append(np.random.randint(-2**d, 2**d, dim))

    for it in range(max_iter):
        # find fitness of population
        fitness_values = calculate_fitnesses(F, population)
        
        # list of probabilities of choosing different individuals from the whole population, its sum should be 1
        probabilities = [roulette_with_scaling(F, population, p, fitness_values) for p in population]
        
        # construct list of population (2-d array) as list of Genotypes so that it can be used in numpy.random.choice(a, ...) where a must be 1-d array
        genotypes = []
        for i in range(population_size):
            genotype = Genotype(population[i], dim)
            genotypes.append(genotype)

        # select parents from population basing on their probabilities
        parents_genotypes = np.random.choice(genotypes, 2, replace = True, p = probabilities)
        
        # convert parents' genotypes to vectors of binary strings
        parents_bin = [to_binary(p, d) for p in parents_genotypes]

        # produce 2 children 
        children_bin = crossover_mutation(parents_bin, dim, cp, mp)

        # convert values in children's vectors from binary to decimal
        children = [np.array(to_decimal(binary_child, dim, d)) for binary_child in children_bin]
        
        # removing values bearing worst result from the population and adding new children
        population = FIFO(population, children, fitness_values)

        # calculate values of function F for new population and pick the max
        fitness_values = calculate_fitnesses(F, population)
        max_value = max(fitness_values)
    
    # return max fitness, x and new_population
    return population, fitness_values, max_value

def calculate_fitnesses(F, population):
    "Return array of fitness values of the given population"
    fitness_values = []
    
    for i in range(len(population)):
        value = F.compute(population[i])
        fitness_values.append(value)

    return fitness_values

def roulette_with_scaling(F, population, individual, fitness_values_list):
    "Assigning probabilities to each individual"
    "The probability of selection of an individual is proportional to its target value (fitness value?)"

    fitness_values = np.array(fitness_values_list)
    min_fitness = np.min(fitness_values)
    max_fitness = np.max(fitness_values)
    divider = max_fitness - min_fitness

    if divider == 0.0:   # if all fitness values were the same
        return (1 / len(population) for p in population) # as a loop, because we need to return an array
    else:
        pop_sum = []
        for p in population:
            pop_sum.append((F.compute(p) - min_fitness) / divider)
        q_sum = np.sum(pop_sum)

        # probability of selection is proportional to its target value, so we return probability = (scaled target value) / (sum of scaled target values across the entire population)  
        probability = ((F.compute(individual) - min_fitness) / divider) / q_sum
        return probability


def to_binary(genotype, d):
    "Convert each element of genotype vector to binary representation (vectors of 0s and 1s, of length d + 1)"
    nr_of_bits = d + 1
    binary_vector = []
    for i in range(genotype.getSize()):
        decimal_number = genotype.getArray()[i]
        binary_number = bin(decimal_number & int("1"*nr_of_bits, 2))[2:]    # convert to binary and remove 0b from the beginning (hence [2:])
        formated = ("{0:0>%s}" % (nr_of_bits)).format(binary_number)        # format so that it hase proper number of bits
        binary_vector.append(formated)
        
    return binary_vector

def crossover_mutation(binary_parents, dim, cp, mp):
    "Perform crossover and mutation using the given parents"
    children = binary_parents
    # single-point crossover
    if np.random.rand() < cp:
        children = []
        child1 = []
        child2 = []
        cross_point = random.randint(1, len(binary_parents[0][0]) - 1)
        
        for i in range(dim):    # crossing all parents' elements, so there are 2*dim crossovers
            child1.append(binary_parents[0][i][:cross_point] + binary_parents[1][i][cross_point:])
            child2.append(binary_parents[1][i][:cross_point] + binary_parents[0][i][cross_point:])
        children.append(child1)
        children.append(child2)
    
    # mutation - choose a point and switch its bit value
    # mutating the children 
    for i in range(len(children)):
        if np.random.rand() < mp:
            mutation_point = random.randint(1, len(binary_parents[0][0]) - 1)
            for j in range(dim):
                if children[i][j][mutation_point] == "1":
                    children[i][j] = children[i][j][:mutation_point] + "0" + children[i][j][mutation_point+1:]
                else:
                    children[i][j] = children[i][j][:mutation_point] + "1" + children[i][j][mutation_point+1:]

    return children

def FIFO(population, children, fitness_values):

    for i in range(len(children)):
        min_value = min(fitness_values)
        min_index = fitness_values.index(min_value)
        population.append(children[i])
        #population.remove(population[min_index])   # removing the one bearing the worst result, we found this to be more efficient 
        del population[min_index]
  
    return population

def to_decimal(binary_child, dim, d):
    "Convert binary strings of length d + 1 back to decimal representation"
    bits = d+1
    child = []
    for i in range(dim):
        binary_number = binary_child[i]
        if binary_number[0] == "0":
            # if the msb is 0 - number is positive -> we can perform regular conversion
            decimal = int(binary_number, 2)
        else:
            # if the msb is 1 - number is negative
            decimal = int(binary_number, 2) - (1 << bits)
        child.append(decimal)
        
    return child

def main(args):

    # if no input file specified
    if len(sys.argv) <= 1:    
        print("No input file.")
        welcome()
        return 0

    # reading parameters from the input file
    [function_parameters, diversified_parameters] = read_input(args[0])

    # no errors, so saving the parameters and proceeding with the algorithm
    A = function_parameters[0]
    b = function_parameters[1]
    c = function_parameters[2]
    F = Function(A, b, c)

    dim = diversified_parameters[0]     # dimensionality of x vectors
    d = diversified_parameters[1] 
    population_size = diversified_parameters[2] 
    cp = diversified_parameters[3] 
    mp = diversified_parameters[4] 
    max_iter = diversified_parameters[5] 

    # if there are errors in the input file
    if check_input(A, b, c, dim, d, population_size, cp, mp, max_iter) == False:
        print("Please read the instructions below.")
        welcome()
        return 0
   
    last_population, Fx_values, max = maximizeF(F, population_size, cp, mp, max_iter, dim, d)
    
    for i in range(population_size):
        print("x[" + str(i+1) + "] = " + str(last_population[i]) + " F(x) =" + str(Fx_values[i]))
    print("Found max value =" + str(max))


if __name__ == '__main__':
    main(sys.argv[1:])
