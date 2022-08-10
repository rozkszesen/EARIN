'''
Laura Ploch, 300176, 01143517@pw.edu.pl
Julia Czmut, 300168, 01143509@pw.edu.pl

Lab 2.
functions for reading and checking the input
'''
#import sympy as sp
import numpy as np
import ast

def read_input(filename):
    # opening the input file
    with open(filename) as input_file:
        parameters_list = input_file.readlines()

        for parameter in parameters_list:
            parameter = parameter.split("; ")

            # A; b; c; dim; d; population_size; cp; mp; max_iter

            A = np.array(ast.literal_eval(parameter[0]))
            b = np.array(ast.literal_eval(parameter[1]))
            c = int(ast.literal_eval(parameter[2]))
            dim = int(ast.literal_eval(parameter[3]))
            d = int(ast.literal_eval(parameter[4]))
            population_size = int(ast.literal_eval(parameter[5]))
            cp = float(ast.literal_eval(parameter[6]))
            mp = float(ast.literal_eval(parameter[7]))
            max_iter = int(ast.literal_eval(parameter[8]))

            function_parameters = [A, b, c]
            diversified_parameters = [dim, d, population_size, cp, mp, max_iter]

    return function_parameters, diversified_parameters


# function parameters A, b, c
# diversified parameters: dim, d, population_size, cp, mp, max_iter, 
def check_input(A, b, c, dim, d, population_size, cp, mp, max_iter):
    correct_input = True
    
    if A.shape[0] != b.shape[0]:
        print("ERROR: Dimensions of A and b do not match")
        correct_input = False

    if np.all(A.shape[0] != A.shape[1]):
        print("ERROR: A is not a square matrix")
        correct_input = False
        
    if b.shape[1] != 1:
        print("ERROR: b is not a vector")
        correct_input = False

    if np.isreal(c) == False:
        print("ERROR: c is not a real number")
        correct_input = False

    if dim != A.shape[0]:
        print("ERROR: dimensionality (dim) of the problem must correspond to dimensionality of vector b and matrix A")
        correct_input = False

    if isinstance(d, (np.int)) == False or d < 1 or d > 16:
        print("ERROR: d must be an integer between 1 and 16")
        correct_input = False
        
    if isinstance(population_size, (np.int)) == False or population_size < 1:
        print("ERROR: Population size must be an integer equal to or greater than 1")
        correct_input = False

    if np.isreal(cp) == False or cp < 0.0 or cp > 1.0:
        print("ERROR: crossover probability (cp) must be a real number between 0.0 and 1.0")
        correct_input = False

    if np.isreal(mp) == False or mp < 0.0 or mp > 1.0:
        print("ERROR: mutation probability (mp) must be a real number between 0.0 and 1.0")
        correct_input = False
    
    if isinstance(max_iter, (np.int)) == False or max_iter <= 0:
        print("ERROR: Number of maximum interations (max_iter) must be an integer greater than 0")
        correct_input = False
    
    return correct_input

def welcome():
    # instructions for the contents of the input file
    print("_________________Function maximization using Genetic Algorithms_________________\n")
    print("This program will find the maximum of your function.")
    print("Specifications of the chosen method and the parameters should be included in the input .txt file, which you should provide as a command line parameter.")
    print("The expected file format: one line with 9 fields separated by a semicolon followed by a space.")
    print("The expected contents of the input file are as follows:")
    print("\t[A; b; c; dim; d; population_size; cp; mp; max_iter]\n")
    print("Descriptions of individual fields:")
    print("\tA -> dim x dim sized matrix")
    print("\tb -> dim sized vector")
    print("\tc -> scalar number")
    print("\tdim -> problem dimensionality, the size of vector x")
    print("\td -> integer needed for specifying the range of searched integers such that -2^d <= xi < 2^d")
    print("\tpopulation_size -> integer, number of population members")
    print("\tcp -> crossover probability")
    print("\tmp -> mutation probability")
    print("\tmax_iter -> maximum number of algorithm iterations")