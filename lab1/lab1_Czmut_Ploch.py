'''
Laura Ploch, 300176, 01143517@pw.edu.pl
Julia Czmut, 300168, 01143509@pw.edu.pl

Lab 1. - finding minimum of a function using Gradient Descent and Newton's method
'''
import sympy as sp
import numpy as np
import sys
import time
import random
import ast

x = sp.Symbol('x')

def gradient_descent_F(x0, lrate, a, b, c ,d , desired_value, max_it, max_time):
    Fx = a * x**3 + b * x**2 + c*x + d
    i = 0
    result = x0
    yprime = Fx.diff()
    start_time = time.time()

    while i < max_it and time.time() - start_time <= max_time and result > desired_value:
     
        result = result - lrate*yprime.subs(x, result)
        i += 1

    return result

def newton_F(x0, eps, a, b, c ,d , desired_value, max_it, max_time):
    Fx = a * x**3 + b * x**2 + c*x + d
    start_time = time.time()
    i = 0
    result = x0
    yprime = Fx.diff()
    
    while i < max_it and time.time() - start_time <= max_time and result > desired_value and abs(Fx.subs(x,result)) > eps:
        result_old = result
        result = result - (yprime/Fx).subs(x,result)
        #print(result)

        if abs(result - result_old) < eps:
            print("increment size was smaller than epsilon")
            break
        
        i += 1

    return result

def hessian(A):
    return 2*A

def gradient(A, b, x):
    # derivative of x^T*A*x is equal to (A^T + A)*x = 2Ax (because A is symmetric so A^T = A)
    return b + 2*A.dot(x)

def calculate(expression, value):
    return expression.subs(x, value)

def evaluateMatrix(M, values):
    vcalculate = np.vectorize(calculate) #vectorized function to return a numpy array
    
    j = 0
    with np.nditer(M, flags=["refs_ok"], op_flags=['readwrite']) as it:
        for a in it:
            a[...] = vcalculate(a, values[0][j])
        j += 1
    
    return M

def gradient_descent_G(x0, lrate, A, b, c, desired_value, max_it, max_time):
   
    i = 0
    result = x0
    grad = gradient(A,b,x)

    start_time = time.time()

    while i < max_it and time.time() - start_time <= max_time and (result.transpose() > desired_value).all():
        result = result - lrate*evaluateMatrix(grad, result)
        i += 1
    
    return result.diagonal().transpose()


def newton_G(x0, eps, A, b, c, desired_value, max_it, max_time):
    
    i = 0
    result = x0
    
    M = np.linalg.inv(hessian(A)).dot(gradient(A,b,x))

    start_time = time.time()

    while i < max_it and time.time() - start_time <= max_time and (result.transpose() > desired_value).all():
        
        result = result - evaluateMatrix(M, result)
        i += 1
    #result.diagonal().transpose()
    return result.diagonal().transpose()

def check_input(A, b):
    correct_input = True
    if A.shape[0] != b.shape[0]:
        print("Dimensions of A and b do not match!")
        correct_input = False

    if not np.all(A-A.T <= 1e-12):
        print("A must be symmetric!")
        correct_input = False

    if  (np.any(np.linalg.eigvals(A) <= 0)):
        print("A is not positive definite!")
        correct_input = False # A is not positive definite
    return correct_input

def hello():
    # instructions for the contents of the input file
    print("_________________Function minimalization using Gradient Descent/Newton's method_________________\n")
    print("This program will find the minima of your function.")
    print("Specifications of the chosen method and the parameters should be included in the input .txt file, which you should provide as a command line parameter.")
    print("The expected file format: one line, 11 or 12 fields separated by a semicolon followed by a space ")
    print("The expected contents of the input file are as follows:")
    print("1) For function F(x) = ax^3 + bx^2 + cx + d")
    print("\t[which_function='f'; a; b; c; d; method; max_it; desired_value; max_time; lrate; batch_mode; x0/x0_min; 'n'/x0_max]\n")
    print("2) For function G(x) = c + b^T*x + x^T*A*x")
    print("\t[which_function='g'; c; b; A; method; max_it; desired_value; max_time; lrate; batch_mode; x0/x0_min; 'n'/x0_max]\n")
    print("Descriptions of individual fields:")
    print("\twhich_function -> 'f' for F(x), 'g' for G(x)")
    print("\ta -> scalar number")
    print("\tb -> scalar number for F(x), A-dimensional vector for G(x)")
    print("\tc -> scalar number")
    print("\td -> scalar number")
    print("\tA -> positive-definite matrix, for G(x)")
    print("\tmethod -> 'g' for Gradient Descent method, 'n' for Newton's method")
    print("\tmax_it -> maximum number of iterations")
    print("\tdesired_value -> desired value to reach")
    print("\tmax_time -> maximum computation time in seconds")
    print("\tlrate -> learning rate")
    print("\tbatch_mode -> positive integer indicating number of batch iterations; 1 - no batch mode, 2,3,... - batch mode enabled and value=number of iterations")
    print("\tx0/x0_min -> if you choose to provide the starting point as an exact value/vector, it's x0, and if as a range, it's the lower end of the range")
    print("\tn/x0_max -> if you choose to provide the starting point as an exact value/vector, it's 'n', and if as a range, it's the upper end of the range")
    print("\tPlease note that it is not possible to provide x0 as an exact value/vector if batch mode is enabled (batch_mode>1)")

    return

def meanArr(results_list, result_size):
    mean_values = []

    for i in result_size:
        mean_values[i] = 0

    for result in results_list:
        for i in result_size:
            mean_values[i] += result[i]

    for i in result_size:
        mean_values[i] /= len(results_list)
    
    return mean_values

def main(args):
    '''
    #test
    desired = -10
    x0 = -0.2
    lrate = 0.05
    f1 = [2, -2, -3, 4]
   
    #gradient_descent_F(x0, lrate, f1, desired, 20)
    '''
   
    '''
    #test
    desired = np.array([[-10, -10]])
    x0 = np.array([[1,2]])
    c = 2
    b = np.array([[1,1]])
    A = np.array([[8,4],[4,4]])
    X = np.array([[x, x]])
    
    gradient_descent_G(x0, lrate, A, b, c, desired, 2, 2)
    '''

     # if no input file specified, print sth and exit
    if len(sys.argv) <= 1:    
        print("No input file.")
        hello()
        return

    # opening the input file
    with open(args[0]) as input_file:
        parameters_list = input_file.readlines()
        for parameter in parameters_list:
            parameter = parameter.split("; ")

            # literal_eval safely evaluates an expression node or a string containing a Python literal structure (strings, bytes, numbers, tuples, lists, etc.)
            which_function = parameter[0]   # "f" or "g" to specify which function we want optimized

            if which_function == "f":
                a = float(ast.literal_eval(parameter[1]))   # scalar number a
                b = float(ast.literal_eval(parameter[2]))   # scalar number b
                c = float(ast.literal_eval(parameter[3]))   # scalar number c
                d = float(ast.literal_eval(parameter[4]))   # scalar number d
                method = parameter[5]   # "g" if gradient descent, "n" if Newton's
                max_it = int(ast.literal_eval(parameter[6]))    # maximum number of iterations
                desired_value = float(ast.literal_eval(parameter[7]))
                max_time = float(ast.literal_eval(parameter[8]))    # maximum computation time
                lrate = float(ast.literal_eval(parameter[9]))   # learning rate
                batch_mode = int(ast.literal_eval(parameter[10]))   # 1 - no batch mode, 1+ - batch mode, number of iterations

                if batch_mode < 1:
                    print("Batch must be positive.")
                    return

                # if batch mode is disabled
                elif batch_mode == 1:     
                    if parameter[12] == "n":    # so if the starting point is given as an exact value
                        x0 = float(ast.literal_eval(parameter[11]))   # starting point (in case of F it's a scalar)

                        # gradient descent method for F(x) in normal mode
                        if method == "g":
                            result = gradient_descent_F(x0, lrate, a, b, c, d, desired_value, max_it, max_time)
                            print(result)

                        # Newton's method for F(x) in normal mode
                        elif method == "n":
                            eps = lrate
                            result = newton_F(x0, eps, a, b, c, d, desired_value, max_it, max_time)
                            print(result)

                    # so if a range is given for the random generation of the starting point
                    else:    
                        x0_min = float(ast.literal_eval(parameter[11]))  # lower end of the range
                        x0_max = float(ast.literal_eval(parameter[12]))  # higher end of the range

                        # generate random x0 from the range provided by the user
                        x0 = random.uniform(x0_min, x0_max, b.size())

                        # gradient descent method for F(x) in normal mode
                        if method == "g":
                            result = gradient_descent_F(x0, lrate, a, b, c, d, desired_value, max_it, max_time)
                            print(result)
                            
                        # Newton's method for F(x) in normal mode
                        elif method == "n":
                            eps = lrate
                            result = newton_F(x0, eps, a, b, c, d, desired_value, max_it, max_time)
                            print(result)

                # if batch mode is enabled and the value is the number of iterations of the algorithm
                elif batch_mode > 1:    
                    print("batch mode")

                    if parameter[12] == "n":
                        print("If batch mode is selected, a range for generating x0 must be provided.")
                        return

                    x0_min = float(ast.literal_eval(parameter[11]))  # lower end of the range
                    x0_max = float(ast.literal_eval(parameter[12]))  # higher end of the range

                    results_list = []    # list consisting of results of all batch iterations

                    # gradient descent method for F(x) in batch mode
                    if method == "g":
                        for _ in range(batch_mode):
                            x0 = random.uniform(x0_min, x0_max)
                            results_list.append(gradient_descent_F(x0, lrate, a, b, c, d, desired_value, max_it, max_time))
                    
                    # Newton's method for F(x) in batch mode
                    if method == "n":
                        eps = lrate
                        for _ in range(batch_mode):
                            x0 = random.uniform(x0_min, x0_max)
                            results_list.append(newton_F(x0, eps, a, b, c, d, desired_value, max_it, max_time))
                    
                    print(results_list)
                    mean_value = np.mean(results_list, axis=0)
                    #standard_deviation = np.std(results_list, axis=0)

                    print("mean value = " + str(mean_value))
                    #print(standard_deviation)

                
            elif which_function == "g":

                c = float(ast.literal_eval(parameter[1]))   # scalar number c
                b = np.array(ast.literal_eval(parameter[2]))    # A-dimensional vector
                A = np.array(ast.literal_eval(parameter[3]))    # positive-definite matrix
                if check_input(A,b) == False:
                    exit()
                method = parameter[4]   # "g" if gradient descent, "n" if Newton's
                max_it = int(ast.literal_eval(parameter[5]))    # maximum number of iterations
                desired_value = np.array(ast.literal_eval(parameter[6]))
                max_time = float(ast.literal_eval(parameter[7]))    # maximum computation time
                lrate = float(ast.literal_eval(parameter[8]))   # learning rate
                batch_mode = int(ast.literal_eval(parameter[9]))   # 1 - no batch mode, 1+ - batch mode, number of iterations
                
                if batch_mode < 1:
                    print("Batch must be positive.")
                    return

                # if batch mode is disabled
                if batch_mode == 1:   
                    if parameter[11] == "n":    # so if the starting point is given as an exact value
                        x0 = np.array(ast.literal_eval(parameter[10]))

                        # gradient descent method for G(x) in normal mode
                        if method == "g":
                            result = gradient_descent_G(x0, lrate, A, b, c, desired_value, max_it, max_time)
                            print(result)

                        # Newton's method for G(x) in normal mode
                        elif method == "n":
                            eps = lrate
                            print("method='n'")
                            result = newton_G(x0, eps, A, b, c, desired_value, max_it, max_time)
                            print(result)
                    
                    else:    # so if a range is given for the random generation of the starting point
                        x0_min = float(ast.literal_eval(parameter[10]))  # lower end of the range
                        x0_max = float(ast.literal_eval(parameter[11]))  # higher end of the range
                        
                        # generate random x0 from the range provided by the user
                        x0 = random.uniform(x0_min, x0_max)
                        print(x0)

                        # gradient descent method for G(x) in normal mode
                        if method == "g":
                            result = gradient_descent_G(x0, lrate, A, b, c, desired_value, max_it, max_time)
                            print(result)

                        # Newton's method for G(x) in normal mode
                        elif method == "n":
                            eps = lrate
                            print("method='n'")
                            result = newton_G(x0, eps, A, b, c, desired_value, max_it, max_time)
                            print(result)
                            

                # if batch mode is enabled and the value is the number of iterations of the algorithm
                elif batch_mode > 1:    
                    print("batch mode")

                    if parameter[11] == "n":
                        print("If batch mode is selected, a range for generating x0 must be provided.")
                        return

                    x0_min = np.array(ast.literal_eval(parameter[10]))  # lower end of the range
                    x0_max = np.array(ast.literal_eval(parameter[11]))  # higher end of the range

                    results_list = []    # list consisting of results of all batch iterations

                    # gradient descent method for G(x) in batch mode
                    if method == "g":
                        for _ in range(batch_mode):
                            x0 = random.uniform(x0_min, x0_max)
                            results_list.append(gradient_descent_G(x0, lrate, A, b, c, desired_value, max_it, max_time))

                    # Newton's method for G(x) in batch mode
                    if method == "n":
                        for _ in range(batch_mode):
                            eps = lrate
                            x0 = random.uniform(x0_min, x0_max, b.size())
                            result = newton_G(x0, eps, a, b, c, d, desired_value, max_it, max_time)
                            results_list.append(result)
                        
                    
                    print(results_list)
                    mean_value = meanArr(results_list, A.shape[0])
                    #standard_deviation = np.std(results_list, axis=0)

                    print("mean value = " + str(mean_value))
                    #print(standard_deviation)
    

if __name__ == '__main__':
    main(sys.argv[1:])