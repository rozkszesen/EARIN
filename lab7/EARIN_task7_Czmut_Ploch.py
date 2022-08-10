'''
EARIN task 7: Bayesian Networks
Authors:
        Julia Czmut 300168
        Laura Ploch 300176
'''

import sys
import numpy as np
import ast
import json
import random

class BayesianNetwork():
    def __init__(self):
        self.nodes = dict()
        self.relations = dict()

    # Markov chain Monte Carlo algorithm with Gibbs sampling
    def MCMC(self, evidence, query, num_steps):
        '''
        evidence = dictionary {node_name: True/False}
        query = list of nodes for which the algorithm will provide an answer (updated belief)
        num_steps = number of steps of the algorithm
        '''
        
        # set values to observed variables
        # "variables" is a dictionary {variable_name: state} for observed, as well as unknown variables
        variables = evidence.copy()

        # set random values to unobserved variables
        unobserved_variables_names = [node for node in self.nodes if node not in evidence]
        for var_name in unobserved_variables_names:
            variables[var_name] = random.choice([True, False])

        # initialize counters for variables of interest to 0 (from the specified query)
        counters = dict()
        for var_name in query:
            counters[var_name] = {True: 0, False: 0}

        # random walking: draw an unobserved variable -> set it with a new value -> update counters
        for i in range(num_steps):
            # step 1: draw an unobserved variable
            selected_variable = random.choice(unobserved_variables_names)

            # step 2: set it with a new value -> P(X|MarkovBlanket(X))
            variables[selected_variable] = self.sample_from_Markov_blanket(variables, selected_variable)

            # step 3: update counters for all nodes of interest
            for var_name in query:
                counters[var_name][variables[var_name]] += 1

        # normalize the counters to get a probability distribution
        normalized_counters = dict()
        for var_name in counters.keys():
            counter_sum = sum(counters[var_name].values())
            normalized_counters[var_name] = dict()
            normalized_counters[var_name][True] = counters[var_name][True] / counter_sum
            normalized_counters[var_name][False] = counters[var_name][False] / counter_sum
            
        return normalized_counters

    def sample_from_Markov_blanket(self, variables, selected_variable):
        '''
        Markov blanket for selected node contains its parents, children and all parents of these children.
        Here, we're calculating the probabilities conditioned on the Markov blanket of selected variable
        and then drawing a sample using a simple roulette strategy.
        '''
        
        # initialize list for probabilities of every possible value of selected variable
        # P(X=xj|MarkovBlanket(X)) = alpha * P(X=xj|Parents(X)) * [ P(child=zi|Parents(child)) for all children of X ]
        probabilities = []
        
        # node without parents but with children
        if len(self.relations[selected_variable]["children"]) == 1 and len(self.relations[selected_variable]["parents"]) == 0:
            # no parents, so P(X=xj|Parents(X)) becomes P(X=xj)

            child = self.relations[selected_variable]["children"][0]
            # P(X) * P(child(X)=value|Parents(child(X)))
            for value in [False, True]:
                probability = self.nodes[selected_variable][value]
                probability *= self.nodes[child][value][variables[child]]
                probabilities.append(probability)
            
            # normalizing coefficient obtained from the fact that sum of P(X=x|Parents(X)) = 1
            alpha = 1 / sum(probabilities)

            # multiply all probabilities (beliefs) by alpha
            probabilities = [alpha * p for p in probabilities]

        # node without children but with a parent
        elif len(self.relations[selected_variable]["children"]) == 0 and len(self.relations[selected_variable]["parents"]) == 1:
            # no children, so P(child=zi|Parents(child)) is omitted from the equation
            parent = self.relations[selected_variable]["parents"][0]
            # P(X=xj|Parents(X)) * alpha(=1)
            probabilities.append(self.nodes[selected_variable][variables[parent]][False])
            probabilities.append(1 - probabilities[0])  # since probabilities sum up to 1

        # draw the new value for selected variable randomly
        return np.random.choice([False, True], p=probabilities)
  

    def read_JSON(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            nodes = data["nodes"]
            
            for node in nodes:
                relation = data["relations"][node]
                parents = relation["parents"]
                self.relations[node] = dict()
                prob = dict()

                if len(parents) == 0:
                    prob[True] = relation["probabilities"]["T"]
                    prob[False] = relation["probabilities"]["T"]

                elif len(parents) == 1:
                    prob[True] = dict()
                    prob[False] = dict()
                    prob[True][True] = relation["probabilities"]["T,T"]
                    prob[True][False] = relation["probabilities"]["T,F"]
                    prob[False][True] = relation["probabilities"]["F,T"]
                    prob[False][False] = relation["probabilities"]["F,F"]

                self.nodes[node] = prob.copy()

            # read parent-child relations 
            for node in nodes:
                relation = data["relations"][node]
                parents = relation["parents"]

                if len(parents) == 0:
                    self.relations[node]["parents"] = []
                
                elif len(parents) == 1:

                    self.relations[node]["parents"] = parents
                    par = parents[0]
                    self.relations[par]["children"] = []
                    self.relations[par]["children"].append(node)
                
                if "children" not in self.relations[node]:
                    self.relations[node]["children"] = []

    def read_evidence(self):
        ''' Returns a dictionary of evidence read from the user input '''
        while True:
            try:
                evidence = ast.literal_eval(input('Please provide evidence. Example: {"Flu": True}\n'))
                break
            except (ValueError, SyntaxError):
                print('Wrong formatting of evidence. Try again. Example: {"Flu": True}\n')

        return evidence        

    def read_query(self):
        ''' Returns a list of queries read from the user input '''
        while True:
            try:
                evidence = ast.literal_eval(input('Please provide query. Example: ["HighFever"]\n'))
                break
            except (ValueError, SyntaxError):
                print('Wrong formatting of query. Try again. Example: ["HighFever"]\n')

        return evidence  
        
    def read_numsteps(self):
        ''' Returns a dictionary of evidence read from the user input '''
        while True:
            try:
                evidence = int(input('Please provide number of steps as a positive integer.\n'))
                break
            except (ValueError, SyntaxError):
                print('Please provide number of steps as a positive integer.\n')

        return evidence

    def print_network(self):
        print("________Bayesian network structure________")
        for node in self.relations:
            print(str(node))
            print("\tparents:" + str(self.relations[node]["parents"]))
            print("\tchildren:" + str(self.relations[node]["children"]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Please provide the parameters needed")
        
    filename = sys.argv[1]
    bayesian_network = BayesianNetwork()
    bayesian_network.read_JSON(filename)
    
    # ask the user for evidence
    evidence = bayesian_network.read_evidence()
    
    # ask the user for the query
    query = bayesian_network.read_query()

    # ask the user for the number of steps for the MDMC algorithm
    num_steps = bayesian_network.read_numsteps()

    # print the network structure
    bayesian_network.print_network()
    
    # calculate the probability distribution of queried variables
    distribution = bayesian_network.MCMC(evidence, query, num_steps) # temporarily num_steps=5
    print("\nProbability distribution calculated using MCMC with Gibbs sampling in " + str(num_steps) + " steps")
    print("for variable of interest: " + str(query) + " knowing that " + str(evidence))
    print(distribution)
    