'''
Currently under maintanance
'''


from docplex.mp.model import Model
import pandas as pd
import numpy as np
from route_optimizer import SingleSourceRouteOptimizer

class MultiSourceRouteOptimizer(SingleSourceRouteOptimizer):
    '''
    Route Optimizer class to solve single source routing problem
    '''
    
    def __init__(self, spots_df, num_days, cost_per_mile):

        super().__init__(spots_df, num_days, cost_per_mile)

    def SolveUsingDocplex(self):
        '''
        Function to solve routing problem

        Output:
            model: Docplex Model: Model result
            sol: Docplex Model: Solution
            X: Docplex Variable: Decision Variable
        '''
        print("\nOptimizing route...")

        # Initialize data structures
        numSpots = self.spots_df.shape[0]
        S = [i for i in range(numSpots)]
        N = [i for i in S if i != 0]
        A = [(i, j) for i in S for j in S if i != j]
        F = list(self.spots_df[self.spots_df["is_restaurant"] == 1].index)
        R = [i for i in N if i not in F]
        k = 0  # index for current lodging location
        dist = self.__generate_dist_dict(self.spots_df)
        time_spent = self.spots_df.loc[1:,'time_spent'].to_dict()

        # Create docplex model
        mdl = Model("Routing")
        
        # Add decision variable
        X = mdl.binary_var_dict(A, lb=0, ub=1, name="X")
        CUM_TIME = mdl.continuous_var_dict(N, ub=self.MAX_TIME_SPENT_PER_DAY, name='cum_time_spent')
        
        # Add objective function
        mdl.minimize(mdl.sum(X[i, j] * dist[i, j] for i, j in A))
        
        # Add total outflow constraint
        mdl.add_constraints(mdl.sum(X[i, j] for j in S if j != i) == 1 for i in N)
        
        # Add total inflow constraint
        mdl.add_constraints(mdl.sum(X[i, j] for i in S if i != j) == 1 for j in N)
        
        # Add cumulative sum constraint
        mdl.add_indicator_constraints(
            mdl.indicator_constraint(
                X[i, j], CUM_TIME[i] + time_spent[j] == CUM_TIME[j]
            ) for i, j in A if i != k and j != k
        )
        
        # Add lower bound constraint
        mdl.add_constraints(CUM_TIME[i] >= time_spent[i] for i in N)
        
        # Add dinner constraint 
        mdl.add_constraint(mdl.sum(X[i, k] for i in F) == self.NUM_DAYS)
        
        # Add max lunch time constraint
        mdl.add_constraints(
            CUM_TIME[i] <= (X[i,k] * self.MAX_TIME_SPENT_PER_DAY) + 
            (self.MAX_HOUR_TO_LUNCH + time_spent[i]) for i in F
        )
        
        sol = mdl.solve()
        
        print("Done")

        self.X = X
        self.model = mdl
        self.sol = sol

        return X, mdl, sol

    def get_list_cycle(self):
        '''
        Function to get all cycles in our model starting from 0
        '''
        if not self.X or not self.model:
            print("Model hasn't been solved!!")
            return

        # Initialize necessary variables
        cycle_list = []

        for i in range(1, self.spots_df.shape[0]):
            if self.X[0, i].solution_value == 1:
                cycle_list.append(
                    self.__get_cycle_starting_from_idx([0, i])
                )
            
        return cycle_list