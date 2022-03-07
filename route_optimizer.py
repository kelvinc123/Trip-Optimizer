from docplex.mp.model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SingleSourceRouteOptimizer:
    '''
    Route Optimizer class to solve single source routing problem
    '''
    
    def __init__(self, cur_loc, food_df, rec_df, num_days, cost_per_mile):

        '''
        Input:
            spots_df: Pandas DataFrame: Spots data frame
            num_days: Integer: Number of days for trip
            cost_per_mile: Integer: Estimated gas cost per mile
        '''

        # Constants
        self.MAX_TIME_SPENT_PER_DAY = 12
        self.MAX_HOUR_TO_LUNCH = 3

        # Input Data
        self.cur_loc = cur_loc
        self.food_df = food_df
        self.rec_df = rec_df
        self.spots_df = pd.concat(
            [pd.DataFrame({'name': ['current location'],
                           'location_x': [cur_loc[0]],
                           'location_y': [cur_loc[1]],
                           'time_spent': [0],
                           'is_restaurant': [0]}),
             food_df,
             rec_df], ignore_index=True)
        self.NUM_DAYS = num_days
        self.COST_PER_MILE = cost_per_mile

        # Model Data
        self.model = None
        self.X = None
        self.sol = None
        
    def __generate_dist_dict(self, spots_df):
        '''
        Helper function to generate distance dictionary from a given dataframe
        '''
        dist_dict = {}
        for i in range(spots_df.shape[0]):
            for j in range(spots_df.shape[0]):
                if i != j:
                    loc_1 = spots_df.loc[i,['location_x', 'location_y']].values.astype('float')
                    loc_2 = spots_df.loc[j,['location_x', 'location_y']].values.astype('float')
                    dist_dict[(i, j)] = np.linalg.norm(loc_1 - loc_2)
        return dist_dict

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
        S = [i for i in range(numSpots)]  # all spots
        N = [i for i in S if i != 0]  # all spots - current lodging
        A = [(i, j) for i in S for j in S if i != j]  # arcs/routes 
        F = list(self.spots_df[self.spots_df["is_restaurant"] == 1].index)  # Food
        R = [i for i in N if i not in F]  # Recreation
        k = 0  # index for current lodging location
        
        # Data
        dist = self.__generate_dist_dict(self.spots_df)
        time_spent = self.spots_df.loc[1:,'time_spent'].to_dict()
        is_restaurant = list(self.spots_df["is_restaurant"].values)

        # Create docplex model
        mdl = Model("Routing")
        
        # Add decision variable
        X = mdl.binary_var_dict(A, lb=0, ub=1, name="X")
        CUM_TIME = mdl.continuous_var_dict(S, ub=self.MAX_TIME_SPENT_PER_DAY, name='cum_time_spent')
        CUM_NUM_EATS = mdl.integer_var_dict(S, ub=2, name='cum_num_eats')
        
        # Add objective function
        mdl.minimize(mdl.sum(X[i, j] * dist[i, j] for i, j in A))
        
        # Add total outflow constraint
        mdl.add_constraints(mdl.sum(X[i, j] for j in S if j != i) == 1 for i in N)
        
        # Add total inflow constraint
        mdl.add_constraints(mdl.sum(X[i, j] for i in S if i != j) == 1 for j in N)
        
        # Total outflow and inflow from lodging node must be the equals to the number of days
        mdl.add_constraint(mdl.sum(X[i, k] for i in N) == self.NUM_DAYS)
        mdl.add_constraint(mdl.sum(X[k, j] for j in N) == self.NUM_DAYS)
        
        # Add cumulative sum constraint
        mdl.add_indicator_constraints(
            mdl.indicator_constraint(
                X[i, j], CUM_TIME[i] + time_spent[j] == CUM_TIME[j]
            ) for i, j in A if j != k
        )
        
        # Add lower bound constraint
        mdl.add_constraints(CUM_TIME[i] >= time_spent[i] for i in N)
        
        # Add max number of restaurant visited in a day constraint
        mdl.add_indicator_constraints(
            mdl.indicator_constraint(
                X[i, j], CUM_NUM_EATS[i] + is_restaurant[j] == CUM_NUM_EATS[j]
            ) for i, j in A if j != k
        )
        
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

    def __get_cycle_starting_from_idx(self, s_arc):
        
        '''
        Function to get cycle from a given 
        Input:
            model: Docplex Variable Solution: Docplex result
            s_arc: List: arc of the first route
        Output:
            cycle: list: sequence of visited location on a subtour starting from idx
        '''
        subset = []
        
        # starting location
        subset.append(s_arc)
        j = s_arc[1]

        # While we're not back to the starting location
        while j != s_arc[0]:
            
            # Go to the next location and add it to the cycle
            i = j
            for j in range(self.spots_df.shape[0]):
                # Since (i, i) is not an arc
                if i == j:
                    continue
                # Visit j after i
                if self.X[i, j].solution_value == 1:
                    subset.append([i, j])
                    break
        return subset

    def get_routes(self):
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

    def get_total_distance(self):
        '''
        Function to get total distance
        '''

        if not self.X or not self.model:
            return

        # Get distance dictionary
        dist = self.__generate_dist_dict(self.spots_df)

        # Create new list
        dist_list = []
        routes = self.get_routes()

        # Loop all trip days
        for route in routes:
            # Initialize total cost
            total_dist = 0

            # Loop all the arc in one day trip
            for arc in route:
                total_dist += dist[(arc[0], arc[1])]

            dist_list.append(total_dist)

        return dist_list

    def get_total_cost(self):
        '''
        Function to get total cost
        '''

        dist_list = self.get_total_distance()
        cost_list = []

        # Calculating cost per day
        for dist in dist_list:
            cost_list.append(dist * self.COST_PER_MILE)

        return cost_list
    
    def plot_map(self):
        '''
        Function to plot route map
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        ax = sns.scatterplot(
            x=[self.cur_loc[0]],
            y=[self.cur_loc[1]],
            color = "black",
            s=100,
            label="Lodging"
        )
        ax = sns.scatterplot(
            x=self.food_df['location_x'],
            y=self.food_df['location_y'],
            color="red",
            s=50,
            label="Restaurant"
        )
        ax = sns.scatterplot(
            x=self.rec_df['location_x'],
            y=self.rec_df['location_y'],
            color = "blue",
            s=50,
            label="Recreational Spot"
        )
        for x, y, t in zip(
            self.food_df['location_x'],
            self.food_df['location_y'],
            self.food_df['name']
        ):
            ax.text(x, y, t, color='black')
            
        for x, y, t in zip(
            self.rec_df['location_x'],
            self.rec_df['location_y'],
            self.rec_df['name']
        ):
            ax.text(x, y, t, color='black')

        for route in self.get_routes():
            for arc in route:
                x = [
                    self.spots_df.loc[arc[0], 'location_x'],
                    self.spots_df.loc[arc[1], 'location_x']
                    ]
                y = [
                    self.spots_df.loc[arc[0], 'location_y'],
                    self.spots_df.loc[arc[1], 'location_y']
                    ]
                ax = sns.lineplot(x=x, y=y, color = "black", linewidth=0.5)

        ax.set_title("Trip Plan")
        ax.legend()

        plt.show()