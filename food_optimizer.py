# Imports
from pyomo.environ import *
from pyomo.opt import *
from pyomo.core import * 

class FoodOptimizer:
    '''
    Food Optimizer class to solve LP problem in maximizing total satisfiability
    by selecting restaurants
    '''
    
    def __init__(self, food_df, num_days, num_people, food_budget):
        '''
        Input:
            food_df: Pandas DataFrame: Food data frame
            num_days: Integer: Number of days for trip
            num_people: Integer: Number of people in the trip
            food_budget: Float: Budget limit to spend on restaurants
        '''

        # Input Data
        self.food_df = food_df
        self.num_days = num_days
        self.num_people = num_people
        self.food_budget = food_budget

        # Model Data
        self.model = None

    def __objective_rule(self, model):
        # Create objective function
        return sum(model.satisfaction[i] * model.SELECT[i] for i in model.i)

    def __budget_rule(self, model):
        # Constraint on available budget 
        return sum(model.cost[i] * model.SELECT[i] * model.person for i in model.i) <= model.budget

    def __eat_twice_a_day_rule(self, model):
        # Constraint on requiring to eat exactly twice a day
        return sum(model.SELECT[i] for i in model.i) == (model.days * 2)

    def SolveUsingPyomo(self, verbose=True):
        '''
        Function to create and solve a concrete Pyomo model 
        Output:
            model: Pyomo Model: Model result
        '''

        if verbose:
            print("\nOptimizing restaurant satisfaction...")

        # Initialize data structures
        numRestaurants = self.food_df.shape[0] # Items
        cost = {i+1: self.food_df.loc[i, "cost_per_person"] for i in range(numRestaurants)} # Dict of cost
        satisfaction = {i+1: self.food_df.loc[i, "satisfaction"] for i in range(numRestaurants)} # Dict of satisfaction

        # Create a concrete Pyomo model
        if verbose:
            print("\tBuilding Pyomo model...")
        model = ConcreteModel() 

        # Define indices and sets 
        if verbose:
            print("\tCreating indices and set...")
        model.i = Set(initialize=[i for i in range(1, numRestaurants+1)], ordered=True) # Index on each restaurant

        # Define variables
        if verbose:
            print("\tCreating variables...")
        model.SELECT = Var(model.i, domain=Binary, initialize = 0) # Indicates if restaurant i is selected

        # Create parameters (i.e., data)
        if verbose:
            print("\tCreating parameters...")
        model.cost = Param(model.i, initialize = cost)
        model.satisfaction = Param(model.i, initialize = satisfaction)
        model.budget = Param(initialize = self.food_budget)
        model.person = Param(initialize = self.num_people)
        model.days = Param(initialize = self.num_days)

         # Create objective function
        if verbose:
            print("\tCreating objective function...")
        model.objective = Objective(rule=self.__objective_rule, sense = maximize)

        if verbose:
            print("\tCreating constraint on available budget...")
        model.budgetConstraint = Constraint(rule=self.__budget_rule)

        if verbose:
            print("\tCreating constraint on requirement to eat exactly twice a day...")
        model.eatFrequencyConstraint = Constraint(rule=self.__eat_twice_a_day_rule)

        if verbose:
            print("\tDone.")
            print("\tRunning solver...")

        opt = SolverFactory("glpk")
        opt.solve(model) # This runs the solver
        
        if verbose:
            print("Done optimizing restaurant choices.")

        self.model = model

        return model

    def __get_selected_index(self):
        '''
        Helper function to get selected decision variable index
        '''
        if not self.model:
            return

        selected_idx = []
        for i in range(self.food_df.shape[0]):
            if self.model.SELECT[i+1].value == 1:
                selected_idx.append(i)
        
        return selected_idx

    def filter_selected(self, verbose=True):    
        '''
        Function to filter selected variable from pyomo model result
        '''
        if not self.model:
            print("Model hasn't been solved!!")
            return
        selected_idx = self.__get_selected_index()
        print("\nRestaurants:")
        if verbose:
            for i in range(self.food_df.shape[0]):
                print(f" - {self.food_df.loc[i, 'name']}", end=' ')
                if i in selected_idx:
                    print(f"(selected)", end='')
                print()
                
        return self.food_df.iloc[selected_idx].reset_index(drop=True)

    def get_total_cost(self):
        '''
        Function to get total cost
        '''
        if not self.model:
            print("Model hasn't been solved!!")
            return

        selected_idx = self.__get_selected_index()
        return sum(self.food_df.loc[selected_idx, 'cost_per_person']) * self.num_people
        