# Imports
from pyomo.environ import *
from pyomo.opt import *
from pyomo.core import * 

class RecreationOptimizer:
    '''
    Recreational Spot Optimizer class to solve LP problem in maximizing total satisfiability
    by selecting spots
    '''
    
    def __init__(self, recreation_df, num_days, rec_budget):
        '''
        Input:
            recreation_df: Pandas DataFrame: Recreational spots data frame
            num_days: Integer: Number of days for trip
            rec_budget: Float: Budget limit to spend on recreational spots
        Output:
            model: Pyomo Model: Model result
        '''

        # Constants
        self.MAX_HOURS_PER_DAY = 9
        self.MAX_REC_PER_DAY_ON_AVG = 2

        # Input data
        self.recreation_df = recreation_df
        self.num_days = num_days
        self.rec_budget = rec_budget
        self.model = None

    def __objective_rule(self, model):
        # Create objective function
        return sum(model.satisfaction[i] * model.SELECT[i] for i in model.i)

    def __budget_rule(self, model):
        # Constraint on available budget 
        return sum(model.cost[i] * model.SELECT[i] for i in model.i) <= model.budget
    
    def __time_rule(self, model):
        # Constraint on available time
        return sum(model.time[i] * model.SELECT[i] for i in model.i) <= (model.days * self.MAX_HOURS_PER_DAY)
    
    def __three_places_rule(self, model):
        # Constraint on no more than three places a day on average
        return sum(model.SELECT[i] for i in model.i) <= (model.days * self.MAX_REC_PER_DAY_ON_AVG)

    def SolveUsingPyomo(self, verbose=True):
        '''
        Function to create and solve a concrete Pyomo model 
        Output:
            model: Pyomo Model: Model result
        '''

        if verbose:
            print("\nOptimizing restaurant satisfaction...")

        # Initialize data structures
        numSpots = self.recreation_df.shape[0] # Items
        cost = {i+1: self.recreation_df.loc[i, "cost"] for i in range(numSpots)} # Dict of cost
        satisfaction = {i+1: self.recreation_df.loc[i, "satisfaction"] for i in range(numSpots)} # Dict of satisfaction
        time = {i+1: self.recreation_df.loc[i, "time_spent"] for i in range(numSpots)} # Dict of time spent

        # Create a concrete Pyomo model
        if verbose:
            print("\tBuilding Pyomo model...")
        model = ConcreteModel() 

        # Define indices and sets 
        if verbose:
            print("\tCreating indices and set...")
        model.i = Set(initialize=[i for i in range(1, numSpots+1)], ordered=True) # Index on each restaurant

        # Define variables
        if verbose:
            print("\tCreating variables...")
        model.SELECT = Var(model.i, domain=Binary, initialize = 0) # Indicates if restaurant i is selected

        # Create parameters (i.e., data)
        if verbose:
            print("\tCreating parameters...")
        model.cost = Param(model.i, initialize = cost)
        model.satisfaction = Param(model.i, initialize = satisfaction)
        model.time = Param(model.i, initialize = time)
        model.budget = Param(initialize = self.rec_budget)
        model.days = Param(initialize = self.num_days)

         # Create objective function
        if verbose:
            print("\tCreating objective function...")
        model.objective = Objective(rule=self.__objective_rule, sense = maximize)

        # Create constraints
        if verbose:
            print("\tCreating constraint on available budget...")
        model.budgetConstraint = Constraint(rule=self.__budget_rule)

        if verbose:
            print("\tCreating constraint on available time...")
        model.timeConstraint = Constraint(rule=self.__time_rule)
        
        if verbose:
            print("\tCreating constraint on three places a day...")
        model.threePlacesConstraint = Constraint(rule=self.__three_places_rule)

        if verbose:
            print("\tDone.")
            print("\tRunning solver...")

        opt = SolverFactory("cplex")
        opt.solve(model) # This runs the solver
        if verbose:
            print("Done optimizing recreational spot choices.")

        self.model = model

        return model

    def __get_selected_index(self):
        '''
        Helper function to get selected decision variable index
        '''
        if not self.model:
            return

        selected_idx = []
        for i in range(self.recreation_df.shape[0]):
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
        print("\nRecreational Spots:")
        if verbose:
            for i in range(self.recreation_df.shape[0]):
                print(f" - {self.recreation_df.loc[i, 'name']}", end=' ')
                if i in selected_idx:
                    print(f"(selected)", end='')
                print()
                
        return self.recreation_df.iloc[selected_idx].reset_index(drop=True)

    def get_total_cost(self):
        '''
        Function to get total cost
        '''
        if not self.model:
            print("Model hasn't been solved!!")
            return

        selected_idx = self.__get_selected_index()
        return sum(self.recreation_df.loc[selected_idx, 'cost'])