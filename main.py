
import os
import pandas as pd
import numpy as np
from food_optimizer import FoodOptimizer
from recreation_optimizer import RecreationOptimizer
from route_optimizer import SingleSourceRouteOptimizer

################
#  Input Data  #
################
num_days = 4
num_people = 2
food_budget = 400
# lodging_budget = 0
recreation_budget = 300
cur_loc_x = 20
cur_loc_y = 70

# Read data
DATA_PATH = "_data"
recreation = pd.read_csv(os.path.join(DATA_PATH, "recreation.csv"))
food = pd.read_csv(os.path.join(DATA_PATH, "food.csv"))


# CONSTANTS
MAX_TIME_SPENT_IN_RESTAURANT = 1.5
COST_PER_MILE = 0.3


###############################################
# Data Preparation for Optimizing Sub Problem #
###############################################

# Get satisfaction column by multiplying rating with the log of number of reviews
# this metric is subject to change
recreation["satisfaction"] = recreation["rating"] * np.log(recreation["num_reviewer"])
food["satisfaction"] = food["rating"] * np.log(food["num_reviewer"])

# Sanity check
assert "satisfaction" in recreation.columns
assert "satisfaction" in food.columns

##########################
# Solve the Sub-problems #
##########################

# Instantiate optimizer with given parameters
food_opt = FoodOptimizer(food, num_days, num_people, food_budget)
rec_opt = RecreationOptimizer(recreation, num_days, recreation_budget)

# Run the optimizer
food_model = food_opt.SolveUsingPyomo()
rec_model = rec_opt.SolveUsingPyomo()

# Grab the results
food_selected = food_opt.filter_selected()
rec_selected = rec_opt.filter_selected()

#############################################
# Data Preparation to Solve Routing Problem #
#############################################

# Subset the necessary column 
food_selected = food_selected.loc[:,["name", "location_x", "location_y"]]
rec_selected = rec_selected.loc[:,["name", "location_x", "location_y", "time_spent"]]
food_selected["time_spent"] = MAX_TIME_SPENT_IN_RESTAURANT 
food_selected["is_restaurant"] = 1
rec_selected["is_restaurant"] = 0

# Concate the dataframe into one
spots = pd.concat([pd.DataFrame({'name': ['current location'],
                                 'location_x': [cur_loc_x],
                                 'location_y': [cur_loc_y],
                                 'time_spent': [0],
                                 'is_restaurant': [0]}),
                   food_selected,
                   rec_selected], ignore_index=True)


#########################
# Solve Routing Problem #
#########################

# Instantiate optimizer with given parameters
route_optimizer = SingleSourceRouteOptimizer(
    (cur_loc_x, cur_loc_y), food_selected, rec_selected, num_days, COST_PER_MILE
)
X, mdl, sol = route_optimizer.SolveUsingDocplex()

# Get list of routes
routes = route_optimizer.get_routes()
route_dist = route_optimizer.get_total_distance()
route_cost = route_optimizer.get_total_cost()
print(routes)

################
# Route Result #
################
print()
print("*******************************************************")
print("*******************************************************")
print("\n\n")
print("Schedule:")
for i in range(len(routes)):
    print()
    print(f"Day {i+1}")
    for route in routes[i]:
        if route[1] != 0:
            print(f" * {spots.loc[route[1], 'name'].capitalize()}")
    print(f"Total day {i+1} distance: {round(route_dist[i], 3)}")
    print(f"Estimated cost: {round(route_cost[i], 3)}")

route_optimizer.plot_map()