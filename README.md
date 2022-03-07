# Trip Optimizer

An automated trip planner given the restaurants and recreational spots data. This program solves three problems: Choosing the restaurants, choosing the recreational spots, and scheduling the routes.

The inputs for this program are:
 * Number of days
 * Number of people
 * Food budget
 * Recreation budget

Columns for restaurants data: (*name*, *cost_per_person*, *rating*, *num_reviewer*, *location_x*, *location_y*)
Columns for recreations data: (*name*, *rating*, *num_reviewer*, *cost*, *location_x*, *location_y*, *time_spent*)

The current data are hand-generated data, they can be substituted with the real data as long as the required columns are available. The variables *location_x* and *location_y* are taken randomly from 0 to 100. These variables can be substituted with the geospatial data (*latitute* and *longitude*). To choose the restaurants and recreational spots, the new satisfaction metric is created by multiplying rating with the log of number reviewer. 

The program used the hybrid optimization framework: **Integer Programming** and **Network Flow Problem**. To see the formulation, go to *formulation* directory. 

## Run the Program
To run the program, use *main.py* scripts or open the *Trip Optimizer.ipynb* notebook on jupyter.

## Requirements
To install the requirements, follow these steps:
1. Create new virtual environment named *venv* (it's on the gitignore): `python -m venv venv`
2. Activate the virtual environment:
    1. On Windows: `.\venv\Scripts\activate`
        * Note that if there's an error in execution of scripts, run `Set-ExecutionPolicy Unrestricted -Scope Process` before activating environment.
    2. On Mac: `source venv/bin/activate`
3. Install the packages from *requirements.txt*: `pip install requirements.txt`