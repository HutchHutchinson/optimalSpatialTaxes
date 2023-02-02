
import numpy as np

from equilibrium_solver import EquilibriumSolver
from production_functions import CobbDouglasProduction
from utility_functions import CobbDouglasUtility
from utils import create_utility_list, create_prod_func_list 

J = 4
alpha = 0.5 
beta = 0.5

# upper = 0.5
# lower = 2.0
# primitives_dict = {'xi': [],
#                    'L': [],
#                    'A_c': [],
#                    'A_h': []
#                    } 

# for i in primitives_dict.keys():
#     primitives_dict[i] = np.random.uniform(upper, lower, J) 

primitives_dict = {'xi': [1, 1.5, 1.25, 1],
                   'L': [2, 1, 1.5, 1.25],
                   'A_c': [1.5, 1.75, 1.25, 1],
                   'A_h': [1.5, 1, 1.2, 1.5]
                   } 

utility_list = create_utility_list(J, alpha, primitives_dict['xi'], CobbDouglasUtility) 
c_production_list = create_prod_func_list(J, beta, primitives_dict['A_c'], CobbDouglasProduction) 
h_production_list = create_prod_func_list(J, beta, primitives_dict['A_h'], CobbDouglasProduction) 

j = 0
p = 1
N = 1

se = EquilibriumSolver(utility_list, c_production_list, h_production_list, primitives_dict['L']) 

print('Victory!')