
from numpy import random

from production_functions import CobbDouglasProduction
from utility_functions import CobbDouglasUtility
from utils import create_utility_list, create_prod_func_list 

J = 4
alpha = 0.5 
beta = 0.4

upper = 0.5
lower = 2.0
primitives_dict = {'xi': [],
                   'L': [],
                   'A_c': [],
                   'A_h': []
                   } 

for x in primitives_dict.keys():
    primitives_dict[x] = random.uniform(upper, lower, J) 
   
utility_list = create_utility_list(J, alpha, primitives_dict['xi'], CobbDouglasUtility) 
c_production_list = create_prod_func_list(J, beta, primitives_dict['A_c'], CobbDouglasProduction) 
h_production_list = create_prod_func_list(J, beta, primitives_dict['A_h'], CobbDouglasProduction) 

