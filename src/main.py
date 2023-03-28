
import numpy as np

from equilibrium_solver import EquilibriumSolver
from production_functions import CobbDouglasProduction
from utility_functions import CobbDouglasUtility
from utils import create_utility_list, create_prod_func_list 

J = 4
alpha = 0.5 
beta = 0.5
tau_c = 0
tau_h = 0

primitives_dict = {'xi': [1, 1.5, 1.25, 1.1],
                   'L': [1.2, 1, 0.9, 1.1],
                   'A_c': [1.1, 1.25, 1.3, 1.15],
                   'A_h': [1.25,1.15, 1.1, 1.25] 
                   } 

utility_list = create_utility_list(J, 
                                   alpha, 
                                   primitives_dict['xi'], 
                                   tau_c, 
                                   tau_h, 
                                   CobbDouglasUtility) 
c_production_list = create_prod_func_list(J, primitives_dict['A_c'], beta, CobbDouglasProduction) 
h_production_list = create_prod_func_list(J, primitives_dict['A_h'], beta, CobbDouglasProduction) 

se = EquilibriumSolver(utility_list, 
                       c_production_list, 
                       h_production_list, 
                       primitives_dict['L'],
                       T=np.zeros(J)
                       ) 

se.calc_N_star()
se.calc_equilibrium_objects()
se.check_equilibrium_conditions()

print('Equilibrium found!')