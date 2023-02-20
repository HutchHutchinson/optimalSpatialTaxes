
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
                   'L': [1, 1, 1, 1],
                   'A_c': [1.1, 1.25, 1.25, 1.1],
                   'A_h': [1.25,1.1, 1.1, 1.25] 
                   } 

utility_list = create_utility_list(J, alpha, primitives_dict['xi'], CobbDouglasUtility) 
c_production_list = create_prod_func_list(J, beta, primitives_dict['A_c'], CobbDouglasProduction) 
h_production_list = create_prod_func_list(J, beta, primitives_dict['A_h'], CobbDouglasProduction) 

se = EquilibriumSolver(utility_list, 
                       c_production_list, 
                       h_production_list, 
                       primitives_dict['L'],
                       T=np.zeros(J)
                       ) 

N_guess = np.array([0.25, 0.25, 0.25, 0.25])
L_c_guess = np.array(primitives_dict['L']) / 2

# N_c = np.empty(se.J) 
# F_h = np.empty(se.J) 
# w = np.empty(se.J) 
# F_N_h = np.empty(se.J) 

# for x in range(se.J):
#     N_c[x] = se.h_production_list[x]['N_c'](N[x], se.L[x], L_c[x]) 
#     F_h[x] = se.h_production_list[x]['F'](N[x] - N_c[x]
#                                             , se.L[x] - L_c[x])  
#     w[x] = se.c_production_list[x]['F_N'](N_c[x], L_c[x]) 
#     F_N_h[x] = se.h_production_list[x]['F_N'](N[x] - N_c[x]
#                                                 , se.L[x] - L_c[x]) 

# p_guess = np.ones(se.J) 
# p_star = se.fixed_point_solver(p_guess, se.excess_housing_demand, [N, F_h, w], verbose=False)

L_c_star = se.fixed_point_solver(L_c_guess, se.excess_wages, [N_guess], kappa=0.1) 

# j=3
# L_c_grid = np.linspace(0.01, primitives_dict['L'][j]-0.01, 100) 
# excess_wage_graph = np.empty(100) 

# for i in range(100):
#     L_c[j] = L_c_grid[i]
#     excess_wage_graph[i] = se.excess_wages(N, L_c)[j] 

#L_c_star = se.find_L_c_fixed_point(N) 

#se.excess_wages(N, L_c)

print('Victory!')