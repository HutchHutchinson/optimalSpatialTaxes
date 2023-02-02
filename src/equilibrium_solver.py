import numpy as np
from scipy.optimize import root

class EquilibriumSolver:
    def __init__(self,
                 utility_list,
                 c_production_list,
                 h_production_list,
                 L,
                 T,
                 tau = 0,
                 tau_h=0,
                 R=0
                 ):
        self.J = len(utility_list) 
        self.utility_list = utility_list
        self.c_production_list = c_production_list
        self.h_production_list = h_production_list
        self.L = L
        self.T = T
        self.tau = tau
        self.tau_h = tau_h
        self.R = R 
    
    def conditional_fixed_point_mapping(self, p, L_c, N):
        "Create the fixed point mapping in prices, given a guess for L_c."
        output = np.empty(self.J)
        land_income = p
        non_labor_I = land_income + self.R 
        for j in self.J:
            h_j = self.utility_list[j]['h']
            F_jh = self.h_production_list[j]['F']
            N_jc = self.h_production_list[j]['N_c']
            I_j = (1-self.tau)*self.c_production_list[j]['F_N'] + non_labor_I 
            output[j] = N[j]*h_j(p[j], I_j) - F_jh(N_jc(L_c[j]), L_c[j]) 
        return output

    def local_excess_production(self, j, p_j, N_j, N_jc, L_jc):
        F_N_c = self.c_production_list[j]['F_N']
        F_L_c = self.c_production_list[j]['F_L']
        F_N_h = self.h_production_list[j]['F_N']
        F_L_h = self.h_production_list[j]['F_L'] 

        MP_c = np.array([
                         F_N_c(N_jc, L_jc), 
                         F_L_c(N_jc, L_jc) 
                        ]) 
        MP_h = np.array([
                         F_N_h(N_j-N_jc, self.L[j]-L_jc), 
                         F_L_h(N_j-N_jc, self.L[j]-L_jc) 
                        ]) 
        return MP_c - p_j*MP_h 
    
    def find_local_N_L(self, j, p, N):
        init_x = np.array([0.45, 1.05])  
        solution = root(lambda x: self.local_excess_production(j, p, N, x), 
                        init_x, method='hybr', options={'xtol': 1.49012e-12}) 
        return solution 
