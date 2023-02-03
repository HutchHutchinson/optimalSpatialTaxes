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
                 tau_h = 0,
                 R=0
                 ):
        self.J = len(utility_list) 
        self.utility_list = utility_list
        self.c_production_list = c_production_list
        self.h_production_list = h_production_list
        self.L = np.array(L)
        self.T = np.array(T) 
        self.tau = tau
        self.tau_h = tau_h
        self.R = R  
    
    def prices_mapping(self, p, N, F_h, w):
        "Create the fixed point mapping in prices, given a guess for L_c."
        excess_housing_demand = np.empty(self.J)
        q = p * F_h 
        non_labor_I = q @ self.L + self.R 
        h = [self.utility_list[x]['h'] for x in range(self.J)]
        for j in range(self.J):
            I_j = (1-self.tau)*w[j] - self.T[j] + non_labor_I 
            excess_housing_demand[j] = N[j]*h[j](p[j] + self.tau_h, I_j) - F_h[j] 
        return excess_housing_demand 

    def find_prices_fixed_point(self, N, L_c):
        N_c = np.empty(self.J) 
        F_h = np.empty(self.J) 
        w = np.empty(self.J)  

        for x in range(self.J):
            N_c[x] = self.h_production_list[x]['N_c'](N[x], self.L[x], L_c[x]) 
            F_h[x] = self.h_production_list[x]['F'](N[x] - N_c[x]
                                                    , self.L[x] - L_c[x])  
            w[x] = self.c_production_list[x]['F_N'](N_c[x], L_c[x]) 

        init_p = np.ones(self.J)
        solution = root(lambda p: self.prices_mapping(p, N, F_h, w), 
                        init_p, method='hybr') 
        return solution 

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
