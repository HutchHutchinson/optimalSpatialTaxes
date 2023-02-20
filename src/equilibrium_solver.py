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

    def fixed_point_solver(self, 
                           solution_guess, 
                           excess_function, 
                           params,
                           kappa=0.2,
                           max_iter=1000,
                           tol=1e-6, 
                           verbose=True,
                           print_freq=20):
        
        old = solution_guess
        new = np.empty_like(old)
        string = "Solver encountered non-positive values - reduce kappa."

        j = 0
        error = tol + 1
        while j < max_iter and error > tol:
            new = old + kappa * excess_function(old, *params) 
            
            assert np.min(new) > 0, string

            error = np.max(np.abs(new - old))
            j += 1
            old = new 
            if verbose:
                if j % print_freq == 0:
                    print(j, excess_function(old, *params)) 
            if j == max_iter and error > tol:
                print("Max iterations reached. Solver exited.") 
        return new      

    def excess_housing_demand(self, p, N, F_h, w):
        "Create the fixed point mapping in prices, given a guess for L_c."
        excess_housing_demand = np.empty(self.J)
        q = p * F_h 
        non_labor_I = q @ self.L + self.R 
        h = [self.utility_list[x]['h'] for x in range(self.J)]
        for j in range(self.J):
            I_j = (1-self.tau)*w[j] - self.T[j] + non_labor_I 
            excess_housing_demand[j] = N[j]*h[j](p[j] + self.tau_h, I_j) - F_h[j] 
        return excess_housing_demand 

    def excess_wages(self, L_c, N):
        N_c = np.empty(self.J) 
        F_h = np.empty(self.J) 
        w = np.empty(self.J) 
        F_N_h = np.empty(self.J) 

        for x in range(self.J):
            N_c[x] = self.h_production_list[x]['N_c'](N[x], self.L[x], L_c[x]) 
            F_h[x] = self.h_production_list[x]['F'](N[x] - N_c[x]
                                                    , self.L[x] - L_c[x])  
            w[x] = self.c_production_list[x]['F_N'](N_c[x], L_c[x]) 
            F_N_h[x] = self.h_production_list[x]['F_N'](N[x] - N_c[x]
                                                        , self.L[x] - L_c[x]) 

        p_guess = np.ones(self.J) 
        p = self.fixed_point_solver(p_guess, 
                                    self.excess_housing_demand, 
                                    [N, F_h, w],
                                    kappa = 0.3,
                                    verbose=False)  

        return w - p * F_N_h 

        def calc_L_c_star(self):