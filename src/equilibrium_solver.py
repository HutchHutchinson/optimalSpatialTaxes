import numpy as np
from scipy.optimize import root

class EquilibriumSolver:
    def __init__(self,
                 utility_list,
                 c_production_list,
                 h_production_list,
                 L,
                 T=None,
                 tau=0,
                 tau_h=0,
                 R=0
                 ):
        self.J = len(utility_list) 
        self.L = np.array(L) 
        self.tau = tau
        self.tau_h = tau_h
        self.R = R  
        if T is None:
            self.T = np.zeros(T)
        else:
            self.T = np.array(T)   

        self.h_list = [utility_list[x]['h'] for x in range(self.J)] 
        self.N_c_list  = [h_production_list[x]['N_c'] for x in range(self.J)]
        self.F_h_list  = [h_production_list[x]['F'] for x in range(self.J)]
        self.F_N_c_list  = [c_production_list[x]['F_N'] for x in range(self.J)] 
        self.F_N_h_list  = [h_production_list[x]['F_N'] for x in range(self.J)]
        self.F_L_h_list  = [h_production_list[x]['F_L'] for x in range(self.J)] 

    def calc_step_size(self,
                       excess_function,
                       solution_guess,
                       params,
                       max_first_step_size=0.01
                       ):
        excess_val = excess_function(solution_guess, *params) 
        kappa_vec = (max_first_step_size * solution_guess) / excess_val
        abs_kappa_vec = np.abs(kappa_vec)
        return np.min(abs_kappa_vec)  
    
    def fixed_point_solver(self, 
                           excess_function,
                           solution_guess,
                           params,
                           max_iter=10000,
                           tol=1e-6, 
                           verbose=True,
                           print_freq=10):
        
        old = solution_guess
        new = np.empty_like(old)
        warning = "Solver encountered non-positive values."

        kappa = self.calc_step_size(excess_function, solution_guess, params) 

        j = 0
        error = tol + 1
        while j < max_iter and error > tol:
            new = old + kappa * excess_function(old, *params) 
            assert np.min(new) > 0, warning
            error = np.max(np.abs(new - old))
            j += 1
            old = new 
            if verbose:
                if j % print_freq == 0:
                    print(j, excess_function(old, *params)) 
            if j == max_iter and error > tol:
                print("Max iterations reached. Solver exited.") 
        return new      

    def excess_housing_demand(self, p, N, F_h, F_N_c, F_L_h):
        J = self.J
        h = self.h_list 
        L = self.L
        R = self.R
        T = self.T 
        tau = self.tau
        tau_h = self.tau_h

        excess_housing_demand = np.empty(J) 
        q = p * F_L_h 
        location_independent_I = q @ L + R 
        for j,x in enumerate(h):
            I_j = (1-tau)*F_N_c[j] - T[j] + location_independent_I 
            excess_housing_demand[j] = N[j]*x(p[j] + tau_h, I_j) - F_h[j] 
        return excess_housing_demand 

    def calc_L_c_dependent_objects(self, L_c, N, N_c):
        J, L = self.J, self.L
        F_h_list = self.F_h_list
        F_N_c_list = self.F_N_c_list
        F_N_h_list = self.F_N_h_list
        F_L_h_list = self.F_L_h_list
        
        F_h = np.empty(J) 
        F_N_c = np.empty(J) 
        F_N_h = np.empty(J) 
        F_L_h = np.empty(J) 

        for x in range(self.J):
            F_N_c[x] = F_N_c_list[x](N_c[x], L_c[x]) 
            F_h[x] = F_h_list[x](N[x] - N_c[x], L[x] - L_c[x])   
            F_N_h[x] = F_N_h_list[x](N[x] - N_c[x], L[x] - L_c[x])  
            F_L_h[x] = F_L_h_list[x](N[x] - N_c[x], L[x] - L_c[x])  
        
        return F_N_c, F_h, F_N_h, F_L_h 
    
    def excess_wages(self, L_c, N):
        J, L, N_c_list = self.J, self.L, self.N_c_list
        N_c = np.empty(J) 
        for j,x in enumerate(N_c_list):
            N_c[j] = x(N[j], L[j], L_c[j])  

        F_N_c, F_h, F_N_h, F_L_h = self.calc_L_c_dependent_objects(L_c, 
                                                                   N, 
                                                                   N_c)        

        p_guess = np.ones(self.J) 
        p = self.fixed_point_solver(self.excess_housing_demand, 
                                    p_guess, 
                                    [N, F_h, F_N_c, F_L_h],
                                    verbose=False)  
        
        return F_N_c - p * F_N_h 
