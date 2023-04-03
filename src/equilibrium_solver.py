import numpy as np

class EquilibriumSolver:
    def __init__(self,
                 utility_list,
                 c_production_list,
                 h_production_list,
                 L,
                 tau_N,
                 tau_L,
                 T=None,
                 R=0
                 ):
        self.J = len(utility_list) 
        self.L = np.array(L) 
        self.tau_N = tau_N
        self.tau_L = tau_L 
        if T is None:
            self.T = np.zeros(T)
        else:
            self.T = np.array(T) 
        self.R = R   

        # Assignment for improved readability.
        self.c_list = [utility_list[x]['c'] for x in range(self.J)] 
        self.h_list = [utility_list[x]['h'] for x in range(self.J)] 
        self.v_list = [utility_list[x]['v'] for x in range(self.J)] 

        self.F_c_list  = [c_production_list[x]['F'] for x in range(self.J)] 
        self.F_N_c_list  = [c_production_list[x]['F_N'] for x in range(self.J)] 
        self.F_L_c_list  = [c_production_list[x]['F_L'] for x in range(self.J)] 

        self.N_c_list  = [h_production_list[x]['N_c'] for x in range(self.J)]

        self.F_h_list  = [h_production_list[x]['F'] for x in range(self.J)]
        self.F_N_h_list  = [h_production_list[x]['F_N'] for x in range(self.J)]
        self.F_L_h_list  = [h_production_list[x]['F_L'] for x in range(self.J)] 
    
    def fixed_point_solver(self, 
                           excess_function,
                           solution_guess,
                           params,
                           kappa = 0.05,
                           max_iter=10000,
                           tol=1e-6, 
                           verbose=True,
                           print_freq=10):
        
        old = solution_guess
        new = np.empty_like(old)
        warning = "Solver encountered non-positive values."

        j = 0
        error = tol + 1
        while j < max_iter and error > tol:
            if params:
                new = old + kappa * excess_function(old, *params) 
            else:
                new = old + kappa * excess_function(old) 
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
    
    def fixed_point_solver_numba(self, 
                                excess_function,
                                solution_guess,
                                kappa = 0.05,
                                max_iter=10000,
                                tol=1e-6, 
                                verbose=True,
                                print_freq=10):
        
        old = solution_guess
        new = np.empty_like(old)
        warning = "Solver encountered non-positive values."

        j = 0
        error = tol + 1
        while j < max_iter and error > tol:
            new = old + kappa * excess_function(old) 
            assert np.min(new) > 0, warning
            error = np.max(np.abs(new - old))
            j += 1
            old = new 
            if verbose:
                if j % print_freq == 0:
                    print(j, excess_function(old)) 
            if j == max_iter and error > tol:
                print("Max iterations reached. Solver exited.") 
        return new  

    def calc_income(self, p, F_L_h, F_N_c):
        L, R, T, tau_N, tau_L = self.L, self.R, self.T, self.tau_N, self.tau_L
        I = np.empty(self.J) 
        
        q = p * F_L_h - tau_L 
        location_independent_I = q @ L + R 
        w = F_N_c - tau_N 
        for j in range(self.J):
            I[j] = w[j] - T[j] + location_independent_I        
        return I 

    def excess_housing_demand(self, p, N, F_h, F_N_c, F_L_h):
        # Assignment for improved readability.
        h = self.h_list

        excess_housing_demand = np.empty(self.J) 
        I = self.calc_income(p, F_L_h, F_N_c)
        for j,x in enumerate(h):
            excess_housing_demand[j] = N[j]*x(p[j], I[j]) - F_h[j] 
        return excess_housing_demand 

    def calc_N_c(self, L_c, N):
        N_c = np.empty(self.J)
        for j,x in enumerate(self.N_c_list):
            N_c[j] = x(N[j], self.L[j], L_c[j]) 
        return N_c 

    def calc_prod_func_objects(self, 
                               L_c, 
                               N, 
                               N_c,
                               pf_object_list, 
                               consumption=True
                               ):
        L = self.L 
        pf_object = np.empty(self.J) 
        
        for x in range(self.J):
            if consumption:
                pf_object[x] = pf_object_list[x](N_c[x], L_c[x]) 
            else:
                pf_object[x] = pf_object_list[x](N[x] - N_c[x], 
                                                 L[x] - L_c[x]) 
        return pf_object
    
    def excess_wages(self, L_c, N):  
        N_c = self.calc_N_c(L_c, N) 
        F_N_c = self.calc_prod_func_objects(L_c, N, N_c, self.F_N_c_list) 
        F_h = self.calc_prod_func_objects(L_c, N, N_c, 
                                          self.F_h_list, consumption=False) 
        F_N_h = self.calc_prod_func_objects(L_c, N, N_c, 
                                          self.F_N_h_list, consumption=False)
        F_L_h = self.calc_prod_func_objects(L_c, N, N_c, 
                                          self.F_L_h_list, consumption=False)      

        p_guess = np.ones(self.J) 
        p = self.fixed_point_solver(self.excess_housing_demand, 
                                    p_guess, 
                                    [N, F_h, F_N_c, F_L_h],
                                    kappa = 0.2,
                                    verbose=False)  
       
        return  F_N_c - p * F_N_h 
    
    def indirect_utility(self, p, I):
        v = np.empty(self.J) 
        for j,x in enumerate(self.v_list):
            v[j] = x(p[j], I[j]) 
        return v
    
    def excess_indirect_utility(self, N):
        N_0 = 1-N.sum()
        full_N = np.append(N_0, N) 

        L_c_guess = self.L / 2 
        L_c = self.fixed_point_solver(self.excess_wages, 
                                           L_c_guess, 
                                           [full_N], 
                                           kappa=0.2,
                                           verbose=False) 

        N_c = self.calc_N_c(L_c, full_N) 
        F_N_c = self.calc_prod_func_objects(L_c, full_N, N_c, self.F_N_c_list) 
        F_h = self.calc_prod_func_objects(L_c, full_N, N_c, 
                                          self.F_h_list, consumption=False) 
        F_L_h = self.calc_prod_func_objects(L_c, full_N, N_c, 
                                          self.F_L_h_list, consumption=False)        

        p_guess = np.ones(self.J) 
        p = self.fixed_point_solver(self.excess_housing_demand, 
                                    p_guess, 
                                    [full_N, F_h, F_N_c, F_L_h],
                                    kappa=0.2,
                                    verbose=False) 
        
        I = self.calc_income(p, F_L_h, F_N_c) 
        v = self.indirect_utility(p, I)

        return v[1:] - v[0] 
    
    def calc_N_star(self):
        N_guess = 1 / self.J 
        N_guess_vector = N_guess * np.ones(self.J-1)
        N_star_sub = self.fixed_point_solver(self.excess_indirect_utility, 
                                             N_guess_vector, 
                                             [], 
                                             kappa=0.05,
                                             verbose=True) 
        N_star_0 = 1 - N_star_sub.sum()
        self.N_star = np.append(N_star_0, N_star_sub) 

    def calc_equilibrium_objects(self):
        L_c_guess = self.L / 2 
        self.L_c_star = self.fixed_point_solver(self.excess_wages, 
                                                L_c_guess, 
                                                [self.N_star], 
                                                kappa=0.2,
                                                verbose=False) 
        
        self.N_c_star = self.calc_N_c(self.L_c_star, self.N_star) 

        self.F_c_star = self.calc_prod_func_objects(self.L_c_star, 
                                                    self.N_star, 
                                                    self.N_c_star, 
                                                    self.F_c_list) 
        
        self.F_N_c_star = self.calc_prod_func_objects(self.L_c_star, 
                                                      self.N_star, 
                                                      self.N_c_star, 
                                                      self.F_N_c_list) 
        
        self.F_L_c_star = self.calc_prod_func_objects(self.L_c_star, 
                                                      self.N_star, 
                                                      self.N_c_star, 
                                                      self.F_L_c_list) 
        
        self.F_h_star = self.calc_prod_func_objects(self.L_c_star, 
                                                    self.N_star, 
                                                    self.N_c_star, 
                                                    self.F_h_list,
                                                    consumption=False) 
        
        self.F_N_h_star = self.calc_prod_func_objects(self.L_c_star, 
                                                      self.N_star, 
                                                      self.N_c_star, 
                                                      self.F_N_h_list,
                                                      consumption=False) 
        
        self.F_L_h_star = self.calc_prod_func_objects(self.L_c_star, 
                                                      self.N_star, 
                                                      self.N_c_star, 
                                                      self.F_L_h_list,
                                                      consumption=False) 
        
        p_guess = np.ones(self.J) 
        self.p_star = self.fixed_point_solver(self.excess_housing_demand, 
                                              p_guess, 
                                              [self.N_star, 
                                               self.F_h_star, 
                                               self.F_N_c_star, 
                                               self.F_L_h_star],
                                              kappa=0.2,
                                              verbose=False) 
        
        self.I_star = self.calc_income(self.p_star, 
                                       self.F_L_h_star, 
                                       self.F_N_c_star) 
        self.v_star = self.indirect_utility(self.p_star, 
                                            self.I_star) 
        
        self.c_star = np.empty(self.J)
        for j,x in enumerate(self.c_list):
            self.c_star[j] = x(self.I_star[j]) 

        self.h_star = np.empty(self.J)
        for j,x in enumerate(self.h_list):
            self.h_star[j] = x(self.p_star[j], self.I_star[j]) 

        self.w_star = self.F_N_c_star - self.tau_N 
        self.q_star = self.p_star * self.F_L_h_star - self.tau_L 

    def check_equilibrium_conditions(self, tol=0.005):
        v_error = np.empty([self.J, self.J]) 
        for i in range(self.J):
            for k in range(self.J):
                v_error[i, k] = np.abs(self.v_star[i] - self.v_star[k]) 
        if np.min(v_error) < tol:
            print('Spatial indifference is satisfied.')
        else:
            print('Spatial indifference is not satisfied.')  

        total_c_consumption = (self.N_star * self.c_star).sum()
        total_G = (self.tau_L*self.L).sum() + self.tau_N
        total_c_production = self.F_c_star.sum()
        c_error = np.abs(total_c_consumption + total_G - total_c_production)
        if c_error < tol:
            print('The consumption good market is cleared.')
        else:
            print('The consumption good market is not cleared.') 

        h_consumption = self.N_star * self.h_star
        h_error = np.min(np.abs(h_consumption - self.F_h_star))
        if h_error < tol:
            print('The housing good markets are cleared.')
        else:
            print('The housing good markets are not cleared.') 

        mp_N_error = np.min(np.abs(self.F_N_c_star - self.w_star - self.tau_N))
        if mp_N_error < tol:
            print('MP of labor = w + tau_N.')
        else:
            print('MP of labor != w + tau_N.') 
       
        mrp_N = self.p_star * self.F_N_h_star
        mrp_N_error = np.min(np.abs(mrp_N - self.w_star - self.tau_N))
        if mrp_N_error < tol:
            print('MRP of labor = w + tau_N.')
        else:
            print('MRP of labor != w + tau_N.')    

        mp_L_error = np.min(np.abs(self.F_L_c_star - self.q_star - self.tau_L))
        if mp_L_error < tol:
            print('MP of land = q + tau_L.')
        else:
            print('MP of land != q + tau_L.') 
       
        mrp_L = self.p_star * self.F_L_h_star
        mrp_L_error = np.min(np.abs(mrp_L - self.q_star - self.tau_L))
        if mrp_L_error < tol:
            print('MRP of land = q + tau_L.')
        else:
            print('MRP of land != q + tau_L.')  
