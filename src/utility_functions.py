
class CobbDouglasUtility:
    def __init__(self, 
                 alpha, #budget share of housing
                 xi):   #amenity vaulues of the location
        self.alpha = alpha
        self.xi = xi 
        
        self.v_cons = self.alpha**self.alpha * (1-self.alpha)**(1-self.alpha) 
    
    def u(self, c, h):
        "Utility."
        alpha, xi = self.alpha, self.xi
        return xi * c**(1-alpha) * h**alpha 
    
    def v(self, p, I):
        "Indirect utility."
        alpha, xi, v_cons = self.alpha, self.xi, self.v_cons
        return v_cons * xi * I * p**(-alpha) 
    
    def c(self, I):
        "Consumption good demand function."
        alpha = self.alpha
        return (1-alpha) * I
    
    def h(self, p, I):
        "Housing good demand function."
        alpha = self.alpha
        return (alpha * I) / p
    
    def mu_c(self, c, h):
        "Marginal utility of consumption."
        alpha, xi = self.alpha, self.xi
        return (1-alpha) * xi * c**(-alpha) * h**alpha 
    
    def mu_h(self, c, h):
        "Marginal utility of housing."
        alpha, xi = self.alpha, self.xi
        return alpha * xi * c**(1-alpha) * h**(alpha-1) 
    
    def MRS_hc(self, c, h):
        "Marginal rate of substitution between housing and consumption."
        return self.mu_h(c, h) / self.mu_c(c, h) 
    