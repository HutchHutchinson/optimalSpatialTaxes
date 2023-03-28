
class CobbDouglasUtility:
    def __init__(self, 
                 alpha,  #budget share of housing
                 xi,     #amenity vaulues of the location
                 tau_c,  #consumption good tax
                 tau_h): #housing good tax   
        self.alpha = alpha
        self.xi = xi 
        self.tau_c, self.tau_h = tau_c, tau_h
        
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
        alpha, tau_c = self.alpha, self.tau_c
        num = (1-alpha) * I 
        denom = 1 + tau_c
        return num / denom 
    
    def h(self, p, I):
        "Housing good demand function."
        alpha, tau_h = self.alpha, self.tau_h
        num = (alpha * I) 
        denom = p + tau_h
        return num / denom 
    
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
    