
class CobbDouglasUtility:
    def __init__(self, 
                 alpha, #budget share of housing
                 xi):   #amenity vaulues of the location
        self.alpha = alpha
        self.xi = xi 
        
        self.v_cons = self.alpha**self.alpha * (1-self.alpha)**(1-self.alpha) 
    
    def u(self, c, h):
        return self.xi * c**(1-self.alpha) * h**self.alpha
    
    def v(self, p, I):
        return self.v_cons * self.xi * I * p**(-self.alpha)
    
    def c(self, I):
        return (1-self.alpha) * I
    
    def h(self, p, I):
        return (self.alpha * I) / p
    
    def mu_c(self, c, h):
        return (1-self.alpha) * self.xi * c**(-self.alpha) * h**self.alpha
    
    def mu_h(self, c, h):
        return self.alpha * self.xi * c**(1-self.alpha) * h**(self.alpha-1)
    
    def MRS_hc(self, c, h):
        return self.mu_h(c, h) / self.mu_c(c, h) 
    