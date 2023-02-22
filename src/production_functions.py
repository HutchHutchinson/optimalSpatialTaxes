
class CobbDouglasProduction:
    def __init__(self,                
                 A, #productivity parameter,
                 beta #share parameter
                 ):
        self.A, self.beta = A, beta 

    def F(self, N, L):
        "Output."
        A, beta = self.A, self.beta
        return A * N**beta * L**(1-beta) 

    def F_N(self, N, L):
        "Marginal product of labor."
        A, beta = self.A, self.beta
        return A * beta * N**(beta-1) * L**(1-beta) 
    
    def F_L(self, N, L):
        "Marginal product of land."
        A, beta = self.A, self.beta
        return A * (1-self.beta) * N**beta * L**(-beta) 
    
    def MRTS_NL(self, N, L):
        "Marginal rate of technical substitution between labor and land."
        return self.F_N(N, L) / self.F_L(N, L) 
    
    def N_c(self, N, L, L_c):
        return (N*L_c) / L 
    