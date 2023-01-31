
class CobbDouglasProduction:
    def __init__(self,
                 beta, #share parameter
                 A #productivity parameter
                 ):
        self.beta = beta 
        self.A = A 

    def F(self, N, L):
        return self.A * N**self.beta * L**(1-self.beta)

    def F_N(self, N, L):
        return self.A * self.beta * N**(self.beta-1) * L**(1-self.beta)
    
    def F_L(self, N, L):
        return self.A * (1-self.beta) * N**self.beta * L**(-self.beta)
    
    def MRTS_NL(self, N, L):
        return self.F_N(N, L) / self.F_L(N, L) 
    