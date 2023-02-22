
def create_utility_list(J, alpha, xi, utility_class):
    utility_list = []
    for i in range(J):
        preferences = utility_class(alpha, xi[i])
        utility_list.append({'u': preferences.u,
                            'v': preferences.v,
                            'c': preferences.c,
                            'h': preferences.h,
                            'mu_c': preferences.mu_c,
                            'mu_h': preferences.mu_h,
                            'MRS_hc': preferences.MRS_hc
                            }) 
    return utility_list 

def create_prod_func_list(J, A, beta, prod_func_class):
    prod_func_list = []
    for i in range(J):
        technology = prod_func_class(A[i], beta) 
        prod_func_list.append({'F': technology.F,
                               'F_N': technology.F_N,
                               'F_L': technology.F_L,
                               'MRTS_NL': technology.MRTS_NL,
                               'N_c': technology.N_c
                              }) 
    return prod_func_list 
