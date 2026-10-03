from Init_tuning_parameters import *

def get_lambda_grids(X, Y, n_lambdas=5, min_ratio=0.05, fit_intercept = False):
    
    lambda_B_max, lambda_C_max = get_zero_threshold_lambdas(X, Y, fit_intercept= fit_intercept)
    
    lambdas_B = np.logspace(
        np.log10(lambda_B_max * 0.1),
        np.log10(lambda_B_max * 0.5),
        n_lambdas 
    )

    lambdas_B = lambdas_B[::-1]


    lambdas_C = np.logspace(
        np.log10(lambda_C_max * 0.8),
        np.log10(lambda_C_max * 1.03),
        n_lambdas
    )
    lambdas_C = lambdas_C[::-1]

    return lambdas_B, lambdas_C


def get_lambda_B_grids(X, Y, n_lambdas=10, min_ratio=0.01, fit_intercept = False):
    lambda_B_max = get_zero_threshold_lambdas_B(X, Y, fit_intercept= fit_intercept)
    lambdas_B = np.logspace(
        np.log10(lambda_B_max * min_ratio),
        np.log10(lambda_B_max),
        n_lambdas
    )

    lambdas_B = lambdas_B[::-1]


    return lambdas_B

def get_lambda_C_grids(X, Y, n_lambdas=10, min_ratio=0.5):
    lambda_C_max = get_zero_threshold_lambdas_C(X, Y)
    lambdas_C = np.logspace(
        np.log10(lambda_C_max * min_ratio),
        np.log10(lambda_C_max),
        n_lambdas
    )

    lambdas_C = lambdas_C[::-1]


    return lambdas_C