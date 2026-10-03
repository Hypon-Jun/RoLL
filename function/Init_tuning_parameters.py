########################################
#计算使得最优解 B 和 C 全为 0 的最小 Lambda 值。
########################################

import numpy as np
from sklearn.linear_model import LogisticRegression

def get_zero_threshold_lambdas(X, Y, fit_intercept=False):

    Y_float = Y.astype(float)

 
    if fit_intercept:
        p_null = np.mean(Y_float, axis=0)
    else:
        p_null = 0.5

    residual_null = p_null - Y_float


    grad_B_full = X.T @ residual_null


    if fit_intercept:
        grad_B_to_check = grad_B_full[:-1, :]
    else:
        grad_B_to_check = grad_B_full


    lambda_B_max = np.linalg.norm(grad_B_to_check, ord=2)


    grad_C = residual_null


    row_norms = np.linalg.norm(grad_C, axis=1)
    lambda_C_max = np.max(row_norms)

    return lambda_B_max * 1.1, lambda_C_max * 1.1



def get_zero_threshold_lambdas_B(X, Y, fit_intercept=False):
   
    Y_float = Y.astype(float)

    if fit_intercept:
        p_null = np.mean(Y_float, axis=0)
    else:

        p_null = 0.5

    residual_null = p_null - Y_float


    grad_B_full = X.T @ residual_null


    if fit_intercept:
        grad_to_check = grad_B_full[:-1, :]
    else:
        grad_to_check = grad_B_full


    lambda_B_max = np.linalg.norm(grad_to_check, ord=2)

    return lambda_B_max * 1.1


def get_zero_threshold_lambdas_C(X, Y):
    
    n, m = Y.shape

    P_pred = np.zeros((n, m))

    for k in range(m):
        y_k = Y[:, k]
        clf = LogisticRegression(penalty='none', fit_intercept=True, solver='lbfgs', max_iter=10000)
        clf.fit(X, y_k)
        P_pred[:, k] = clf.predict_proba(X)[:, 1]


    residual_after_B = P_pred - Y.astype(float)

    grad_C = residual_after_B

    row_norms = np.linalg.norm(grad_C, axis=1)

    lambda_C_max = np.max(row_norms)

    return lambda_C_max

