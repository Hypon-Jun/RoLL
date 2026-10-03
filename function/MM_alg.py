########################################
#MM算法求解问题。 
########################################

import numpy as np
from scipy.linalg import svd, norm
from scipy.special import expit  

def prox_scad_vector(s, lam, rho, a=3.7):
    """
    求解 1/rho * SCAD 的近端算子。
    目标函数: min 0.5 * ||x - s||^2 + (1/rho) * SCAD(x)
    """
    res = np.zeros_like(s)
    abs_s = np.abs(s)


    thresh = lam / rho


    boundary_1 = lam * (1 + 1.0 / rho)

    boundary_2 = a * lam

    
    mask1 = abs_s <= boundary_1

    res[mask1] = np.sign(s[mask1]) * np.maximum(abs_s[mask1] - thresh, 0)

    
    mask2 = (abs_s > boundary_1) & (abs_s <= boundary_2)

    
    numerator = (a - 1) * s[mask2] - np.sign(s[mask2]) * a * thresh
    denominator = a - 1 - 1.0 / rho

    
    if denominator <= 1e-8:
        res[mask2] = s[mask2]
    else:
        res[mask2] = numerator / denominator

    
    mask3 = abs_s > boundary_2
    res[mask3] = s[mask3]

    return res

def prox_B_scad(A, lamb, rho, a=3.7):
    
    U, s, Vt = svd(A, full_matrices=False)

    s_new = prox_scad_vector(s, lamb, rho, a)

    return U @ np.diag(s_new) @ Vt

def prox_group_scad(A, threshold, rho, a=3.7):
    
    
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)

    
    new_norms = np.zeros_like(row_norms)

    
    eff_threshold = threshold / rho

    
    boundary_1 = threshold * (1 + 1.0 / rho)

    
    boundary_2 = a * threshold

    
    mask_soft = (row_norms > eff_threshold) & (row_norms <= boundary_1)
    new_norms[mask_soft] = row_norms[mask_soft] - eff_threshold

    
    mask_inter = (row_norms > boundary_1) & (row_norms <= boundary_2)

    if np.any(mask_inter):
        denominator = a - 1 - 1.0 / rho

        
        numerator = (a - 1) * row_norms[mask_inter] - a * eff_threshold

        
        if denominator <= 1e-8:
            new_norms[mask_inter] = row_norms[mask_inter]  # Fallback
        else:
            new_norms[mask_inter] = numerator / denominator

    
    mask_large = row_norms > boundary_2
    new_norms[mask_large] = row_norms[mask_large]

    
    scale = np.zeros_like(row_norms)

    nonzero_mask = row_norms > 0
    scale[nonzero_mask] = new_norms[nonzero_mask] / row_norms[nonzero_mask]

    return A * scale

def scad_penalty_term(x, lambda_val, a=3.7):
    
    abs_x = np.abs(x)

    values = np.zeros_like(abs_x)

    idx1 = abs_x <= lambda_val
    values[idx1] = lambda_val * abs_x[idx1]

    idx2 = (abs_x > lambda_val) & (abs_x <= a * lambda_val)
    if np.any(idx2):
        values[idx2] = (2 * a * lambda_val * abs_x[idx2] - abs_x[idx2] ** 2 - lambda_val ** 2) / (2 * (a - 1))

    idx3 = abs_x > a * lambda_val
    values[idx3] = (a + 1) * lambda_val ** 2 / 2


    return np.sum(values)

def scad_B(A, lambda_val, a=3.7):
    
    singular_values = np.linalg.svd(A, compute_uv=False)


    return scad_penalty_term(singular_values, lambda_val, a)

def scad_C(A, lambda_val, a=3.7):
    
    row_norms = np.linalg.norm(A, axis=1)

    
    return scad_penalty_term(row_norms, lambda_val, a)

def plogis(x):
    return 1 / (1 + np.exp(-x))

def loss_function(X, Y, B, C, lamb_B, lamb_C):
    eta = X @ B + C
    prob = np.clip(plogis(eta), 1e-10, 1 - 1e-10)
    Y_bin = (Y == 1).astype(float)
    loss = - np.sum(Y_bin * np.log(prob) + (1 - Y_bin) * np.log(1 - prob))
    return loss + scad_B(B, lamb_B) + scad_C(C, lamb_C)

def MM_alg(X, Y, lambda_B, lambda_C, B_init, C_init, max_iter=100000, tol=1e-7):

    X = np.array(X, dtype=float)


    n, p = X.shape
    q = Y.shape[1]

    lambda_B = lambda_B
    lambda_C = lambda_C


    
    X_aug = np.hstack([X, np.eye(n)])

    
    s_aug = svd(X_aug, full_matrices=False, compute_uv=False)
    max_singular_value = s_aug[0]

    max_eigen_svd = max_singular_value ** 2
    rho = 0.25 * max_eigen_svd

    B = B_init
    C = C_init


    for i in range(max_iter):
        # print(i)
        # print(loss_function(X, Y, B, C, lambda_B, lambda_C))
        B_old = B.copy()
        C_old = C.copy()

        # 计算预测值和残差
        eta = X @ B + C
        prob = expit(eta)
        resid = prob - Y

        grad_B = X.T @ resid
        grad_C = resid

        # Update B
        B_tilde = B.copy() - grad_B / rho
        B = prox_B_scad(B_tilde, lambda_B, rho)

        # Update C
        C_tilde = C.copy() - grad_C / rho
        C = prox_group_scad(C_tilde, lambda_C, rho)

        # 检查收敛性
        diff_B = np.linalg.norm(B - B_old) / (p * q)
        diff_C = np.linalg.norm(C - C_old) / (n * q)




        if diff_B < tol and diff_C < tol:
        #if (loss_function_scad(X, Y, B, C, lambda_B, lambda_C) - loss_function_scad(X, Y, B_old, C_old, lambda_B, lambda_C))/loss_function_scad(X, Y, B_old, C_old, lambda_B, lambda_C) < tol:
            print(f"Converged at iteration {i + 1}")
            break


    # return {
    #     "B": B,
    #     "C": C
    # }
    return B, C