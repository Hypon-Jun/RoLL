import numpy as np
from scipy.linalg import svd, norm

# ==================== I. 基础算子 ====================

def plogis(x):
    return 1 / (1 + np.exp(-x))


def prox_nuclear(A, threshold):

    U, s, Vt = svd(A, full_matrices=False)

    s_new = np.maximum(s - threshold, 0)
    return U @ np.diag(s_new) @ Vt

def prox_l1_vector(s, lam, rho): 

    thresh = lam / rho
    return np.sign(s) * np.maximum(np.abs(s) - thresh, 0.0)

def prox_B_l1(A, lamb, rho):

    U, s, Vt = svd(A, full_matrices=False)
    s_new = prox_l1_vector(s, lamb, rho)
    return U @ np.diag(s_new) @ Vt


def prox_scad_vector(s, lam, rho, a=3.7):

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


def prox_mcp_vector(s, lam, rho, a=3.0):

    res = np.zeros_like(s)
    abs_s = np.abs(s)


    eff_threshold = lam / rho


    mask_inter = (abs_s > eff_threshold) & (abs_s <= a * lam)

    if np.any(mask_inter):


        numerator = a * (rho * abs_s[mask_inter] - lam)
        denominator = a * rho - 1.0


        if denominator <= 1e-8:

            res[mask_inter] = s[mask_inter]
        else:
            res[mask_inter] = np.sign(s[mask_inter]) * numerator / denominator


    mask_large = abs_s > a * lam
    res[mask_large] = s[mask_large]

    return res


def prox_B_mcp(A, lamb, rho, a=3.0):

    U, s, Vt = svd(A, full_matrices=False)


    s_new = prox_mcp_vector(s, lamb, rho, a)


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

def prox_group_l1(A, threshold, rho):


    row_norms = np.linalg.norm(A, axis=1, keepdims=True)


    eff_threshold = threshold / rho


    scale = np.maximum(1.0 - eff_threshold / (row_norms + 1e-12), 0.0)


    return A * scale

def prox_l20(A, q):
    
    n_rows = A.shape[0]


    if q >= n_rows:
        return A.copy()
    if q <= 0:
        return np.zeros_like(A)


    row_norms = np.linalg.norm(A, axis=1)


    top_q_indices = np.argpartition(-row_norms, q)[:q]


    X = np.zeros_like(A)
    X[top_q_indices] = A[top_q_indices]

    return X

def estimate_rank(M, tol=1e-3):
    if np.all(M == 0): return 0
    s = svd(M, compute_uv=False)
    return np.sum(s > tol)


def prox_group_mcp(A, threshold, rho, a=3.0):
    
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)


    new_norms = np.zeros_like(row_norms)


    eff_threshold = threshold / rho


    mask_firm = (row_norms > eff_threshold) & (row_norms <= a * threshold)

    if np.any(mask_firm):


        numerator = a * (rho * row_norms[mask_firm] - threshold)
        denominator = a * rho - 1.0


        if denominator <= 1e-8:

            new_norms[mask_firm] = row_norms[mask_firm]
        else:
            new_norms[mask_firm] = numerator / denominator


    mask_large = row_norms > a * threshold
    new_norms[mask_large] = row_norms[mask_large]


    scale = np.zeros_like(row_norms)

    nonzero_mask = row_norms > 0
    scale[nonzero_mask] = new_norms[nonzero_mask] / row_norms[nonzero_mask]

    return A * scale

def prox_l21(A, threshold):

    row_norms = np.linalg.norm(A, axis=1, keepdims=True)

    row_norms[row_norms == 0] = 1e-10


    shrinkage = np.maximum(0, 1 - threshold / row_norms)

    return A * shrinkage


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


def scad_nuclear(A, lambda_val, a=3.7):
    
    singular_values = np.linalg.svd(A, compute_uv=False)


    return scad_penalty_term(singular_values, lambda_val, a)


def scad_derivative(x, lambda_val, a=3.7):
    
    x = np.abs(x)

    grad = np.zeros_like(x)


    idx1 = x <= lambda_val
    grad[idx1] = lambda_val

 
    idx2 = (x > lambda_val) & (x <= a * lambda_val)
    if np.any(idx2):
        grad[idx2] = (a * lambda_val - x[idx2]) / (a - 1)


    idx3 = x > a * lambda_val
    grad[idx3] = 0

    return grad


def scad_nuclear_gradient(A, lambda_val, a=3.7):
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)


    ds = scad_derivative(s, lambda_val, a)

    

    Grad = (U * ds[None, :]) @ Vt

    return Grad


def scad_l21(A, lambda_val, a=3.7):
    
    row_norms = np.linalg.norm(A, axis=1)

    return scad_penalty_term(row_norms, lambda_val, a)


def scad_l21_gradient(A, lambda_val, a=3.7):
    
    
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)

    
    derivative_vals = scad_derivative(row_norms, lambda_val, a)

    
    grad = np.zeros_like(A)

    
    non_zero_idx = (row_norms.flatten() > 0)

    if np.any(non_zero_idx):
        r = row_norms[non_zero_idx]
        d = derivative_vals[non_zero_idx]


        scale = d / r  # shape (k, 1)

   
        grad[non_zero_idx] = scale * A[non_zero_idx]



    return grad

def nuclear_norm(A):
   
    return np.sum(np.linalg.svd(A, compute_uv=False))

def l21_norm(A):
    return np.sum(np.linalg.norm(A, axis=1))

# ==================== II. 加速所需的函数 ====================
def loss_function_without_penalty(X, Y, B, C):
    eta = X @ B + C
    prob = np.clip(plogis(eta), 1e-10, 1 - 1e-10)
    Y_bin = (Y == 1).astype(float)
    loss = - np.sum(Y_bin * np.log(prob) + (1 - Y_bin) * np.log(1 - prob))
    return loss


def loss_function(X, Y, B, C, lamb_B, lamb_C):
    eta = X @ B + C
    prob = np.clip(plogis(eta), 1e-10, 1 - 1e-10)
    Y_bin = (Y == 1).astype(float)
    loss = - np.sum(Y_bin * np.log(prob) + (1 - Y_bin) * np.log(1 - prob))
    #return loss + lamb_B * nuclear_norm(B) + lamb_C * l21_norm(C)
    return loss + scad_nuclear(B, lamb_B) + scad_l21(C, lamb_C)
    #return loss + lamb_B * nuclear_norm(B) + scad_l21(C, lamb_C)

def create_objective_function(X, Y):
    def objective_function(B, C):
        eta = X @ B + C
        prob = np.clip(plogis(eta), 1e-10, 1 - 1e-10)
        Y_bin = (Y == 1).astype(float)
        loss = - np.sum(Y_bin * np.log(prob) + (1 - Y_bin) * np.log(1 - prob))
        return loss
    return objective_function

def grad_of_B(X, Y, B, C):
    resid = plogis(X @ B + C) - Y
    return X.T @ resid

def create_grad_B(X, Y, C):
    def grad_B(B):
        resid = plogis(X @ B + C) - Y
        grad = X.T @ resid
        return grad
    return grad_B

def grad_of_C(X, Y, B, C):
    resid = plogis(X @ B + C) - Y
    return X.T @ resid

def create_grad_C(X, Y, B):
    def grad_C(C):
        resid = plogis(X @ B + C) - Y
        grad = resid
        return grad
    return grad_C

def inverse_learning_rate(X):
    n, _ = X.shape
    X_aug = np.hstack([X, np.eye(n)])
    s_aug = svd(X_aug, full_matrices=False, compute_uv=False)
    max_singular_value = s_aug[0]

    max_eigen_svd = max_singular_value ** 2 / 4
    return max_eigen_svd

def lower_mu(X):
    n, _ = X.shape
    X_aug = np.hstack([X, np.eye(n)])
    s_aug = svd(X_aug, full_matrices=False, compute_uv=False)
    min_singular_value = s_aug[-1]

    min_eigen_svd = min_singular_value ** 2 / 4
    return min_eigen_svd

def theta_update(theta_prev, mu_prev, rho_prev, rho):
    prev = theta_prev * (rho_prev * theta_prev + mu_prev)
    return (- prev + np.sqrt(prev**2 + 4 * rho * prev)) / (2 * rho)

def gamma_update(beta, beta_prev, theta, theta_prev, rho_prev, mu_prev):
    gamma = beta + rho_prev * theta * (1 - theta_prev) / (rho_prev * theta_prev + mu_prev) * (beta - beta_prev)
    return gamma

def beta_update(gamma, rho, mu, grad_function, prox_function, lamb):
    beta = prox_function(gamma - grad_function(gamma) / (rho + mu), lamb, (rho + mu))
    return beta

#加速line search所需函数
def breg_loss(parameter1B, parameter1C, parameter2B, parameter2C, loss, grad_B, grad_C):
    return loss(parameter1B, parameter1C) - loss(parameter2B, parameter2C) - np.trace(grad_B(parameter2B).T @ (parameter1B - parameter2B))\
    - np.trace(grad_C(parameter2C).T @ (parameter1C - parameter2C))

def breg_l2(parameter1B, parameter1C, parameter2B, parameter2C):
    return np.sum((parameter1B - parameter2B) ** 2) / 2 + np.sum((parameter1C - parameter2C) ** 2) / 2

def breg_psi(parameter1B, parameter1C, parameter2B, parameter2C, mu, loss, grad_B, grad_C):
    return breg_loss(parameter1B, parameter1C, parameter2B, parameter2C, loss, grad_B, grad_C) - mu * breg_l2(
        parameter1B, parameter1C, parameter2B, parameter2C)


def penalty_function(B, C, lamb_B, lamb_C):
    return scad_nuclear(B, lamb_B) + scad_l21(C, lamb_C)


def C_penalty_function(parameter1B, parameter1C, parameter2B, parameter2C, lamb_B, lamb_C, theta):
    return theta * penalty_function(parameter1B, parameter1C, lamb_B, lamb_C) + (1 - theta) * penalty_function(parameter2B, parameter2C, lamb_B, lamb_C) - penalty_function(theta * parameter1B + (1 - theta) * parameter2B, theta * parameter1C + (1 - theta) * parameter2C, lamb_B, lamb_C)

def breg_penalty(parameter1B, parameter1C, parameter2B, parameter2C, lamb_B, lamb_C):
    return penalty_function(parameter1B, parameter1C, lamb_B, lamb_C) - penalty_function(parameter2B, parameter2C, lamb_B, lamb_C)\
    - np.trace(scad_nuclear_gradient(parameter2B, lamb_B).T @ (parameter1B - parameter2B)) - np.trace(scad_l21_gradient(parameter2C, lamb_C).T @ (parameter1C - parameter2C))


def R_function(gamma_B, gamma_C, beta_B, beta_C, beta_new_B, beta_new_C, theta, mu, rho, loss, grad_B, grad_C):
    return rho * breg_l2(beta_new_B, beta_new_C, gamma_B, gamma_C) - breg_psi(beta_new_B, beta_new_C, gamma_B, gamma_C, mu, loss, grad_B, grad_C) + \
           (1 - theta) * breg_psi(beta_B, beta_C, gamma_B, gamma_C, mu, loss, grad_B, grad_C)

def E_function(optimal_beta_B, optimal_beta_C, gamma_B, gamma_C, beta_B, beta_C, beta_new_B, beta_new_C, theta, mu, loss, grad_B, grad_C, lamb_B, lamb_C):
    return breg_psi(optimal_beta_B, optimal_beta_C, gamma_B, gamma_C, mu, loss, grad_B, grad_C) + C_penalty_function(optimal_beta_B, optimal_beta_C, beta_B, beta_C, lamb_B, lamb_C, theta) / theta \
    + breg_penalty(theta * optimal_beta_B + (1 - theta) * beta_B, theta * optimal_beta_C + (1 - theta) * beta_C, beta_new_B, beta_new_C, lamb_B, lamb_C) / theta

def line_search_mu(
    X, Y, optimal_beta_B, optimal_beta_C,
    theta, mu, rho, L, mu_try,
    beta_B, beta_C, beta_old_B, beta_old_C,
    loss_function, lamb_B, lamb_C,
    con=3.4, search_iterations=3
):
    """
    Line search over mu candidates.

    W_max is computed over all domain-valid candidates.
    R_try >= 0 is imposed only at the final selection stage.
    """

    U_values = []
    mu_values = []
    R_values = []

   
    best_mu = 0.0

    for search_iter in range(search_iterations):

        
        if (
            not np.isfinite(mu_try)
            or mu_try < 0.0
            or mu_try >= L
        ):
            break

        rho_try = L - mu_try

        if not np.isfinite(rho_try) or rho_try <= 0.0:
            break

        

        theta_try = theta_update(
            theta, mu, rho, rho_try
        )

        gamma_B_try = gamma_update(
            beta_B, beta_old_B,
            theta_try, theta, rho, mu
        )
        gamma_C_try = gamma_update(
            beta_C, beta_old_C,
            theta_try, theta, rho, mu
        )

        grad_B = create_grad_B(
            X, Y, gamma_C_try
        )
        grad_C = create_grad_C(
            X, Y, gamma_B_try
        )

        beta_B_try = beta_update(
            gamma_B_try,
            rho_try,
            mu_try,
            grad_B,
            prox_B_scad,
            lamb_B
        )

        beta_C_try = beta_update(
            gamma_C_try,
            rho_try,
            mu_try,
            grad_C,
            prox_group_scad,
            lamb_C
        )

        E_try = E_function(
            optimal_beta_B,
            optimal_beta_C,
            gamma_B_try,
            gamma_C_try,
            beta_B,
            beta_C,
            beta_B_try,
            beta_C_try,
            theta_try,
            mu_try,
            loss_function,
            grad_B,
            grad_C,
            lamb_B,
            lamb_C
        )

        R_try = R_function(
            gamma_B_try,
            gamma_C_try,
            beta_B,
            beta_C,
            beta_B_try,
            beta_C_try,
            theta_try,
            mu_try,
            rho_try,
            loss_function,
            grad_B,
            grad_C
        )

        
        if (
            not np.isfinite(R_try)
            or not np.isfinite(E_try)
        ):
            mu_try *= con
            continue

        RE_try = R_try + theta_try * E_try

        if not np.isfinite(RE_try):
            mu_try *= con
            continue

        
        U_values.append(RE_try)
        R_values.append(R_try)
        mu_values.append(mu_try)

        mu_try *= con

    
    if len(U_values) == 0:
        return 0.0

    U_array = np.asarray(U_values)
    R_array = np.asarray(R_values)
    mu_array = np.asarray(mu_values)

    
    max_value = np.max(U_array)

    if max_value > 0:
        lower_bound = (1.0 - 1e-3) * max_value
        upper_bound = max_value

        eligible = np.where(
            (R_array >= 0.0)
            & (U_array >= lower_bound)
            & (U_array <= upper_bound)
        )[0]
    else:
        eligible = np.where(
            (R_array >= 0.0)
            & np.isclose(U_array, max_value)
        )[0]

    
    if len(eligible) == 0:
        return 0.0

    
    best_index = eligible[-1]

    return mu_array[best_index]

def line_search(
    X, Y, optimal_beta_B, optimal_beta_C,
    theta, mu, rho, L,
    beta_B, beta_C, beta_old_B, beta_old_C,
    loss_function, lamb_B, lamb_C
):

    mu_try = 1e-3 if mu <= 0.0 else mu * 0.4

    best_mu = line_search_mu(
        X, Y,
        optimal_beta_B, optimal_beta_C,
        theta, mu, rho, L, mu_try,
        beta_B, beta_C,
        beta_old_B, beta_old_C,
        loss_function,
        lamb_B, lamb_C,
        con=2.8,
        search_iterations=3
    )

    return best_mu



#主函数
def acc_robust_low_rank_learning(X, Y, lambda_B, lambda_C,B_init, C_init, max_iter=100000, tol=1e-6):
    #loss = create_objective_function(X, Y)
    n, p = X.shape
    _, m = Y.shape
    #lb_sum, lc_sum = lambda_B_avg * n, lambda_C_avg * n
    lb_sum, lc_sum = lambda_B, lambda_C
    mu = 0
    rho = inverse_learning_rate(X)
    #np.random.seed(0)
    #B_init, C_init = np.random.randn(p, m), np.random.randn(n, m)
    #B_init, C_init = np.zeros((p, m)), np.zeros((n, m))
    B_gamma = B_init.copy()
    B_beta = B_init.copy()
    C_gamma = C_init.copy()
    C_beta = C_init.copy()
    theta0 = 1
    theta_prev = theta0
    rho_prev = rho
    mu_prev = mu
    theta = theta0
    acc_iter = max_iter



    for iter in range(max_iter):
        # print(iter)
        # print(loss_function(X, Y, B_beta, C_beta, lb_sum, lc_sum))

        B_beta_tmp = B_beta.copy()
        C_beta_tmp = C_beta.copy()


        gradient_B = create_grad_B(X, Y, C_gamma)
        gradient_C = create_grad_C(X, Y, B_gamma)
        #这里可以换penalty对应的proximity operator
        B_beta = beta_update(B_gamma, rho, mu, gradient_B, prox_B_scad, lb_sum)
        C_beta = beta_update(C_gamma, rho, mu, gradient_C, prox_group_scad, lc_sum)



        diff_B = np.linalg.norm(B_beta - B_beta_tmp) / (p * m)
        diff_C = np.linalg.norm(C_beta - C_beta_tmp) / (n * m)


        if diff_B < tol and diff_C < tol:
            print(f"Converged at iteration {iter + 1}")
            print('function_value', loss_function(X, Y, B_beta, C_beta, lb_sum, lc_sum))
            break

        # if np.abs(loss(B_beta, C_beta) - loss(B_beta_tmp, C_beta_tmp)) < 1e-6:
        #     print(f"Converged at iteration {iter + 1}")
        #     break

        theta_prev = theta
        mu_prev = mu
        rho_prev = rho

        theta = theta_update(theta_prev, mu_prev, rho_prev, rho)



        B_gamma = gamma_update(B_beta, B_beta_tmp, theta, theta_prev, rho_prev, mu_prev)
        C_gamma = gamma_update(C_beta, C_beta_tmp, theta, theta_prev, rho_prev, mu_prev)

        #print(loss_function(X, Y, B_gamma, C_gamma, lb_sum, lc_sum))
    #return {"B": B_beta, "C": C_beta}
    return B_beta, C_beta

def acc_robust_low_rank_learning_with_line_search(
    X,
    Y,
    lambda_B,
    lambda_C,
    B_init,
    C_init,
    max_iter=100000,
    tol=1e-5,
    verbose=False
):
    
    loss = create_objective_function(X, Y)

    n, p = X.shape
    _, m = Y.shape

    lb_sum = lambda_B
    lc_sum = lambda_C

    
    mu0 = 0.0
    mu = mu0

    L = inverse_learning_rate(X)

    rho0 = L
    rho = rho0

    theta = 1.0

    
    T0 = 50

    
    search_interval = 10

    
    B_gamma = B_init.copy()
    B_beta = B_init.copy()

    C_gamma = C_init.copy()
    C_beta = C_init.copy()

    
    best_f = loss_function(
        X, Y,
        B_beta, C_beta,
        lb_sum, lc_sum
    )

    optimal_beta_B = B_beta.copy()
    optimal_beta_C = C_beta.copy()

    
    for iteration in range(max_iter):

        
        B_beta_tmp = B_beta.copy()
        C_beta_tmp = C_beta.copy()

        
        gradient_B = create_grad_B(
            X, Y, C_gamma
        )

        gradient_C = create_grad_C(
            X, Y, B_gamma
        )

        B_beta = beta_update(
            B_gamma,
            rho,
            mu,
            gradient_B,
            prox_B_scad,
            lb_sum
        )

        C_beta = beta_update(
            C_gamma,
            rho,
            mu,
            gradient_C,
            prox_group_scad,
            lc_sum
        )

       
        current_f = loss_function(
            X, Y,
            B_beta, C_beta,
            lb_sum, lc_sum
        )

        if current_f < best_f:
            best_f = current_f
            optimal_beta_B = B_beta.copy()
            optimal_beta_C = C_beta.copy()

        
        diff_B = (
            np.linalg.norm(
                B_beta - B_beta_tmp,
                ord="fro"
            )
            / max(
                1.0,
                np.linalg.norm(
                    B_beta_tmp,
                    ord="fro"
                )
            )
        )

        diff_C = (
            np.linalg.norm(
                C_beta - C_beta_tmp,
                ord="fro"
            )
            / max(
                1.0,
                np.linalg.norm(
                    C_beta_tmp,
                    ord="fro"
                )
            )
        )

        if verbose:
            print(
                f"iter={iteration}, "
                f"objective={current_f:.10e}, "
                f"mu={mu:.6e}, "
                f"rho={rho:.6e}, "
                f"diff_B={diff_B:.3e}, "
                f"diff_C={diff_C:.3e}"
            )

        if diff_B < tol and diff_C < tol:
            if verbose:
                print(
                    f"Converged at iteration {iteration + 1}"
                )
                print(
                    "best_function_value",
                    best_f
                )
            break

       
        theta_prev = theta
        mu_prev = mu
        rho_prev = rho

        
        next_iteration = iteration + 1

        
        if next_iteration < T0:

            mu = mu0
            rho = rho0

        
        elif (
            (next_iteration - T0) % search_interval == 0
        ):

            mu = line_search(
                X,
                Y,
                optimal_beta_B,
                optimal_beta_C,
                theta_prev,
                mu_prev,
                rho_prev,
                L,
                B_beta,
                C_beta,
                B_beta_tmp,
                C_beta_tmp,
                loss,
                lb_sum,
                lc_sum
            )

            
            rho = L - mu

        
        else:

            mu = mu_prev
            rho = rho_prev

        
        theta = theta_update(
            theta_prev,
            mu_prev,
            rho_prev,
            rho
        )

        
        B_gamma = gamma_update(
            B_beta,
            B_beta_tmp,
            theta,
            theta_prev,
            rho_prev,
            mu_prev
        )

        C_gamma = gamma_update(
            C_beta,
            C_beta_tmp,
            theta,
            theta_prev,
            rho_prev,
            mu_prev
        )

 
    return B_beta, C_beta


