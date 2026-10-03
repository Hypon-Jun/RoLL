import numpy as np
from scipy.linalg import svd
import pandas as pd
 

def plogis(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def estimate_rank(M, tol=1e-3):
    if np.all(M == 0): return 0
    s = svd(M, compute_uv=False)
    return np.sum(s > tol)


def calculate_sparsity(C, threshold=1e-4):
    
    row_norms = np.linalg.norm(C, axis=1)

    J = np.sum(row_norms > threshold)

    return int(J)

def calculate_PIC(X, Y, B, C, A_DF = 1, A_IF = 1, fit_intercept=False):
    n, m = C.shape
    p, _ = B.shape
    eta = X @ B + C
    prob = np.clip(plogis(eta), 1e-10, 1 - 1e-10)
    Y_bin = (Y == 1).astype(float)
    loss = - np.sum(Y_bin * np.log(prob) + (1 - Y_bin) * np.log(1 - prob))

    
    if fit_intercept:
        B_core = B[:-1, :]
        X_core = X[:, :-1]
    else:
        B_core = B
        X_core = X

   
    r = estimate_rank(B_core)
    q = estimate_rank(X_core)
    J = calculate_sparsity(C)

    IF = 0 if J == 0 else J * (np.log(np.exp(1) * n / J))
    DF = (q + m - r) * r + J * m

    return loss + A_DF * DF + A_IF * IF


def select_best_model_PIC(X, Y, candidates, A_DF=1, A_IF=1, fit_intercept=False):
    
    results = []


    #print(f"开始评估 {len(candidates)} 个候选模型...")

    for i, model in enumerate(candidates):
        B_curr = model['B']
        C_curr = model['C']
        lam_B = model['lambda_B']
        lam_C = model['lambda_C']

        
        pic_score = calculate_PIC(X, Y, B_curr, C_curr, A_DF, A_IF, fit_intercept=fit_intercept)

        
        if fit_intercept:

            B_for_rank = B_curr[:-1, :]
        else:

            B_for_rank = B_curr

        rank_B = estimate_rank(B_for_rank)

        J = calculate_sparsity(C_curr)

        results.append({
            'lambda_B': lam_B,
            'lambda_C': lam_C,
            'PIC': pic_score,
            'rank_B': rank_B,
            'sparsity_C': J,
            'index': i  # 记录原始索引
        })

    # 3. 转换为 DataFrame
    results_df = pd.DataFrame(results)

    
    results_df = results_df.sort_values(
        by=['PIC', 'rank_B', 'sparsity_C'],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    best_idx = int(results_df.loc[0, 'index'])
    best_model = candidates[best_idx]

    #print(f"评估完成。最优 PIC: {results_df.loc[0, 'PIC']:.4f}")

    return best_model, results_df