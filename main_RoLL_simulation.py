import numpy as np
import scipy.io
import os
import pandas as pd
import time
 
 
from tuning_grid import get_lambda_grids
from acc_alg import acc_robust_low_rank_learning
from PIC import select_best_model_PIC
from EvaluationAll import evaluation_all
from MM_alg import MM_alg


def plogis(x):
    return 1 / (1 + np.exp(-x))


def run_model_selection_pipeline(X, Y, B_init, C_init, fit_intercept=False):
    
    grid_B, grid_C = get_lambda_grids(X, Y, fit_intercept=fit_intercept)

    #grid_B = np.array([121.202615])
    #grid_C = np.array([1.791430])



    candidates = []

    n, p = X.shape
    _, m = Y.shape

    # grid_B = np.logspace(np.log10(0.05), np.log10(0.10), 10) * n
    # grid_C = np.logspace(np.log10(0.001), np.log10(0.01), 10) * n

    #5percent
    # grid_B = np.array([6])
    # grid_C = np.array([1.79143])

    # 6percent
    # grid_B = np.array([6])
    # grid_C = np.array([1.89143])

    # 7percent
    # grid_B = np.array([6])
    # grid_C = np.array([2.19143])

    # 8percent
    # grid_B = np.array([6])
    # grid_C = np.array([2.29143])

    # 9percent
    # grid_B = np.array([6])
    # grid_C = np.array([2.49143])

    # 10percent
    grid_B = np.array([6])
    grid_C = np.array([2.79143])

    B_init = B_init
    C_init = C_init

    
    for lb in grid_B:
        for lc in grid_C:
            B_hat, C_hat = acc_robust_low_rank_learning(X, Y, lb, lc, B_init, C_init)

            #B_hat, C_hat = MM_alg(X, Y, lb, lc, B_init, C_init)

            candidates.append({
                'lambda_B': lb,
                'lambda_C': lc,
                'B': B_hat,
                'C': C_hat
            })

    
    best_model, df_results = select_best_model_PIC(X, Y, candidates, A_DF=1, A_IF=1)
    return best_model, df_results


if __name__ == "__main__":
    
    np.random.seed(42)

    n_repeats = 20


    metric_names = [
        "Hamming Loss",
        "Example Accuracy", "Example Precision", "Example Recall", "Example F1",
        "Subset Accuracy",
        "Label Accuracy", "Label Precision", "Label Recall", "Label F1",
        "Micro F1",
        "Average Precision", "One Error", "Ranking Loss", "Coverage"
    ]


    all_results = np.zeros((n_repeats, len(metric_names)))


    mse_results = []

    times = np.zeros(n_repeats)

    print(f"开始运行 {n_repeats} 次循环实验...")

    for i in range(n_repeats):
        run_idx = i + 1
        print(f"\n{'=' * 25} Run {run_idx} / {n_repeats} {'=' * 25}")


        train_file = f"./10per/simulation_train_{run_idx}.mat"
        test_file = f"./10per/simulation_test_{run_idx}.mat"

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[Warning] 文件不存在: {train_file} 或 {test_file}. 跳过此轮.")
            continue


        print(f"Loading {train_file} and {test_file}...")
        try:
            train_mat = scipy.io.loadmat(train_file)
            test_mat = scipy.io.loadmat(test_file)
        except Exception as e:
            print(f"[Error] 读取文件失败: {e}")
            continue

        train_data = train_mat['data']
        train_target = train_mat['target']
        test_data = test_mat['data']
        test_target = test_mat['target']
        B0 = train_mat['B']
        C0 = train_mat['C']


        B_init = B0
        row_is_outlier = np.sum(np.abs(C0), axis=1) > 1e-6

        C_init = np.zeros_like(C0)

        C_init[row_is_outlier, :] = 10



        N_train = train_data.shape[0]
        if train_target.shape[0] != N_train and train_target.shape[1] == N_train:
            print("  -> Transposing train_target to (Samples, Labels)")
            train_target = train_target.T

        N_test = test_data.shape[0]
        if test_target.shape[0] != N_test and test_target.shape[1] == N_test:
            # print("  -> Transposing test_target to (Samples, Labels)")
            test_target = test_target.T


        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(f"Feature dimension mismatch: Train {train_data.shape[1]}, Test {test_data.shape[1]}")


        start_time = time.time()
        print(f"  -> Training (Data shape: {train_data.shape}, Target shape: {train_target.shape})...")
        best_model, res_df_1 = run_model_selection_pipeline(train_data, train_target, B_init, C_init, fit_intercept=False)
        end_time = time.time()
        total_time = end_time - start_time

        times[i] = total_time


        print("\n查看前 5 个模型的结果:")
        print(res_df_1.head())


        B_hat = best_model['B']

        pred_scores = test_data @ B_hat
        pred_probs = plogis(pred_scores)
        pred_labels = (pred_probs >= 0.5).astype(int)


        if B_hat.shape == B0.shape:
            mse_val = np.mean((B_hat - B0) ** 2)
        elif B_hat.shape[0] == B0.shape[0] + 1:

            mse_val = np.mean((B_hat[:-1, :] - B0) ** 2)
        else:
            print(f"[Warning] Shape mismatch for MSE: B_hat {B_hat.shape}, B0 {B0.shape}")
            mse_val = np.mean((B_hat - B0) ** 2)  

        mse_results.append(mse_val)


        results_list = evaluation_all(
            pre_labels=pred_labels.T,
            outputs=pred_probs.T,
            test_target=test_target.T
        )


        all_results[i, :] = results_list


        print(
            f"  -> Result: Hamming Loss={results_list[0]:.4f}, Ranking Loss={results_list[13]:.4f}, AP={results_list[11]:.4f}")


    print("\n" + "#" * 50)
    print(f"Final Statistics ({n_repeats} Runs)")
    print("#" * 50)


    valid_rows = ~np.all(all_results == 0, axis=1)
    if np.sum(valid_rows) == 0:
        print("No valid results collected.")
    else:
        final_results = all_results[valid_rows]
        mean_vals = np.mean(final_results, axis=0)
        std_vals = np.std(final_results, axis=0)


        mse_mean = np.mean(mse_results)
        mse_std = np.std(mse_results)

        print(f"{'Metric':<25} | {'Mean':<10} | {'Std':<10}")
        print("-" * 50)
        for name, m, s in zip(metric_names, mean_vals, std_vals):
            print(f"{name:<25} | {m:.4f}     | {s:.4f}")


        print("-" * 50)
        print(f"{'MSE (B-B*)':<25} | {mse_mean:.6f}     | {mse_std:.6f}")


        output_file = "final_results_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Final Statistics ({np.sum(valid_rows)} Valid Runs)\n")
            f.write("#" * 50 + "\n")
            f.write(f"{'Metric':<25} | {'Mean':<10} | {'Std':<10}\n")
            f.write("-" * 50 + "\n")
            for name, m, s in zip(metric_names, mean_vals, std_vals):
                f.write(f"{name:<25} | {m:.6f} | {s:.6f}\n")

 
            f.write("-" * 50 + "\n")
            f.write(f"{'MSE (B-B*)':<25} | {mse_mean:.6f} | {mse_std:.6f}\n")

        print(f"\nResults have been saved to: {output_file}")

    print('avg_time', np.mean(times))
    print('std_time', np.std(times))