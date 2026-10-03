import numpy as np
import scipy.io
import os
import pandas as pd
 
 
from tuning_grid import get_lambda_grids
from acc_alg import acc_robust_low_rank_learning
from acc_alg_real_data import acc_robust_low_rank_learning_real_data
from PIC import select_best_model_PIC
from EvaluationAll import evaluation_all


def plogis(x):
    return 1 / (1 + np.exp(-x))


def run_model_selection_pipeline(X, Y, fit_intercept=False):
    grid_B, grid_C = get_lambda_grids(X, Y, fit_intercept=fit_intercept)

    grid_B = np.array([50])

    candidates = []

    n, p = X.shape
    _, m = Y.shape

    B_init = np.zeros((p, m))
    C_init = np.zeros((n, m))


    for lb in grid_B:
        for lc in grid_C:
            B_hat, C_hat = acc_robust_low_rank_learning_real_data(X, Y, lb, lc, B_init, C_init)

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

    data_name = "emotions"


    metric_names = [
        "Hamming Loss",
        "Example Accuracy", "Example Precision", "Example Recall", "Example F1",
        "Subset Accuracy",
        "Label Accuracy", "Label Precision", "Label Recall", "Label F1",
        "Micro F1",
        "Average Precision", "One Error", "Ranking Loss", "Coverage"
    ]


    all_results = np.zeros((n_repeats, len(metric_names)))

    print(f"开始运行 {n_repeats} 次循环实验...")

    for i in range(n_repeats):
        run_idx = i + 1
        print(f"\n{'=' * 25} Run {run_idx} / {n_repeats} {'=' * 25}")

    
        train_file = f"./dataset/emotions_runs/emotions_train_{run_idx}.mat"
        test_file = f"./dataset/emotions_runs/emotions_test_{run_idx}.mat"

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

        
        N_train = train_data.shape[0]
        if train_target.shape[0] != N_train and train_target.shape[1] == N_train:
            print("  -> Transposing train_target to (Samples, Labels)")
            train_target = train_target.T

        N_test = test_data.shape[0]
        if test_target.shape[0] != N_test and test_target.shape[1] == N_test:
            # print("  -> Transposing test_target to (Samples, Labels)")
            test_target = test_target.T

        if np.var(train_data[:, -1]) < 1e-6:
            print("  -> Detected intercept (constant column) in Train Data. Removing it.")
            train_data = train_data[:, :-1]

        if np.var(test_data[:, -1]) < 1e-6:
            # print("  -> Detected intercept in Test Data. Removing it.")
            test_data = test_data[:, :-1]

  
        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(f"Feature dimension mismatch: Train {train_data.shape[1]}, Test {test_data.shape[1]}")
        
        # print("  -> Adding intercept column (ones)...")
        train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))])
        test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))])


        
        print(f"  -> Training (Data shape: {train_data.shape}, Target shape: {train_target.shape})...")
        best_model, _ = run_model_selection_pipeline(train_data, train_target, fit_intercept=True)

        
        B_hat = best_model['B']
        
        pred_scores = test_data @ B_hat
        pred_probs = plogis(pred_scores)
        
        pred_labels = (pred_probs >= 0.5).astype(int)

        
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

        header = f"{'Metric':<25} | {'Mean':<10} | {'Std':<10}"
        separator = "-" * 50

        print(header)
        print(separator)
        for name, m, s in zip(metric_names, mean_vals, std_vals):
            print(f"{name:<25} | {m:.4f}     | {s:.4f}")
        print(separator)

  
        if not os.path.exists("./results"):
            os.makedirs("./results")

        txt_filename = f"./results/{data_name}_summary.txt"

        print(f"\n正在保存统计摘要到: {txt_filename} ...")

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"Method: Robust Low-Rank Learning (XB+C)\n")
            f.write(f"Dataset: {data_name} ({n_repeats} runs)\n")
            f.write("=" * 50 + "\n")
            f.write(header + "\n")  
            f.write(separator + "\n")  

            
            for name, m, s in zip(metric_names, mean_vals, std_vals):
                
                f.write(f"{name:<25} | {m:.4f}     | {s:.4f}\n")

            f.write(separator + "\n")

        print("✅ 保存完成！")