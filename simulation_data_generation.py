import numpy as np
import scipy.io
import os
 
 
def plogis(x):
    return 1 / (1 + np.exp(-x))


def generate_simulation_datasets(num_datasets=20):
    print(f"开始生成 {num_datasets} 组仿真数据...")

    n_train = 200  
    n_test = 10000  
    p = 10  
    m = 10  
    r_true = 3  
    tau = 0.5  


    outlier_ratio = 0.1
    outlier_value = -180.0

    U_true = np.abs(np.random.randn(p, r_true))
    V_true = np.abs(np.random.randn(r_true, m))
    B0 = (U_true @ V_true)

    for run_idx in range(1, num_datasets + 1):

        # run_idx 1 -> seed 1, run_idx 2 -> seed 2...
        np.random.seed(run_idx)
        #np.random.seed(21)



        Sigma = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = tau ** abs(i - j)


        X_train = np.random.multivariate_normal(np.zeros(p), Sigma, n_train)


        C_true = np.zeros((n_train, m))


        n_outliers = int(outlier_ratio * n_train)


        C_true[:n_outliers, :] = outlier_value

        X_train[:n_outliers, :] = 3


        train_logit = X_train @ B0 + C_true
        train_probs = plogis(train_logit)
        Y_train = np.random.binomial(1, train_probs)




        X_test = np.random.multivariate_normal(np.zeros(p), Sigma, n_test)


        test_logit = X_test @ B0
        test_probs = plogis(test_logit)
        Y_test = np.random.binomial(1, test_probs)



        train_filename = f"simulation_train_{run_idx}.mat"
        test_filename = f"simulation_test_{run_idx}.mat"

        scipy.io.savemat(train_filename, {
            'data': X_train,  # (N, P)
            'target': Y_train,  # (N, M)
            'B': B0,  # (P, M)
            'C': C_true  # (N, M)
        })

        scipy.io.savemat(test_filename, {
            'data': X_test,  # (N_test, P)
            'target': Y_test  # (N_test, M)
        })

        if run_idx % 5 == 0:
            print(f"  已生成 {run_idx} / {num_datasets} 组数据...")

        print("数据生成完毕。")
        print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}, C={C_true.shape}")
        print(f"Test Shape:  X={X_test.shape}, Y={Y_test.shape}")


if __name__ == "__main__":
    np.random.seed(1)
    generate_simulation_datasets(20)