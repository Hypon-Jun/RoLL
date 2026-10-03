import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


 
def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    mat = sio.loadmat(path)

    if 'data' in mat:
        X = mat['data']
    elif 'X' in mat:
        X = mat['X']
    else:
        raise ValueError("Mat文件中缺少 'data' 变量")

    if 'target' in mat:
        Y = mat['target']
    elif 'Y' in mat:
        Y = mat['Y']
    else:
        raise ValueError("Mat文件中缺少 'target' 变量")

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    if X.shape[0] != Y.shape[0]:
        print(f"注意: Data行数 {X.shape[0]} 与 Target行数 {Y.shape[0]} 不一致，尝试转置 Data...")
        X = X.T
    return X, Y


def inject_structured_outliers(Y, ratio, random_state=None):

    if ratio <= 0: return Y

    rng = np.random.RandomState(random_state)
    n_samples, n_labels = Y.shape
    n_noise = int(n_samples * ratio)

    noise_indices = rng.choice(n_samples, n_noise, replace=False)

    Y_noisy = Y.copy()
    Y_noisy[noise_indices, :] = 1 - Y_noisy[noise_indices, :]

    return Y_noisy




if __name__ == "__main__":

    data_name = 'emotions'
    base_input_path = f'dataset/{data_name}.mat'

    output_dir = f'dataset/{data_name}_runs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    n_repeats = 20 
    noise_ratio = 0.1  
    test_size = 0.2  


    print(f"正在读取源数据: {base_input_path}")
    X_orig, Y_orig = load_data(base_input_path)

    Y_orig[Y_orig == -1] = 0

    print(f"数据总维度: X={X_orig.shape}, Y={Y_orig.shape}")
    print(f"准备生成 {n_repeats} 组数据，污染比例={noise_ratio}...\n")


    for run_idx in range(1, n_repeats + 1):
        current_seed = run_idx + 2024


        X_train, X_test, Y_train, Y_test = train_test_split(
            X_orig, Y_orig,
            test_size=test_size,
            random_state=current_seed,
            shuffle=True
        )


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        Y_train_noisy = inject_structured_outliers(Y_train, noise_ratio, random_state=current_seed)


        train_path = os.path.join(output_dir, f'{data_name}_train_{run_idx}.mat')
        test_path = os.path.join(output_dir, f'{data_name}_test_{run_idx}.mat')

        sio.savemat(train_path, {'data': X_train, 'target': Y_train_noisy})
        sio.savemat(test_path, {'data': X_test, 'target': Y_test})

        if run_idx % 5 == 0 or run_idx == 1:
            print(f" [Run {run_idx}/{n_repeats}] 完成 -> 训练集: {X_train.shape}, 测试集: {X_test.shape}")

    print("\n✅ 所有 20 组数据生成完毕！")
    print(f"文件保存在: {output_dir}")