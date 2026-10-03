import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

 
 

def load_data_triple(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    mat = sio.loadmat(path)


    if 'data' in mat:
        X = mat['data']
    else:
        raise ValueError("Mat文件中缺少 'data' 变量")


    if 'target' in mat:
        Y_gt = mat['target']
    else:
        raise ValueError("Mat文件中缺少 'target' 变量")


    if 'candidate_labels' in mat:
        Y_cand = mat['candidate_labels']
    elif 'partial_labels' in mat:
        Y_cand = mat['partial_labels']
    else:
        raise ValueError("Mat文件中缺少 'candidate_labels' 变量")

    X = X.astype(np.float64)
    Y_gt = Y_gt.astype(np.float64)
    Y_cand = Y_cand.astype(np.float64)


    Y_gt = Y_gt.T
    Y_cand = Y_cand.T


    if X.shape[0] != Y_gt.shape[0]:
        print(f"提示: Data行数 {X.shape[0]} 与 转置后的Target行数 {Y_gt.shape[0]} 不一致，正在尝试转置 Data 以匹配...")
        X = X.T

    if X.shape[0] != Y_cand.shape[0]:
        raise ValueError(f"错误: Candidate Labels 转置后行数 ({Y_cand.shape[0]}) 与 Data ({X.shape[0]}) 不匹配！")

    return X, Y_gt, Y_cand




if __name__ == "__main__":
    data_name = 'music_emotion'  
    base_input_path = f'dataset/{data_name}.mat'


    output_dir = f'dataset/{data_name}_runs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    n_repeats = 20  
    test_size = 0.2  


    print(f"正在读取源数据: {base_input_path}")
    X_orig, Y_gt_orig, Y_cand_orig = load_data_triple(base_input_path)

    Y_gt_orig[Y_gt_orig == -1] = 0
    Y_cand_orig[Y_cand_orig == -1] = 0

    print(f"数据加载完成: X={X_orig.shape}, Target(转置后)={Y_gt_orig.shape}, Candidate(转置后)={Y_cand_orig.shape}")
    print(f"准备生成 {n_repeats} 组数据 (Train使用Candidate, Test使用Target)...\n")


    for run_idx in range(1, n_repeats + 1):

        current_seed = run_idx + 2024



        X_train, X_test, \
        Y_cand_train, Y_cand_test, \
        Y_gt_train, Y_gt_test = train_test_split(
            X_orig, Y_cand_orig, Y_gt_orig,
            test_size=test_size,
            random_state=current_seed,
            shuffle=True
        )


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_path = os.path.join(output_dir, f'{data_name}_train_{run_idx}.mat')
        test_path = os.path.join(output_dir, f'{data_name}_test_{run_idx}.mat')

 
        sio.savemat(train_path, {'data': X_train, 'target': Y_cand_train})

   
        sio.savemat(test_path, {'data': X_test, 'target': Y_gt_test})

        if run_idx % 5 == 0 or run_idx == 1:
            print(f" [Run {run_idx}/{n_repeats}] 完成")
            print(f"    - Train(w/ Cand): {X_train.shape}, Labels: {Y_cand_train.shape}")
            print(f"    - Test (w/ GT)  : {X_test.shape},  Labels: {Y_gt_test.shape}")

    print("\n✅ 所有 20 组数据生成完毕！")
    print(f"文件保存在: {output_dir}")