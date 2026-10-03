import numpy as np
 
def hamming_loss(pre_labels, test_target):
    """
    计算 Hamming Loss
    :param pre_labels: 预测标签矩阵 (numpy array)
    :param test_target: 真实标签矩阵 (numpy array)
    :return: Hamming Loss 值 (float)
    """
    pre_labels = np.array(pre_labels)
    test_target = np.array(test_target)

    miss_pairs = np.sum(pre_labels != test_target)

    total_elements = pre_labels.size

    return miss_pairs / total_elements


def example_based_measure(test_targets, predict_targets):
    """
    计算基于实例的多标签评估指标 (Example-Based Metrics)

    :param test_targets: 真实标签矩阵 (Labels x Samples)
    :param predict_targets: 预测标签矩阵 (Labels x Samples)
    :return: (Accuracy, Precision, Recall, Fmeasure)
    """

    test_targets = np.array(test_targets == 1, dtype=float)
    predict_targets = np.array(predict_targets == 1, dtype=float)


    intersection = np.sum(test_targets * predict_targets, axis=0)
    union = np.sum(np.maximum(test_targets, predict_targets), axis=0)

    # 2. 计算每个样本的 预测标签总数 和 真实标签总数
    pred_sum = np.sum(predict_targets, axis=0)
    true_sum = np.sum(test_targets, axis=0)


    accuracy_vec = np.zeros_like(union)
    mask = union > 0
    accuracy_vec[mask] = intersection[mask] / union[mask]

    # Precision: Intersection / Predicted_Positives
    precision_vec = np.zeros_like(pred_sum)
    mask = pred_sum > 0
    precision_vec[mask] = intersection[mask] / pred_sum[mask]

    # Recall: Intersection / Ground_Truth_Positives
    recall_vec = np.zeros_like(true_sum)
    mask = true_sum > 0
    recall_vec[mask] = intersection[mask] / true_sum[mask]

    # F-Measure: 2 * P * R / (P + R)
    fmeasure_vec = np.zeros_like(accuracy_vec)
    denom = precision_vec + recall_vec
    mask = denom > 0
    fmeasure_vec[mask] = 2 * (precision_vec[mask] * recall_vec[mask]) / denom[mask]

    # --- 返回平均值 ---
    return (np.mean(accuracy_vec),
            np.mean(precision_vec),
            np.mean(recall_vec),
            np.mean(fmeasure_vec))


def subset_accuracy_evaluation(test_target, predict_target):
    """
    计算子集准确率 (Subset Accuracy / Exact Match Ratio)

    :param test_target: 真实标签矩阵 (Labels x Samples)
    :param predict_target: 预测标签矩阵 (Labels x Samples)
    :return: Subset Accuracy (float)
    """

    test_target = np.array(test_target)
    predict_target = np.array(predict_target)

    matches = np.all(test_target == predict_target, axis=0)

    subset_accuracy = np.mean(matches)

    return subset_accuracy



def label_based_measure(test_targets, predict_targets):
    """
    计算基于标签的多标签评估指标 (Label-Based / Macro-Averaging)

    :param test_targets: 真实标签矩阵 (Labels x Samples)
    :param predict_targets: 预测标签矩阵 (Labels x Samples)
    :return: (Accuracy, Precision, Recall, Fmeasure)
    """

    test_targets = np.array(test_targets == 1, dtype=float)
    predict_targets = np.array(predict_targets == 1, dtype=float)


    intersection = np.sum(test_targets * predict_targets, axis=1)


    union = np.sum(np.maximum(test_targets, predict_targets), axis=1)


    pred_sum = np.sum(predict_targets, axis=1)
    true_sum = np.sum(test_targets, axis=1)


    accuracy_vec = np.zeros_like(union)
    mask = union > 0
    accuracy_vec[mask] = intersection[mask] / union[mask]

    # Precision: Intersection / Predicted_Positives
    precision_vec = np.zeros_like(pred_sum)
    mask = pred_sum > 0
    precision_vec[mask] = intersection[mask] / pred_sum[mask]

    # Recall: Intersection / Ground_Truth_Positives
    recall_vec = np.zeros_like(true_sum)
    mask = true_sum > 0
    recall_vec[mask] = intersection[mask] / true_sum[mask]

    # F-Measure: 2 * P * R / (P + R)
    fmeasure_vec = np.zeros_like(accuracy_vec)
    denom = precision_vec + recall_vec
    mask = denom > 0
    fmeasure_vec[mask] = 2 * (precision_vec[mask] * recall_vec[mask]) / denom[mask]

    # --- 返回所有类别的平均值 (Macro Average) ---
    return (np.mean(accuracy_vec),
            np.mean(precision_vec),
            np.mean(recall_vec),
            np.mean(fmeasure_vec))


def micro_f_measure(test_targets, predict_targets):
    """
    计算 Micro-F1 Measure (微平均 F1 值)

    :param test_targets: 真实标签矩阵 (Labels x Samples)
    :param predict_targets: 预测标签矩阵 (Labels x Samples)
    :return: Micro-F1 Score
    """

    test_targets = np.array(test_targets == 1, dtype=float)
    predict_targets = np.array(predict_targets == 1, dtype=float)


    intersection = np.sum(test_targets * predict_targets)

    # 预测为正的总数 (TP + FP)
    pre_sum = np.sum(predict_targets)

    # 真实为正的总数 (TP + FN)
    grd_sum = np.sum(test_targets)

    # --- 计算指标 ---

    # Global Precision
    if pre_sum > 0:
        precision = intersection / pre_sum
    else:
        precision = 0.0

    # Global Recall
    if grd_sum > 0:
        recall = intersection / grd_sum
    else:
        recall = 0.0

    # Micro-F1
    if (precision + recall) > 0:
        micro_f1 = 2 * precision * recall / (precision + recall)
    else:
        micro_f1 = 0.0

    return micro_f1


def average_precision(outputs, test_target):
    """
    计算平均精度 (Average Precision)

    :param outputs: 预测得分矩阵 (Labels x Samples)
    :param test_target: 真实标签矩阵 (Labels x Samples), 值为 +1 或 -1
    :return: Average Precision (float)
    """
    outputs = np.array(outputs)
    test_target = np.array(test_target)
    num_class, num_instance = outputs.shape

    ave_prec = 0.0
    valid_count = 0

    for i in range(num_instance):
        temp_target = test_target[:, i]

        # --- 1. 过滤逻辑 ---
        if np.sum(temp_target) != num_class and np.sum(temp_target) != -num_class:
            valid_count += 1

            temp_output = outputs[:, i]

            # 获取正例的原始索引 (0-based)
            label_indices = np.where(temp_target == 1)[0]
            label_size = len(label_indices)

            if label_size == 0:
                continue

            # --- 2. 排序 ---
            sorted_indices = np.argsort(temp_output)

            # --- 3. 构建 Indicator ---
            indicator = np.zeros(num_class)

            relevant_locs = []
            for lbl_idx in label_indices:
                loc = np.where(sorted_indices == lbl_idx)[0][0]
                relevant_locs.append(loc)
                indicator[loc] = 1

            # --- 4. 计算 AP ---
            summary = 0.0
            for loc in relevant_locs:
                precision_at_k = np.sum(indicator[loc:]) / (num_class - loc)
                summary += precision_at_k

            ave_prec += summary / label_size

    # 计算所有有效样本的平均值
    return ave_prec / valid_count if valid_count > 0 else 0.0


def one_error(outputs, test_target):
    """
    计算 One Error (One-Error Loss)

    :param outputs: 预测得分矩阵 (Labels x Samples)
    :param test_target: 真实标签矩阵 (Labels x Samples), 值为 +1 或 -1
    :return: One Error (float)
    """
    outputs = np.array(outputs)
    test_target = np.array(test_target)
    num_class, num_instance = outputs.shape

    one_err_count = 0
    valid_count = 0

    for i in range(num_instance):
        temp_target = test_target[:, i]
        temp_output = outputs[:, i]

        # --- 1. 过滤无效样本 ---
        if np.sum(temp_target) != num_class and np.sum(temp_target) != -num_class:
            valid_count += 1

            # --- 2. 寻找最高分 ---
            max_val = np.max(temp_output)

            # --- 3. 检查最高分是否命中 ---

            # 找到所有得分等于最大值的类别的索引
            max_indices = np.where(temp_output == max_val)[0]

            # 找到真实标签为 1 的索引
            true_labels = np.where(temp_target == 1)[0]

            # 检查是否有交集
            # np.intersect1d 返回两个数组的交集
            # 如果交集为空 (size == 0)，说明最高分的预测全都是错的 -> Error + 1
            if np.intersect1d(max_indices, true_labels).size == 0:
                one_err_count += 1

    return one_err_count / valid_count if valid_count > 0 else 0.0





def ranking_loss(outputs, test_target):
    """
    计算 Ranking Loss (排序损失)

    :param outputs: 预测得分矩阵 (Labels x Samples)
    :param test_target: 真实标签矩阵 (Labels x Samples), 值为 +1 或 -1
    :return: Ranking Loss (float)
    """
    outputs = np.array(outputs)
    test_target = np.array(test_target)
    num_class, num_instance = outputs.shape

    rank_loss_sum = 0.0
    valid_count = 0

    for i in range(num_instance):
        temp_target = test_target[:, i]
        temp_output = outputs[:, i]

        # --- 1. 过滤无效样本 ---
        if np.sum(temp_target) != num_class and np.sum(temp_target) != -num_class:
            valid_count += 1

            # --- 2. 区分正负标签的索引 ---
            pos_indices = np.where(temp_target == 1)[0]
            neg_indices = np.where(temp_target != 1)[0]  # 假设非1即为负例(-1)

            m = len(pos_indices)  # 正例数量
            n = len(neg_indices)  # 负例数量

            if m == 0 or n == 0:
                continue

            # --- 3. 计算错误对 (Violations) ---
            pos_scores = temp_output[pos_indices].reshape(-1, 1)
            neg_scores = temp_output[neg_indices].reshape(1, -1)

            violations = np.sum(pos_scores <= neg_scores)

            # --- 4. 归一化并累加 ---

            rank_loss_sum += violations / (m * n)

    return rank_loss_sum / valid_count if valid_count > 0 else 0.0


def coverage_score(outputs, test_target):
    """
    计算 Coverage 指标

    :param outputs: 预测得分矩阵 (Labels x Samples)
    :param test_target: 真实标签矩阵 (Labels x Samples), 值为 +1 或 -1
    :return: Coverage (float)
    """
    outputs = np.array(outputs)
    test_target = np.array(test_target)
    num_class, num_instance = outputs.shape

    cover_sum = 0.0

    for i in range(num_instance):
        temp_target = test_target[:, i]
        temp_output = outputs[:, i]


        label_indices = np.where(temp_target == 1)[0]

        if len(label_indices) == 0:
            continue

        # --- 1. 排序 ---
        sorted_indices = np.argsort(temp_output, kind='stable')

        # --- 2. 寻找最“差”的真实标签 ---

        min_loc_ascending = num_class

        for lbl_idx in label_indices:

            loc = np.where(sorted_indices == lbl_idx)[0][0]

            if loc < min_loc_ascending:
                min_loc_ascending = loc

        # --- 3. 计算覆盖深度 ---

        cover_sum += (num_class - min_loc_ascending)

    return ((cover_sum / num_instance) - 1) / num_class

