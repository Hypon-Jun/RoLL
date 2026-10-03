import numpy as np

from metrics import (
    hamming_loss,
    example_based_measure,
    subset_accuracy_evaluation,
    label_based_measure,
    ranking_loss,
    one_error,
    coverage_score,
    average_precision,
    micro_f_measure
)
 

def evaluation_all(pre_labels, outputs, test_target):
    """
    Evaluation for MLC algorithms, calculating fifteen evaluation metrics.

    Syntax:
        result_all = evaluation_all(pre_labels, outputs, test_target)

    Input:
        test_target : np.array
            L x num_test data matrix of groundtruth labels (Labels x Samples)
            Note: Usually contains {+1, -1} or {1, 0}
        pre_labels  : np.array
            L x num_test data matrix of predicted labels (Labels x Samples)
            Note: Usually contains {+1, -1} or {1, 0}
        outputs     : np.array
            L x num_test data matrix of scores (Labels x Samples)

    Output:
        result_all : np.array (15 x 1)
            1:  HammingLoss
            2:  ExampleBasedAccuracy
            3:  ExampleBasedPrecision
            4:  ExampleBasedRecall
            5:  ExampleBasedFmeasure
            6:  SubsetAccuracy
            7:  LabelBasedAccuracy
            8:  LabelBasedPrecision
            9:  LabelBasedRecall
            10: LabelBasedFmeasure
            11: MicroF1Measure
            12: Average_Precision
            13: OneError
            14: RankingLoss
            15: Coverage
    """

    # 确保输入是 numpy 数组
    pre_labels = np.array(pre_labels)
    outputs = np.array(outputs)
    test_target = np.array(test_target)

    # 初始化结果向量 (15 x 1)
    #result_all = np.zeros((15, 1))

    # --- 1. 计算各项指标 ---

    # Hamming Loss
    hl = hamming_loss(pre_labels, test_target)

    # Example-Based Metrics
    ex_acc, ex_prec, ex_rec, ex_f1 = example_based_measure(test_target, pre_labels)

    # Subset Accuracy
    subset_acc = subset_accuracy_evaluation(test_target, pre_labels)

    # Label-Based Metrics
    lbl_acc, lbl_prec, lbl_rec, lbl_f1 = label_based_measure(test_target, pre_labels)

    # Ranking Loss
    rl = ranking_loss(outputs, test_target)

    # One Error
    oe = one_error(outputs, test_target)

    # Coverage
    cov = coverage_score(outputs, test_target)

    # Average Precision
    ap = average_precision(outputs, test_target)

    # Micro F1 Measure
    micro_f1 = micro_f_measure(test_target, pre_labels)

    # --- 2. 填入结果向量 ---

    # result_all[0, 0] = hl
    #
    # result_all[1, 0] = ex_acc
    # result_all[2, 0] = ex_prec
    # result_all[3, 0] = ex_rec
    # result_all[4, 0] = ex_f1
    #
    # result_all[5, 0] = subset_acc
    #
    # result_all[6, 0] = lbl_acc
    # result_all[7, 0] = lbl_prec
    # result_all[8, 0] = lbl_rec
    # result_all[9, 0] = lbl_f1
    #
    # result_all[10, 0] = micro_f1
    # result_all[11, 0] = ap
    # result_all[12, 0] = oe
    # result_all[13, 0] = rl
    # result_all[14, 0] = cov
    #
    # return result_all
    result_list = [
        hl,  # 1: HammingLoss
        ex_acc,  # 2: ExampleBasedAccuracy
        ex_prec,  # 3: ExampleBasedPrecision
        ex_rec,  # 4: ExampleBasedRecall
        ex_f1,  # 5: ExampleBasedFmeasure
        subset_acc,  # 6: SubsetAccuracy
        lbl_acc,  # 7: LabelBasedAccuracy
        lbl_prec,  # 8: LabelBasedPrecision
        lbl_rec,  # 9: LabelBasedRecall
        lbl_f1,  # 10: LabelBasedFmeasure
        micro_f1,  # 11: MicroF1Measure
        ap,  # 12: Average_Precision
        oe,  # 13: OneError
        rl,  # 14: RankingLoss
        cov  # 15: Coverage
    ]

    return result_list




