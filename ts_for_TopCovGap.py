import torch
import torch.nn.functional as F

from src.metrics import MetricConfig
from src.initialize_configs import initialize_configs_ts4cp
from src.utils import compute_top_cov_gap

def main():
    # ------------------------
    # Initialize configs
    # ------------------------
    device, dataset_model, predictor, ts4cp = initialize_configs_ts4cp()
    n_cal = predictor.n
    n_eval = ts4cp.n_eval

    # ------------------------
    # Prepare evaluation and validation sets
    # ------------------------
    eval_indexes = torch.randperm(dataset_model.smx.shape[0])
    logits_eval = dataset_model.logits[eval_indexes[:(n_cal + n_eval)], :]
    smx_eval = dataset_model.smx[eval_indexes[:(n_cal + n_eval)], :]
    true_labels_eval = dataset_model.true_labels[eval_indexes[:(n_cal + n_eval)]]

    logits_val = dataset_model.logits[eval_indexes[(n_cal + n_eval):], :]
    smx_val = dataset_model.smx[eval_indexes[(n_cal + n_eval):], :]
    true_labels_val = dataset_model.true_labels[eval_indexes[(n_cal + n_eval):]]

    # ------------------------
    # Metric setup and calculation of metrics as function of temperature
    # ------------------------
    n_iter = 1
    temperatures_vec = torch.arange(0.3, 3, 0.1, device=device)
    top_cov_gap_metric = MetricConfig(**{'name': 'TopCovGap'}).create_metric_instance()

    top_cov_gap_metric_values_vec, mean_qhat_vec = top_cov_gap_metric.compute_metric_as_func_of_temp(temperatures_vec, logits_eval, true_labels_eval, predictor, n_iter)

    # ------------------------
    # Choose prioritized temperature
    # ------------------------
    index_for_prioritized_top_cov_gap = torch.argmin(top_cov_gap_metric_values_vec)
    temp_prioritized_for_top_cov_gap = temperatures_vec[index_for_prioritized_top_cov_gap]
    temp_opt_for_calibration = dataset_model.optimal_temperature_for_calibration

    print(
        f"You should use TS for 2 usages\n"
        f"1. Prioritized for Calibration, with T={temp_opt_for_calibration}\n"
        f"2. Prioritized for AvgSize, with T={temp_prioritized_for_top_cov_gap}"
    )

    # ------------------------
    # Evaluate metrics before and after temperature scaling
    # ------------------------
    index_of_temp_1 = torch.argmin(torch.abs(temperatures_vec - 1)).item()

    val_pred_sets, _, _ = predictor.predict(smx_val, true_labels_val, mean_qhat_vec[index_of_temp_1])
    val_pred_sets_ts, _, _ = predictor.predict(F.softmax(logits_val/temp_prioritized_for_top_cov_gap), true_labels_val, mean_qhat_vec[index_for_prioritized_top_cov_gap])

    top_cov_gap = compute_top_cov_gap(val_pred_sets, true_labels_val, predictor.alpha)
    top_cov_gap_after_ts = compute_top_cov_gap(val_pred_sets_ts, true_labels_val, predictor.alpha)

    print(f"TopCovGap metric value before applying TS (T = 1.0): {top_cov_gap}")

    print(f"TopCovGap metric value after TS with prioritzed temperature for TopCovGap (at T = {temp_prioritized_for_top_cov_gap}): "
          f"{top_cov_gap_after_ts}")


    

if __name__ == "__main__":
    main()
