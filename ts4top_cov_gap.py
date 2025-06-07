import torch

from src.metrics import MetricConfig
from src.initialize_configs import initialize_configs_ts4cp


def main():
    device, dataset_model, predictor, _ = initialize_configs_ts4cp()

    n_iter = 100
    temperatures_vec = torch.arange(0.3, 3, 0.1, device=device)
    top_cov_gap_metric = MetricConfig(**{'name': 'TopCovGap'}).create_metric_instance()
    top_cov_gap_metric_values_vec = top_cov_gap_metric.compute_metric_as_func_of_temp(temperatures_vec, dataset_model, predictor, n_iter)

    temp_optimal_for_top_cov_gap = temperatures_vec[torch.argmin(top_cov_gap_metric_values_vec)]
    temp_opt_for_calibration = dataset_model.optimal_temperature_for_calibration

    print(
        f"You should use TS for 2 usages\n"
        f"1. Optimal for Calibration, with T={temp_opt_for_calibration}\n"
        f"2. Optimal for TopCovGap, with T={temp_optimal_for_top_cov_gap}"
    )

    temp_1_index = torch.abs(temperatures_vec - 1).argmin().item()
    optimal_top_cov_gap_metric_value = torch.min(top_cov_gap_metric_values_vec).item()

    print(f"TopCovGap metric value before applying TS (T = 1.0): {top_cov_gap_metric_values_vec[temp_1_index]}")


    print(f"TopCovGap metric value after TS with tradeoff temperature (at T = {temp_optimal_for_top_cov_gap}): " 
          f"{optimal_top_cov_gap_metric_value}")


    

if __name__ == "__main__":
    main()
