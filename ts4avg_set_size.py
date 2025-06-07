import torch

from src.metrics import MetricConfig
from src.initialize_configs import initialize_configs_ts4cp


def main():
    device, dataset_model, predictor, _ = initialize_configs_ts4cp()

    n_iter = 100
    temperatures_vec = torch.arange(0.3, 3, 0.1, device=device)
    avg_size_metric = MetricConfig(**{'name': 'AvgSize'}).create_metric_instance()
    avg_size_metric_values_vec = avg_size_metric.compute_metric_as_func_of_temp(temperatures_vec, dataset_model, predictor, n_iter)

    temp_optimal_for_avg_size = temperatures_vec[torch.argmin(avg_size_metric_values_vec)]
    temp_opt_for_calibration = dataset_model.optimal_temperature_for_calibration

    print(
        f"You should use TS for 2 usages\n"
        f"1. Optimal for Calibration, with T={temp_opt_for_calibration}\n"
        f"2. Optimal for AvgSize, with T={temp_optimal_for_avg_size}"
    )

    temp_1_index = torch.abs(temperatures_vec - 1).argmin().item()
    optimal_avg_size_metric_value = torch.min(avg_size_metric_values_vec).item()

    print(f"AvgSize metric value before applying TS (T = 1.0): {avg_size_metric_values_vec[temp_1_index]}")

    print(f"AvgSize metric value after TS with tradeoff temperature (at T = {temp_optimal_for_avg_size}): "
          f"{optimal_avg_size_metric_value}")

    

if __name__ == "__main__":
    main()
