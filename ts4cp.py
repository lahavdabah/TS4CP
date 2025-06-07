import torch

from src.metrics import MetricConfig
from src.initialize_configs import initialize_configs_ts4cp


def main():
    # beta is the trade-off parameter between AvgSize and TopCovGap which is user-configurable and takes values in the range [0, 1]. 
    # Setting beta = 0 corresponds to optimizing the temperature for AvgSize, while beta = 1 corresponds to optimizing for TopCovGap.
    device, dataset_model, predictor, beta = initialize_configs_ts4cp()

    n_iter = 100
    temperatures_vec = torch.arange(0.3, 3, 0.1, device=device)
    avg_size_metric = MetricConfig(**{'name': 'AvgSize'}).create_metric_instance()
    top_cov_gap_metric = MetricConfig(**{'name': 'TopCovGap'}).create_metric_instance()
    avg_size_metric_values_vec = avg_size_metric.compute_metric_as_func_of_temp(temperatures_vec, dataset_model, predictor, n_iter)
    top_cov_gap_metric_values_vec = top_cov_gap_metric.compute_metric_as_func_of_temp(temperatures_vec, dataset_model, predictor, n_iter)

    temp_optimal_for_avg_size = temperatures_vec[torch.argmin(avg_size_metric_values_vec)]
    temp_optimal_for_top_cov_gap = temperatures_vec[torch.argmin(top_cov_gap_metric_values_vec)]

    temp_trade_off = temp_optimal_for_top_cov_gap * beta + temp_optimal_for_avg_size * (1 - beta)
    temp_opt_for_calibration = dataset_model.optimal_temperature_for_calibration

    print(
        f"Recommended temperature scaling (TS) usage:\n"
        f"1. For calibration: use T = {temp_opt_for_calibration}\n"
        f"2. For AvgSize–TopCovGap trade-off: use T = {temp_trade_off}\n"
    )

    temp_1_index = torch.abs(temperatures_vec - 1).argmin().item()
    tradeoff_temp_index = torch.abs(temperatures_vec - temp_trade_off).argmin().item()

    print(f"Metrics before applying TS (T = 1.0):")
    print(f"  - AvgSize:    {avg_size_metric_values_vec[temp_1_index]}")
    print(f"  - TopCovGap:  {top_cov_gap_metric_values_vec[temp_1_index]}\n")

    print(f"Metrics after TS with tradeoff temperature (at T = {temp_trade_off}):")
    print(f"  - AvgSize:    {avg_size_metric_values_vec[tradeoff_temp_index]}")
    print(f"  - TopCovGap:  {top_cov_gap_metric_values_vec[tradeoff_temp_index]}")


    

if __name__ == "__main__":
    main()
