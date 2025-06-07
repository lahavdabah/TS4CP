import torch

from src.initialize_configs import initialize_configs_plots
from src.utils import plot_metric_as_function_of_temp


def main():
    device, dataset_model, predictor, metric = initialize_configs_plots()

    n_iter = 100
    temperatures_vec = torch.arange(0.3, 5, 0.1, device=device)
    mean_metric_values_vec = metric.compute_metric_as_func_of_temp(temperatures_vec, dataset_model, predictor, n_iter)

    plot_metric_as_function_of_temp(temperatures_vec.cpu().detach().numpy(), mean_metric_values_vec.cpu().detach().numpy(), metric.name, dataset_model, predictor)


if __name__ == "__main__":
    main()
