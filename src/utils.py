import torch
import matplotlib.pyplot as plt

from src.data import DatasetModelPair
from src.algorithms import ConformalPredictor


def compute_avg_size(prediction_sets: torch.Tensor) -> float:
    """Compute the average size of prediction sets.

    Args:
        prediction_sets (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.

    Returns:
        avg_size (float): The average size of the prediction sets.
    """
    avg_size = prediction_sets.sum(dim=1).float().mean()
    return avg_size


def compute_mar_cov_gap(prediction_sets: torch.Tensor, val_labels: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute the Mar coverage gap.
    Args:
        prediction_sets (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.
        val_labels (torch.Tensor): A tensor of true labels for the validation set.
        alpha (float): The desired coverage level.
    Returns:
        mar_cov_gap (torch.Tensor): The Mar coverage gap.
    """
    mar_cov_gap = torch.abs(prediction_sets[torch.arange(prediction_sets.size(0)), val_labels.long()].float().mean() - (1 - alpha))
    return mar_cov_gap


def get_coverage_of_specific_class(prediction_sets_of_specific_class: torch.Tensor, true_label: torch.Tensor, class_size: int) -> float:
    """Compute the coverage for a specific class.
    Args:
        prediction_sets_of_specific_class (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.
        true_label (int): The true label of the class for which coverage is computed.
        class_size (int): The number of samples in the specific class.
    Returns:
        coverage (float): The coverage for the specific class.
    """
    return torch.sum(prediction_sets_of_specific_class[:, true_label]) / class_size


def get_conditional_coverage_by_classes(prediction_sets: torch.Tensor, val_labels: torch.Tensor) -> torch.Tensor:
    """Compute the conditional coverage for each classes.
    Args:
        prediction_sets (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.
        val_labels (torch.Tensor): A tensor of true labels for the validation set.
    Returns:
        coverage_vec (torch.Tensor): A tensor containing the coverage for each class.
    """
    sorted_indexes = torch.argsort(val_labels)
    val_labels = val_labels[sorted_indexes]
    prediction_sets = prediction_sets[sorted_indexes, :]

    unique_elements, counts = torch.unique(val_labels, return_counts=True)
    counts_arr = torch.tensor(counts)
    coverage_vec = torch.zeros(len(unique_elements))
    last_size = 0
    for index, true_label in enumerate(unique_elements):
        coverage_vec[index] = get_coverage_of_specific_class(prediction_sets
                                                             [last_size:last_size + counts_arr[index], :], true_label,
                                                             counts_arr[index])
        last_size = last_size + counts_arr[index]
    return coverage_vec


def compute_top_cov_gap(prediction_sets: torch.Tensor, val_labels: torch.Tensor, alpha: float) -> float:
    """Compute the top coverage gap.
    Args:
        prediction_sets (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.
        val_labels (torch.Tensor): A tensor of true labels for the validation set.
        alpha (float): The desired coverage level.
    Returns:
        top_cov_gap (torch.Tensor): TopCovGap metric value.
    """
    num_classes = prediction_sets.shape[1]
    coverage_vec = get_conditional_coverage_by_classes(prediction_sets, val_labels)
    top_cov_gap = torch.mean(torch.topk(torch.abs(coverage_vec - (1 - alpha)), k=max(int(0.05 * num_classes), 1))[0])
    return top_cov_gap


def compute_avg_cov_gap(prediction_sets: torch.Tensor, val_labels: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute the average coverage gap.
    Args:
        prediction_sets (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.
        val_labels (torch.Tensor): A tensor of true labels for the validation set.
        alpha (float): The desired coverage level.
    Returns:
        avg_cov_gap (torch.Tensor): AvgCovGap metric value.
    """
    coverage_vec = get_conditional_coverage_by_classes(prediction_sets, val_labels)
    avg_cov_gap = torch.mean(torch.abs(coverage_vec - (1 - alpha)))
    return avg_cov_gap



   


def plot_metric_as_function_of_temp(temperatures_vec: torch.Tensor, metric_vec: torch.Tensor, metric_name: str, dataset_model: DatasetModelPair, predictor: ConformalPredictor) -> None:
    """Plot the metric as a function of temperature.
    Args:
        temperatures_vec (torch.Tensor): A tensor of temperatures to evaluate.
        metric_vec (torch.Tensor): A tensor containing the metric values for each temperature.
        metric_name (str): The name of the metric to be plotted.
        dataset_model: An object containing the dataset and model information.
        predictor: An object that can predict using the model with temperature scaling.
    """
    T_opt_for_calibration = dataset_model.optimal_temperature_for_calibration
    plt.plot(temperatures_vec, metric_vec, label=metric_name, linewidth=2.5)
    plt.title(f"{predictor.method.upper()} Algorithm", fontstyle='normal', fontsize=18)
    plt.xlabel('Temperature T', fontstyle='normal', fontsize=16)
    plt.ylabel(metric_name, fontstyle='normal', fontsize=16)
    plt.axvline(x=T_opt_for_calibration, color='r', linestyle='--', label='$T^*$')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'plots/{metric_name}_{dataset_model.dataset_model_pair_name}_{predictor.method}_plot.png')
    plt.show()



