from typing import Literal

import torch
from tqdm import tqdm
from pydantic import BaseModel

from src.utils import compute_avg_size, compute_mar_cov_gap, compute_top_cov_gap, compute_avg_cov_gap
from src.data import DatasetModelPair
from src.algorithms import ConformalPredictor


class MetricConfig(BaseModel):
    name: Literal["AvgSize", "MarCovGap", "TopCovGap", "AvgCovGap"]

    def create_metric_instance(self):
        return Metric(self)


class Metric:
    def __init__(self, config: MetricConfig):
        self.name = config.name
    

    def compute_metric(self, prediction_sets, val_labels, alpha):
        """Compute the specified metric based on the prediction sets and validation labels.
        Args:
            prediction_sets (torch.Tensor): A tensor of shape (n_samples, n_classes) where each row represents the prediction set for a sample.
            val_labels (torch.Tensor): A tensor of true labels for the validation set.
            alpha (float): The desired coverage level.
        Returns:
            metric_value (float): The computed metric value.
        """
        if self.name == "AvgSize":
            return compute_avg_size(prediction_sets)
        
        elif self.name == "MarCovGap":
            return compute_mar_cov_gap(prediction_sets, val_labels, alpha)
        
        elif self.name == "TopCovGap":
            return compute_top_cov_gap(prediction_sets, val_labels, alpha)
        
        elif self.name == "AvgCovGap":
            return compute_avg_cov_gap(prediction_sets, val_labels, alpha)
        
        else:
            raise ValueError(f"Unknown metric name: {self.name}")
        

    def compute_metric_as_func_of_temp(self, temperatures_vec: torch.Tensor, dataset_model: DatasetModelPair, predictor: ConformalPredictor, n_iter: int) -> torch.Tensor:
        """Compute the metric as a function of temperature.
        Args:
            temperatures_vec (torch.Tensor): A tensor of temperatures to evaluate.
            dataset_model: An object containing the dataset and model information.
            predictor: An object that can predict using the model with temperature scaling.
            metric: An object that computes the desired metric.
        n_iter (int): Number of iterations for averaging the metric values.

        Returns:
            mean_metric_values_vec (torch.Tensor): A tensor containing the mean metric values for each temperature.
        """
        mean_metric_values_vec = torch.zeros_like(temperatures_vec)
        for _ in tqdm(range(n_iter)):
            metric_values_vec = torch.zeros_like(temperatures_vec)
            for index, temp in enumerate(temperatures_vec):
                smx_ts = dataset_model.smx_after_temp_scaling(temp)
                smx_ts, val_labels_ts = predictor.predict(smx_ts, dataset_model.true_labels)
                metric_value_ts = self.compute_metric(smx_ts, val_labels_ts, predictor.alpha)
                metric_values_vec[index] = metric_value_ts
            mean_metric_values_vec += (1 / n_iter) * metric_values_vec

        return mean_metric_values_vec
        



            
