from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel


class DatasetModelPairConfig(BaseModel):
    dataset_model_pair_name: Literal["Cifar10-ResNet50", "Cifar10-ResNet34", "Cifar100-ResNet50", "Cifar100-DenseNet121", "ImageNet-ResNet152", "ImageNet-ViT"]
    device: Literal["cpu", "cuda"]

    def create_dataset_model_instance(self):
        return DatasetModelPair(self)


class DatasetModelPair:
    def __init__(self, config: DatasetModelPairConfig):
        self.dataset_model_pair_name = config.dataset_model_pair_name
        self.dataset_name, self.model_name = self.dataset_model_pair_name.split("-")
        self.device = torch.device(config.device)
        self.path = f"data/{self.dataset_model_pair_name}_data.npz"
        self._data = np.load(self.path)
        self.logits = torch.from_numpy(self._data['logits']).to(self.device)
        self.smx = F.softmax(self.logits, dim=1)
        self.true_labels = torch.from_numpy(self._data['labels']).to(torch.int64).to(self.device)
        self.optimal_temperature_for_calibration = self.get_opt_temp_for_calibration()

    def smx_after_temp_scaling(self, temperature: float):
        """Apply temperature scaling to the logits and return the softmax probabilities.
        Args:
            temperature (float): The temperature value for scaling.
        Returns:
            torch.Tensor: The softmax probabilities after temperature scaling.
        """
        logits_ts = self.logits / temperature
        smx_ts = F.softmax(logits_ts, dim=1)
        return smx_ts
    

    def get_opt_temp_for_calibration(self) -> float:
        """Get the optimal temperature for calibration based on the dataset and model.
        Args:
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
        Returns:
            float: The optimal temperature for calibration.
        Raises:
            ValueError: If the dataset and model combination is not found in the mapping.
        """ 
        optimal_t_mapping: dict[tuple[str, str], float] = {
        ('ImageNet', 'ResNet152'): 1.22,
        ('ImageNet', 'ResNet50'): 1.143,
        ('ImageNet', 'DenseNet121'): 1.06,
        ('ImageNet', 'ViT'): 1.18,
        ('Cifar100', 'ResNet50'): 1.504,
        ('Cifar100', 'ResNet18'): 1.045,
        ('Cifar100', 'DenseNet121'): 1.426,
        ('Cifar10', 'ResNet50'): 1.761,
        ('Cifar10', 'ResNet34'): 1.802,
        }

        mapping_key = (self.dataset_name, self.model_name)
        if mapping_key in optimal_t_mapping:
            return optimal_t_mapping[mapping_key]
        else:
            raise ValueError(f"Optimal temperature for calibration not found for dataset '{self.dataset_name}' and model '{self.model_name}'.")
 

