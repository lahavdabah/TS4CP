import yaml
import torch

from src.data import DatasetModelPairConfig, DatasetModelPair
from src.algorithms import ConformalConfig, ConformalPredictor
from src.metrics import MetricConfig, Metric
from src.ts_for_cp import TS4CPConfig, TS4CP


def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def initialize_configs_plots() -> tuple[torch.device, DatasetModelPair, ConformalPredictor, Metric]:
    """Initialize configurations for plotting metric as function of temperatures.
    Returns:
        device (torch.device): The device to be used for computations.
        dataset_model (DatasetModelPair): An instance of DatasetModelPair containing dataset and model configurations.
        predictor (ConformalPredictor): An instance of ConformalPredictor containing conformal prediction configurations.
        metric (Metric): An instance of Metric containing metric configurations.
    """
    path = "config/plots_config.yaml"

    config_dict = load_yaml_config(path)
    device = torch.device(config_dict["conformal"]["device"])

    # Load dataset/model config
    dataset_model_cfg = DatasetModelPairConfig(**config_dict["dataset_model"])
    dataset_model = DatasetModelPair(dataset_model_cfg)
    
    # Load conformal config
    conformal_cfg = ConformalConfig(**config_dict["conformal"])
    predictor = ConformalPredictor(conformal_cfg)

    # Load metric config
    mtc_cfg = MetricConfig(**config_dict["metric"])
    metric = Metric(mtc_cfg)

    return device, dataset_model, predictor, metric


def initialize_configs_ts4cp() -> tuple[torch.device, DatasetModelPair, ConformalPredictor, float]:
    """Initialize configurations for the using TS4CP.
    Returns:
        device (torch.device): The device to be used for computations.
        dataset_model (DatasetModelPair): An instance of DatasetModelPair containing dataset and model configurations.
        predictor (ConformalPredictor): An instance of ConformalPredictor containing conformal prediction configurations.
        beta (float): The beta value used in the conformal prediction.
    """
    path = "config/ts4cp_config.yaml"

    config_dict = load_yaml_config(path)
    device = torch.device(config_dict["conformal"]["device"])

    # Load dataset/model config
    dataset_model_cfg = DatasetModelPairConfig(**config_dict["dataset_model"])
    dataset_model = DatasetModelPair(dataset_model_cfg)
    
    # Load conformal config
    conformal_cfg = ConformalConfig(**config_dict["conformal"])
    predictor = ConformalPredictor(conformal_cfg)

    # Load beta
    ts4cp_cfg = TS4CPConfig(**config_dict["ts4cp"])
    ts4cp = TS4CP(ts4cp_cfg)

    return device, dataset_model, predictor, ts4cp
