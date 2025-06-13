# TS4CP (ICML 2025)

**TS4CP** is the code accompanying the paper:  
[**On Temperature Scaling and Conformal Prediction of Deep Classifiers**](https://arxiv.org/abs/2402.05806)  
by *Lahav Dabah* and *Dr. Tom Tirer*.

## 🧠 Overview
This repository contains the implementation for our paper, focusing on a novel approach to utilize temperature scaling in the context of conformal prediction. The code allows replication of our results and experimentation with your own models.

Main scripts:

- `figure1_plots.py`: Generates the primary visualizations presented in **Figure 1** of the paper.
- `ts_for_AvgSize.py`: Computes prioritized temperature values for calibration and for average prediction set size metric (AvgSize).
- `ts_for_TopCovGap.py`: Computes prioritized temperature values for calibration and for conditional covarge metric (TopCovGap).
- `ts_for_tradeoff.py`: Computes prioritized temperature values for calibration and for tradeoffs between average prediction set size and conditional coverage metrics (AvgSize and TopCovGap). We denote the trade-off parameter as **`beta`**, which is user-configurable and takes values in the range **[0, 1]**. Setting **`beta = 0`** corresponds to optimizing the temperature for **AvgSize**, while **`beta = 1`** corresponds to optimizing for **TopCovGap**.

## Setup
Clone the repo and install dependencies:
```
git clone https://github.com/lahavdabah/TS4CP.git
cd TS4CP
pip install -r requirements.txt
```


## 🚀 Usage

Each main script relies on its respective YAML configuration file, which controls experiment parameters.

### Configuration Options

**In `config/plots_config.yaml` and `config/ts4cp_config.yaml`:**

```yaml
dataset_model:
  dataset_model_pair_name:  # Options: "Cifar10-ResNet50", "Cifar10-ResNet34", ...
  device: "cpu" | "cuda"

conformal:
  n:         # Number of samples for the CP procedure
  alpha:     # Coverage level (float in [0, 1])
  device:    # "cpu" or "cuda"
  method:    # "lac", "aps", "raps"
  lam_reg:   # (Only for RAPS) lambda regularization term
  k_reg:     # (Only for RAPS) k regularization term

metric:
  name:      # One of "AvgSize", "MarCovGap", "TopCovGap", "AvgCovGap" (only in plots_config.yaml)

ts4cp:
  n_eval:    # Number of samples used for evaluation - calculation of prioritized temperature (based on the user's preference) 
  beta:      # (Only in ts4cp_config.yaml) Tradeoff parameter ∈ [0, 1]
             # 0 → prioritized for AvgSize, 1 → prioritized for TopCovGap
```

Modify the configuration as needed, then run one of the main scripts based on your objective:

**Examples:**

- To apply the CP algorithm for prioritized *average prediction set size* (AvgSize), run:  
  `ts_for_AvgSize.py`

- To run the CP algorithm for prioritized *conditional coverage gap* (TopCovGap), run:  
  `ts4topcov_gap.py`

- To explore the trade-off between AvgSize and TopCovGap, adjust the **beta** parameter in `config/ts4cp_config.yaml` and run:  
  `ts_for_tradeoff.py`



### 💡 Using Your Own Model

You can use **TS4CP** with your own trained models!

1. Run your pretrained model on a validation set and save the logits (output scores before softmax) and the corresponding true labels in a `.npz` file.  
   The `.npz` file should contain two arrays:
   - `logits`: a 2D array of shape `(num_samples, num_classes)`
   - `labels`: a 1D array of shape `(num_samples,)`

2. Save the file in the `data/` directory using the following naming format:  `DatasetName-ModelName_data.npz`

3. In `src/data.py`, add your custom dataset-model pair to the list of supported options so it can be selected in the config file.

4. Update the relevant configuration YAML file (`plots_config.yaml` or `ts4cp_config.yaml`) with your new `dataset_model_pair_name`.

You can now run one of the main scripts with your own model and evaluate it using our temperature scaling and conformal prediction framework.



## 📄 Paper

This work has been **accepted for presentation at the 42st International Conference on Machine Learning (ICML 2025).**

If you find this work useful, please consider citing our paper:

```bibtex
@article{dabah2024temperature,
  title     = {On Temperature Scaling and Conformal Prediction of Deep Classifiers},
  author    = {Dabah, Lahav and Tirer, Tom},
  journal   = {arXiv preprint arXiv:2402.05806},
  year      = {2024}
}
```

## 📬 Contact

For questions, feedback, or collaboration inquiries, please contact the authors through the information provided in the [paper](https://arxiv.org/abs/2402.05806).

