from typing import Literal, Optional

import torch
from pydantic import BaseModel


class ConformalConfig(BaseModel):
    n: int
    alpha: float
    device: Literal["cpu", "cuda"]
    method: Literal["LAC", "APS", "RAPS"]
    lam_reg: Optional[float] = 0.01  # Only used by RAPS
    k_reg: Optional[int] = 5         # Only used by RAPS



class ConformalPredictor:
    def __init__(self, config: ConformalConfig):
        self.n = config.n
        self.alpha = config.alpha
        self.device = torch.device(config.device)
        self.method = config.method
        self.lam_reg = config.lam_reg
        self.k_reg = config.k_reg

    def _split(self, smx: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the data into calibration and validation sets.
        Args:
            smx (torch.Tensor): The softmax probabilities of shape (n_samples, n_classes).
            labels (torch.Tensor): The true labels of shape (n_samples,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Calibration and validation sets for softmax probabilities and labels.
        """
        n = self.n
        idx = torch.cat([
            torch.ones(n, dtype=torch.bool),
            torch.zeros(smx.shape[0] - n, dtype=torch.bool)
        ])[torch.randperm(smx.shape[0])]
        return smx[idx], smx[~idx], labels[idx], labels[~idx]

    def predict(self, smx: torch.Tensor, labels: torch.Tensor, qhat=None) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Predict the conformal sets based on the method specified.
        Args:
            smx (torch.Tensor): The softmax probabilities of shape (n_samples, n_classes).
            labels (torch.Tensor): The true labels of shape (n_samples,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted sets and the corresponding labels.
        """
        if self.method == "LAC":
            return self._cp_lac(smx, labels, qhat)
        elif self.method == "APS":
            return self._cp_aps(smx, labels, qhat)
        elif self.method == "RAPS":
            return self._cp_raps(smx, labels, qhat)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _get_qhat(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute the quantile threshold for the calibration scores.
        Args:
            scores (torch.Tensor): The calibration scores of shape (n_samples,).
        Returns:
            torch.Tensor: The quantile threshold for the scores.
        """
        n = self.n
        alpha = self.alpha
        q_level = (torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n).to(self.device).type(scores.dtype)
        return torch.quantile(scores, q_level, interpolation='higher')

    def _cp_lac(self, smx: torch.Tensor, labels: torch.Tensor, qhat=None) -> tuple[torch.Tensor, torch.Tensor, float]:
        """LAC conformal prediction method.
        Args:
            smx (torch.Tensor): The softmax probabilities of shape (n_samples, n_classes).
            labels (torch.Tensor): The true labels of shape (n_samples,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted sets and the corresponding labels.
        """
        if not qhat:
            n = self.n
            cal_smx, val_smx, cal_labels, val_labels = self._split(smx, labels)
            cal_scores = 1 - cal_smx[torch.arange(n), cal_labels]
            qhat = self._get_qhat(cal_scores)
        else:
            val_smx = smx
            val_labels = labels

        pred_sets = val_smx >= (1 - qhat)
        return pred_sets, val_labels, qhat

    def _cp_aps(self, smx: torch.Tensor, labels: torch.Tensor, qhat=None) -> tuple[torch.Tensor, torch.Tensor, float]:
        """APS conformal prediction method.
        Args:
            smx (torch.Tensor): The softmax probabilities of shape (n_samples, n_classes).
            labels (torch.Tensor): The true labels of shape (n_samples,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted sets and the corresponding labels.
        """
        if not qhat:
            n = self.n
            cal_smx, val_smx, cal_labels, val_labels = self._split(smx, labels)
            val_labels, val_order = torch.sort(val_labels)
            val_smx = val_smx[val_order]

            cal_pi = torch.argsort(cal_smx, dim=1, descending=True)
            cal_srt = torch.gather(cal_smx, 1, cal_pi)
            cal_L = (cal_pi == cal_labels.view(-1, 1)).nonzero()[:, 1]
            cal_scores = cal_srt.cumsum(dim=1)[torch.arange(n), cal_L] - torch.rand(n, device=self.device) * cal_srt[torch.arange(n), cal_L]

            qhat = self._get_qhat(cal_scores)

        else:
            val_smx = smx
            val_labels = labels

        val_pi = torch.argsort(val_smx, dim=1, descending=True)
        val_srt = torch.gather(val_smx, 1, val_pi)
        val_cumsum = val_srt.cumsum(dim=1)
        temp = val_cumsum - torch.rand(val_smx.size(0), 1, device=self.device) * val_srt
        indicators = temp <= qhat
        indicators[:, 0] = True
        pred_sets = torch.gather(indicators, 1, val_pi.argsort(dim=1))
        return pred_sets, val_labels, qhat

    def _cp_raps(self, smx: torch.Tensor, labels: torch.Tensor, qhat=None) -> tuple[torch.Tensor, torch.Tensor, float]:
        """RAPS conformal prediction method.
        Args:
            smx (torch.Tensor): The softmax probabilities of shape (n_samples, n_classes).
            labels (torch.Tensor): The true labels of shape (n_samples,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted sets and the corresponding labels.
        """ 
        num_classes = smx.shape[1]
        lam_reg = self.lam_reg
        k_reg = self.k_reg
        reg_vec = torch.cat([
                torch.zeros(k_reg),
                lam_reg * torch.ones(num_classes - k_reg)
            ]).unsqueeze(0).to(self.device)
        if not qhat:
            n = self.n

            cal_smx, val_smx, cal_labels, val_labels = self._split(smx, labels)

            cal_pi = torch.argsort(cal_smx, dim=1, descending=True)
            cal_srt = torch.gather(cal_smx, 1, cal_pi) + reg_vec
            cal_L = (cal_pi == cal_labels.view(-1, 1)).nonzero()[:, 1]
            cal_scores = cal_srt.cumsum(dim=1)[torch.arange(n), cal_L] - torch.rand(n, device=self.device) * cal_srt[torch.arange(n), cal_L]

            qhat = self._get_qhat(cal_scores)
        
        else: 
            val_smx = smx
            val_labels = labels

        val_pi = torch.argsort(val_smx, dim=1, descending=True)
        val_srt = torch.gather(val_smx, 1, val_pi) + reg_vec
        val_cumsum = val_srt.cumsum(dim=1)
        temp = val_cumsum - torch.rand(val_smx.size(0), 1, device=self.device) * val_srt
        indicators = temp <= qhat
        pred_sets = torch.gather(indicators, 1, val_pi.argsort(dim=1))
        return pred_sets, val_labels, qhat
