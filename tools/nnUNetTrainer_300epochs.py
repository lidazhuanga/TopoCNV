import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_300epochs(nnUNetTrainer):
    """nnUNetTrainer with a 300-epoch budget (CNV SOTA protocol). Copied into the venv's
    nnunetv2/training/nnUNetTrainer/variants/training_length/ by the CNV setup; not part of
    upstream nnU-Net v2 2.5.2, which has no 300-epoch variant."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
