import torch
import torch.nn.functional as F
from torch import nn

class Encoder1dCnn(nn.Module):
    """
    1D-CNN encoder, as described in "Bio-Signal Based Multimodal Fusion with Bilinear Model for Emotion Recognition".
    DOI 10.1109/BIBM58861.2023.10385273
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Conv1
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9, stride=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # Conv2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # Conv3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AvgPool1d(kernel_size=2),
            # Conv4, padding=1 in the paper, mistake?
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Conv5
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Linear Block
            nn.Flatten(),
            nn.Linear(in_features=23040, out_features=256)
        )

    def forward(self, data):
        return self.model(data)


class SinghFusionModel(nn.Module):
    """
    Fusion model as described in "Bio-Signal Based Multimodal Fusion with Bilinear Model for Emotion Recognition".
    DOI 10.1109/BIBM58861.2023.10385273
    """
    def __init__(self, n_channels_ecg=3, n_channels_gsr=1):
        """
        Initialize the fusion model from the paper.

        :param n_channels_ecg: Channels of ECG to use
        :param n_channels_gsr: Channels of GSR to use
        """
        super().__init__()

        self.ecg_encoders = nn.ModuleList(
            Encoder1dCnn() for _ in range(n_channels_ecg)
        )

        self.gsr_encoder = Encoder1dCnn()

        self.fusion = nn.Bilinear(in1_features=256 * n_channels_ecg,
                                  in2_features = 256 * n_channels_gsr,
                                  out_features=256,
                                  bias=False)

        self.fc = nn.Linear(256, 2)

    def forward(self, eye, gsr, eeg, ecg):
        # We ignore eye and EEG on purpose since they are not used in the paper
        ecgs = torch.cat([ecg_encoder(ecg[:, None, :, i]) for i, ecg_encoder in enumerate(self.ecg_encoders)], dim=1)
        gsr = self.gsr_encoder(gsr.unsqueeze(1))

        fused = self.fusion(ecgs, gsr)
        return self.fc(fused)