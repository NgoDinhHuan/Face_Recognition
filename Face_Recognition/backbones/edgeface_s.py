import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.timmfr import get_timmfrv2, replace_linear_with_lowrank_2

class EmbeddingModel(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.backbone = get_timmfrv2(model_name="edgenext_x_small", featdim=512)
        replace_linear_with_lowrank_2(self.backbone, rank_ratio=0.2)
        self.l2norm = nn.Identity()  # hoặc dùng L2Norm thủ công

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def get_model(model_path, device="cpu"):
    model = EmbeddingModel(model_path, device)
    return model.to(device).eval()
