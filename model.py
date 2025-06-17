import torch
import torch.nn as nn
from torchvision.models.video import s3d,S3D_Weights


class S3DNEW(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained_model = s3d(weights=S3D_Weights.DEFAULT)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.classifier= nn.Identity()
        self.pretrained_model.avgpool=nn.Identity()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.pretrained_model(x)
        logits = self.classifier(features)
        return logits


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = S3DNEW(pretrained=True).to(device)
#
# model.eval()
#
# B, T, H, W = 2, 24, 400, 400
# x = torch.randn(B, 3, T, H, W).to(device)
# with torch.no_grad():
#      output = model(x)
#      print(output)
#      print(output.shape)
