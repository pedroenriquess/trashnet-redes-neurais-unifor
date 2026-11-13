import torch
import torch.nn as nn

class TrashMLP(nn.Module):
    def __init__(self, input_size=(128, 96), num_classes=6, dropout=0.2):
        super().__init__()
        # 3 canais RGB * largura * altura
        in_dim = 3 * input_size[0] * input_size[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # achata a imagem se vier em formato (B, C, H, W)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier(x)


class TrashCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        
        # 3 blocos conv basicos com batchnorm e relu
        self.conv_layers = nn.Sequential(
            # bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # global average pooling garante tamanho constante independente da resolucao
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
