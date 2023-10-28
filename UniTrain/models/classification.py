import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models


class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, stride=2, padding=1, kernel_size=7)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.residual_layer1 = nn.Conv2d(64, 256, padding=0, kernel_size=1)
        self.softmax = F.softmax
        self.num_classes = num_classes


    def conv_block(self, xb, inp_filter_size, hidden_filter_size, out_filter_size, pool = False):
        layers = nn.Sequential(nn.Conv2d(inp_filter_size, hidden_filter_size, padding=0, kernel_size=1), nn.BatchNorm2d(hidden_filter_size), nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_filter_size, hidden_filter_size, padding=1, kernel_size=3), nn.BatchNorm2d(hidden_filter_size), nn.ReLU(inplace=True),
                  nn.Conv2d(hidden_filter_size, out_filter_size, padding=0, kernel_size=1), nn.BatchNorm2d(out_filter_size), nn.ReLU(inplace=True))
        layers.to(xb.device)
        return layers(xb)

    def forward(self, xb):
        y = self.conv1(xb)
        y = self.maxpool(y)

        y = self.conv_block(y, 64, 64, 256)
        y = self.conv_block(y, 256, 64, 256) + y
        y = self.conv_block(y, 256, 64, 256) + y

        y = self.conv_block(y, 256, 128, 512)
        y = self.conv_block(y, 512, 128, 512) + y
        y = self.conv_block(y, 512, 128, 512) + y
        y = self.conv_block(y, 512, 128, 512) + y

        y = self.conv_block(y, 512, 256, 1024)
        for i in range(0, 22):
            y = self.conv_block(y, 1024, 256, 1024) + y
            i+=1

        y = self.conv_block(y, 1024, 512, 2048)
        y = self.conv_block(y, 2048, 512, 2048) + y
        y = self.conv_block(y, 2048, 512, 2048) + y

        y = self.avgpool(y)
        y = nn.Flatten()(y)
        y = y.reshape(y.shape[0], -1)
        linear_layer = nn.Linear(y.shape[1], self.num_classes)
        linear_layer.to(xb.device)
        y = linear_layer(y)

        return y
    

# Define the ResNet-9 model in a single class
class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super(ResNet9, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.build_residual_block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                self.build_residual_block(self.in_channels, out_channels, stride=1)
            )
        return nn.Sequential(*layers)

    def build_residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

 #ResNet50 functionality addition       
class ResNet9_50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet9_50, self).__init__()
        
        self.resnet9 = ResNet9(num_classes)
        self.resnet50 = models.resnet50(pretrained=True)

    def forward(self, x):
         x = self.resnet50(x)
#GoogLeNet functionality addition
import torch
import torch.nn as nn
import torchvision.models as models

class GoogleNetModel(nn.Module):
    def __init__(self, num_classes):
        super(GoogleNetModel, self).__init()
        
        # Load the pre-trained GoogleNet model
        self.googlenet = models.inception_v3(pretrained=True)
        
        # Modify the classification head to match the number of classes in your dataset
        num_ftrs = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Pass input through the GoogleNet model
        x1 = self.googlenet(x)
        x2 = self.resnet50(x)
        x3 = self.resnet9(x)
        x = x1 + x2 + x3
        return x


# Making a custom transfer learning model
def create_transfer_learning_model(num_classes, model = torchvision.models.resnet18, feature_extract=True, use_pretrained=True):
    """
    Create a transfer learning model with a custom output layer.
    
    Args:
        num_classes (int): Number of classes in the custom output layer.
        model(torchvision.models.<ModelName>): Pre-trained model you want to use.
        feature_extract (bool): If True, freeze the pre-trained model's weights.
        use_pretrained (bool): If True, use pre-trained weights.

    Returns:
        model: A PyTorch model ready for transfer learning.
    """
    # Load a pre-trained model, for example, ResNet-18
    model = model(pretrained=use_pretrained)
    
    # Freeze the pre-trained weights if feature_extract is True
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    
    # Modify the output layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Define the ResNet-18 model in a single class
class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.layer2 = self.make_layer(128, 4, stride=2)
        self.layer3 = self.make_layer(256, 6, stride=2)
        self.layer4 = self.make_layer(512, 3, stride=2)

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.build_residual_block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(self.build_residual_block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def build_residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.layer2 = self.make_layer(128, 4, stride=2)
        self.layer3 = self.make_layer(256, 6, stride=2)
        self.layer4 = self.make_layer(512, 3, stride=2)

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.build_residual_block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(self.build_residual_block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def build_residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.layer2 = self.make_layer(128, 4, stride=2)
        self.layer3 = self.make_layer(256, 23, stride=2)
        self.layer4 = self.make_layer(512, 3, stride=2)

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.build_residual_block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(self.build_residual_block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def build_residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=12, num_blocks=3, num_layers=4):
        super(DenseNet, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        for block in range(num_blocks):
            self.dense_blocks.append(self.make_dense_block(growth_rate, num_layers))

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def make_dense_block(self, growth_rate, num_layers):
        layers = []
        in_channels = self.in_channels
        for _ in range(num_layers):
            layers.extend([
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            ])
            in_channels += growth_rate
        self.in_channels = in_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for block in self.dense_blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

