import torch
import torch.nn as nn

""" 
Tuple (filters, kernel_size, stride) 
"B": residual block + repeats (loops, as per paper)
"S": scale prediction block + yolo loss
"U": upsampling and concatenating with a previous layer ><
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) # since bias has no effect when applying batch norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


#The use of residual blocks in YOLOv3 helps improve the flow of gradients during training, enabling the network to effectively 
# learn from the data and improve performance, especially in the case of deeper networks where vanishing gradients and degradation 
# can occur.

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1), # This basically performs dimensionality reduction followed by feature extraction 
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x

# Output heads from YOLOv3
# bn_act is batch norm + activation function. No bias is needed / has any effect if they are active.
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2) # reordering of dimensions


# Full model
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._implement_architecture()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = [] #$ VERY IMPORTANT FOR SKIP CONNECTIONS
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8: # According to the paper, after every residual block there is a skip connection from a conv before to a conv after
                route_connections.append(x) # since we need to look back, store connection in route_connections

            elif isinstance(layer, nn.Upsample): # if we get to the undersampling part
                x = torch.cat([x, route_connections[-1]], dim=1) # link the connection
                route_connections.pop() # remove the connection (alr linked)

        return outputs

    def _implement_architecture(self): # method to create model layers from model architecture
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple): # (256, 3, 1)
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list): # ["B", 4]
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str): # "S"
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U": # Upsampling
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3
        return layers

if __name__ == "__main__":
    num_classes = 3
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x) # yolov3 outputs 3 heads since it is designed to capture features more efficiently in small and larger images alike
    # This is made possible thanks to the output heads before every upsampling. Smaller images (13 * 13) have larger anchor boxes
    # and larger images (26 * 26, 52 * 52) have smaller anchor boxes.
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")