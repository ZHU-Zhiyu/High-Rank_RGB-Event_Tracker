import imp
import math
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from .base import Backbone
import torch


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        # self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        # TODO better way?
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes, 'layer2': inplanes*2, 'layer3': inplanes*4,
                               'layer4': inplanes*8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)
    def mask_output(self, x, Emb_x, ratio):
        
        B, C, H, W =x.shape
        patch_num = int(1/ratio)
        patch_size = int(H * ratio)
        flag = torch.rand([B,1,1,patch_num,patch_num],device = x.device) > 0.2
        flag = flag.float()
        flag = flag.repeat([1,patch_size, patch_size, 1, 1])
        flag = flag.permute([0,3,1,4,2]).reshape(B,H,W)
        flag = flag[:,None,:,:].repeat(1,C,1,1)

        flag_n = 1- flag

        flag_n = flag_n * x.mean([2,3],keepdim=True)

        x = x*flag + flag_n
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        
        B, C, H, W =x1.shape
        B, C, L =Emb_x.shape
        
        patch_size2 = int(H * ratio)
        flag = torch.rand([B,1,1,patch_num,patch_num],device = x.device) > 0.3
        flag = flag.float()
        flag = flag.repeat([1,patch_size2, patch_size2, 1, 1])
        flag = flag.permute([0,3,1,4,2]).reshape(B,H,W)
        flag = flag[:,None,:,:].repeat(1,C-4,1,1)

        Emb_x1 = Emb_x[:,4:,:]
        Emb_x = Emb_x1.reshape([B, C-4, H, W])

        Emb_x = self.bn1(Emb_x)
        Emb_x = self.relu(Emb_x)
        
        Emb_x = Emb_x * flag

        return x1, Emb_x
        

    def train_forward(self, x, output_layers=None, Emb_x = None, ratio = 0.25):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        # x1 = self.maxpool(x)

        Emb_x_out = []
        x_out = []
        # B, C, L =Emb_x.shape
        x1, Emb_x = self.mask_output(x, Emb_x, ratio)
        # B, C, H, W =x1.shape

        B, C, L,_ =Emb_x.shape

        # # patch_num = int(1/ratio)
        # # patch_size = int(H * ratio)
        # # flag = torch.rand([B,1,1,patch_num,patch_num],device = x1.device) > 0.2

        # # flag = flag.float().repeat([1,patch_sizepyt])
        # # flag2 = torch.rand(B,device = x1.device) > 0.1
        # # flag2 = flag2.float()[:,None,None,None]
        # # flag2 = 1-flag + flag*flag2

        # # B, C, H, W =x1.shape
        # Emb_x1 = Emb_x[:,4:,:]
        # Emb_x = Emb_x1.reshape([B, C-4, H, W])

        # Emb_x = self.bn1(Emb_x)
        # Emb_x = self.relu(Emb_x)
        # Emb_x = (Emb_x*flag + x1*flag2) / (flag + flag2 + 1e-3)
        flag = torch.rand(B,device = x1.device) > 0.1
        flag = flag.float()[:,None,None,None]
        flag2 = torch.rand(B,device = x1.device) > 0.1
        flag2 = flag2.float()[:,None,None,None]
        flag2 = 1-flag + flag*flag2
        Emb_x = (Emb_x*flag + x1*flag2) / (flag + flag2 + 1e-3)
        # Emb_x = (Emb_x + x1) / 2
        Emb_x1 = self.layer1(Emb_x)
        # x11 = self.layer1(x1)

        Emb_x_out.append(Emb_x)
        # x_out.append(x1)

        Emb_x_out.append(Emb_x1)
        # x_out.append(Emb_x1)

        if self._add_output_and_check('layer1', Emb_x1, outputs, output_layers):
            return outputs

        Emb_x2 = self.layer2(Emb_x1)
        # x2 = self.layer2(x11)
        Emb_x_out.append(Emb_x2)
        # x_out.append(Emb_x2)

        if self._add_output_and_check('layer2', Emb_x2, outputs, output_layers):
            return outputs

        Emb_x3 = self.layer3(Emb_x2)
        # x3 = self.layer3(x2)
        Emb_x_out.append(Emb_x3)
        # x_out.append(Emb_x3)

        if self._add_output_and_check('layer3', Emb_x3, outputs, output_layers):
            return outputs, Emb_x_out, Emb_x_out

    def forward(self, x, output_layers=None, Emb_x = None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()
        
        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x1 = self.maxpool(x)

        Emb_x_out = []
        x_out = []
        B, C, L =Emb_x.shape
        B, C, H, W =x1.shape

        B, C, L =Emb_x.shape
        B, C, H, W =x1.shape
        Emb_x1 = Emb_x[:,4:,:]
        Emb_x = Emb_x1.reshape([B, C, H, W])
        
        Emb_x = self.bn1(Emb_x)
        Emb_x = self.relu(Emb_x)
        Emb_x = (Emb_x + x1) / 2

        Emb_x1 = self.layer1(Emb_x)
        Emb_x_out.append(Emb_x1)
        if self._add_output_and_check('layer1', Emb_x1, outputs, output_layers):
            return outputs

        Emb_x2 = self.layer2(Emb_x1)
        # x2 = self.layer2(x11)
        Emb_x_out.append(Emb_x2)
        # x_out.append(x2)

        if self._add_output_and_check('layer2', Emb_x2, outputs, output_layers):
            return outputs

        Emb_x3 = self.layer3(Emb_x2)
        # x3 = self.layer3(x2)
        Emb_x_out.append(Emb_x3)
        # x_out.append(x3)

        if self._add_output_and_check('layer3', Emb_x3, outputs, output_layers):
            return outputs, Emb_x_out, Emb_x_out

        # x = self.layer4(x)

        # if self._add_output_and_check('layer4', x, outputs, output_layers):
        #     return outputs
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #
        # if self._add_output_and_check('fc', x, outputs, output_layers):
        #     return outputs
        #
        # if len(output_layers) == 1 and output_layers[0] == 'default':
        #     return x

        raise ValueError('output_layer is wrong.')


def resnet_baby(output_layers=None, pretrained=False, inplanes=16, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, inplanes=inplanes, **kwargs)

    if pretrained:
        raise NotImplementedError
    return model


def resnet18(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model