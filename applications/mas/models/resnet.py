import math

import torch.nn as nn
import torch.nn.functional as F

from .ozan_rep_fun import ozan_rep_function, trevor_rep_function, OzanRepFunction, TrevorRepFunction

from easyfl.models.model import BaseModel

__all__ = ['resnet18',
           'resnet18_half',
           'resnet18_tripple',
           'resnet34',
           'resnet50',
           'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block, layers, widths=[64, 128, 256, 512], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetEncoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Decoder(nn.Module):
    def __init__(self, output_channels=32, num_classes=None, base_match=512):
        super(Decoder, self).__init__()

        self.output_channels = output_channels
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.upconv0 = nn.ConvTranspose2d(base_match, 256, 2, 2)
            self.bn_upconv0 = nn.BatchNorm2d(256)
            self.conv_decode0 = nn.Conv2d(256, 256, 3, padding=1)
            self.bn_decode0 = nn.BatchNorm2d(256)
            self.upconv1 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.bn_upconv1 = nn.BatchNorm2d(128)
            self.conv_decode1 = nn.Conv2d(128, 128, 3, padding=1)
            self.bn_decode1 = nn.BatchNorm2d(128)
            self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.bn_upconv2 = nn.BatchNorm2d(64)
            self.conv_decode2 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn_decode2 = nn.BatchNorm2d(64)
            self.upconv3 = nn.ConvTranspose2d(64, 48, 2, 2)
            self.bn_upconv3 = nn.BatchNorm2d(48)
            self.conv_decode3 = nn.Conv2d(48, 48, 3, padding=1)
            self.bn_decode3 = nn.BatchNorm2d(48)
            self.upconv4 = nn.ConvTranspose2d(48, 32, 2, 2)
            self.bn_upconv4 = nn.BatchNorm2d(32)
            self.conv_decode4 = nn.Conv2d(32, output_channels, 3, padding=1)

    def forward(self, representation):
        # batch_size=representation.shape[0]
        if self.num_classes is None:
            # x2 = self.conv_decode_res(representation)
            # x2 = self.bn_conv_decode_res(x2)
            # x2 = interpolate(x2,size=(256,256))

            x = self.upconv0(representation)
            x = self.bn_upconv0(x)
            x = self.relu(x)
            x = self.conv_decode0(x)
            x = self.bn_decode0(x)
            x = self.relu(x)

            x = self.upconv1(x)
            x = self.bn_upconv1(x)
            x = self.relu(x)
            x = self.conv_decode1(x)
            x = self.bn_decode1(x)
            x = self.relu(x)
            x = self.upconv2(x)
            x = self.bn_upconv2(x)
            x = self.relu(x)
            x = self.conv_decode2(x)

            x = self.bn_decode2(x)
            x = self.relu(x)
            x = self.upconv3(x)
            x = self.bn_upconv3(x)
            x = self.relu(x)
            x = self.conv_decode3(x)
            x = self.bn_decode3(x)
            x = self.relu(x)
            x = self.upconv4(x)
            x = self.bn_upconv4(x)
            # x = torch.cat([x,x2],1)
            # print(x.shape,self.static.shape)
            # x = torch.cat([x,x2,input,self.static.expand(batch_size,-1,-1,-1)],1)
            x = self.relu(x)
            x = self.conv_decode4(x)

            # z = x[:,19:22,:,:].clone()
            # y = (z).norm(2,1,True).clamp(min=1e-12)
            # print(y.shape,x[:,21:24,:,:].shape)
            # x[:,19:22,:,:]=z/y

        else:

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class ResNet(BaseModel):
    def __init__(self, block, layers, tasks=None, num_classes=None, ozan=False, size=1, **kwargs):
        super(ResNet, self).__init__()
        if size == 1:
            self.encoder = ResNetEncoder(block, layers, **kwargs)
        elif size == 2:
            self.encoder = ResNetEncoder(block, layers, [96, 192, 384, 720], **kwargs)
        elif size == 3:
            self.encoder = ResNetEncoder(block, layers, [112, 224, 448, 880], **kwargs)
        elif size == 0.5:
            self.encoder = ResNetEncoder(block, layers, [48, 96, 192, 360], **kwargs)
        self.tasks = tasks
        self.ozan = ozan
        self.task_to_decoder = {}

        if tasks is not None:
            # self.final_conv = nn.Conv2d(728,512,3,1,1)
            # self.final_conv_bn = nn.BatchNorm2d(512)
            for task in tasks:
                if task == 'segment_semantic':
                    output_channels = 18
                if task == 'depth_zbuffer':
                    output_channels = 1
                if task == 'normal':
                    output_channels = 3
                if task == 'edge_occlusion':
                    output_channels = 1
                if task == 'reshading':
                    output_channels = 3
                if task == 'keypoints2d':
                    output_channels = 1
                if task == 'edge_texture':
                    output_channels = 1
                if size == 1:
                    decoder = Decoder(output_channels)
                elif size == 2:
                    decoder = Decoder(output_channels, base_match=720)
                elif size == 3:
                    decoder = Decoder(output_channels, base_match=880)
                elif size == 0.5:
                    decoder = Decoder(output_channels, base_match=360)
                self.task_to_decoder[task] = decoder
        else:
            self.task_to_decoder['classification'] = Decoder(output_channels=0, num_classes=1000)

        self.decoders = nn.ModuleList(self.task_to_decoder.values())

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, input):
        rep = self.encoder(input)

        if self.tasks is None:
            return self.decoders[0](rep)

        # rep = self.final_conv(rep)
        # rep = self.final_conv_bn(rep)

        outputs = {'rep': rep}
        if self.ozan:
            OzanRepFunction.n = len(self.decoders)
            rep = ozan_rep_function(rep)
            for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
                outputs[task] = decoder(rep[i])
        else:
            TrevorRepFunction.n = len(self.decoders)
            rep = trevor_rep_function(rep)
            for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
                outputs[task] = decoder(rep)

        return outputs


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block=block, layers=layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   **kwargs)


def resnet18_tripple(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, size=3,
                   **kwargs)


def resnet18_half(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, size=0.5,
                   **kwargs)


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained,
                   **kwargs)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   **kwargs)


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   **kwargs)
