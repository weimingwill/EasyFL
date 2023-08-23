""" 
Creates an Xception Model as defined in:
Xception: Deep Learning with Depthwise Separable Convolutions, https://arxiv.org/pdf/1610.02357.pdf
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easyfl.models.model import BaseModel
from .ozan_rep_fun import ozan_rep_function, OzanRepFunction, gradnorm_rep_function, GradNormRepFunction, \
    trevor_rep_function, TrevorRepFunction

__all__ = ['xception',
           'xception_gradnorm',
           'xception_half_gradnorm',
           'xception_ozan',
           'xception_half',
           'xception_quad',
           'xception_double',
           'xception_double_ozan',
           'xception_half_ozan',
           'xception_quad_ozan']


# model_urls = {
#     'xception_taskonomy':'file:///home/tstand/Dropbox/taskonomy/xception_taskonomy-a4b32ef7.pth.tar'
# }


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 groupsize=1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=max(1, in_channels // groupsize), bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            # rep.append(nn.AvgPool2d(3,strides,1))
            rep.append(nn.Conv2d(filters, filters, 2, 2))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(24, 48, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        # do relu here

        self.block1 = Block(48, 96, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(96, 192, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(192, 512, 2, 2, start_with_relu=True, grow_first=True)

        # self.block4=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block5=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block6=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block7=Block(768,768,3,1,start_with_relu=True,grow_first=True)

        self.block8 = Block(512, 512, 2, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(512, 512, 2, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(512, 512, 2, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(512, 512, 2, 1, start_with_relu=True, grow_first=True)

        # self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(512, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        # self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        # self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        # self.bn4 = nn.BatchNorm2d(2048)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)

        representation = self.relu2(x)

        return representation


class EncoderHalf(nn.Module):
    def __init__(self):
        super(EncoderHalf, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(24, 48, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        # do relu here

        self.block1 = Block(48, 64, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(64, 128, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(128, 360, 2, 2, start_with_relu=True, grow_first=True)

        # self.block4=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block5=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block6=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block7=Block(768,768,3,1,start_with_relu=True,grow_first=True)

        self.block8 = Block(360, 360, 2, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(360, 360, 2, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(360, 360, 2, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(360, 360, 2, 1, start_with_relu=True, grow_first=True)

        # self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(360, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        # self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        # self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        # self.bn4 = nn.BatchNorm2d(2048)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)

        representation = self.relu2(x)

        return representation


class EncoderQuad(nn.Module):
    def __init__(self):
        super(EncoderQuad, self).__init__()
        print('entering quad constructor')
        self.conv1 = nn.Conv2d(3, 48, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(48, 96, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        # do relu here

        self.block1 = Block(96, 192, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(192, 384, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(384, 1024, 2, 2, start_with_relu=True, grow_first=True)

        # self.block4=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block5=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block6=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block7=Block(768,768,3,1,start_with_relu=True,grow_first=True)

        self.block8 = Block(1024, 1024, 2, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(1024, 1024, 2, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(1024, 1024, 2, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(1024, 1024, 2, 1, start_with_relu=True, grow_first=True)

        # self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        # self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        # self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        # self.bn4 = nn.BatchNorm2d(2048)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)

        representation = self.relu2(x)

        return representation


class EncoderDouble(nn.Module):
    def __init__(self):
        super(EncoderDouble, self).__init__()
        print('entering double constructor')
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # self.block4=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block5=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block6=Block(768,768,3,1,start_with_relu=True,grow_first=True)
        # self.block7=Block(768,768,3,1,start_with_relu=True,grow_first=True)

        self.block8 = Block(728, 728, 2, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 2, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 2, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 2, 1, start_with_relu=True, grow_first=True)

        # self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(728, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        # self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        # self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        # self.bn4 = nn.BatchNorm2d(2048)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)

        representation = self.relu2(x)

        return representation


def interpolate(inp, size):
    t = inp.type()
    inp = inp.float()
    out = nn.functional.interpolate(inp, size=size, mode='bilinear', align_corners=False)
    if out.type() != t:
        out = out.half()
    return out


class Decoder(nn.Module):
    def __init__(self, output_channels=32, num_classes=None):
        super(Decoder, self).__init__()

        self.output_channels = output_channels
        self.num_classes = num_classes

        if num_classes is not None:
            self.fc = nn.Linear(256, num_classes)
        # else:
        #    self.fc = nn.Linear(256, 1000)
        else:
            self.relu = nn.ReLU(inplace=True)

            self.conv_decode_res = SeparableConv2d(256, 16, 3, padding=1)
            self.conv_decode_res2 = SeparableConv2d(256, 96, 3, padding=1)
            self.bn_conv_decode_res = nn.BatchNorm2d(16)
            self.bn_conv_decode_res2 = nn.BatchNorm2d(96)
            self.upconv1 = nn.ConvTranspose2d(96, 96, 2, 2)
            self.bn_upconv1 = nn.BatchNorm2d(96)
            self.conv_decode1 = SeparableConv2d(96, 64, 3, padding=1)
            self.bn_decode1 = nn.BatchNorm2d(64)
            self.upconv2 = nn.ConvTranspose2d(64, 64, 2, 2)
            self.bn_upconv2 = nn.BatchNorm2d(64)
            self.conv_decode2 = SeparableConv2d(64, 64, 5, padding=2)
            self.bn_decode2 = nn.BatchNorm2d(64)
            self.upconv3 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.bn_upconv3 = nn.BatchNorm2d(32)
            self.conv_decode3 = SeparableConv2d(32, 32, 5, padding=2)
            self.bn_decode3 = nn.BatchNorm2d(32)
            self.upconv4 = nn.ConvTranspose2d(32, 32, 2, 2)
            self.bn_upconv4 = nn.BatchNorm2d(32)
            self.conv_decode4 = SeparableConv2d(48, output_channels, 5, padding=2)

    def forward(self, representation):
        # batch_size=representation.shape[0]
        if self.num_classes is None:
            x2 = self.conv_decode_res(representation)
            x2 = self.bn_conv_decode_res(x2)
            x2 = interpolate(x2, size=(256, 256))
            x = self.conv_decode_res2(representation)
            x = self.bn_conv_decode_res2(x)
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
            x = torch.cat([x, x2], 1)
            # print(x.shape,self.static.shape)
            # x = torch.cat([x,x2,input,self.static.expand(batch_size,-1,-1,-1)],1)
            x = self.relu(x)
            x = self.conv_decode4(x)

            # z = x[:,19:22,:,:].clone()
            # y = (z).norm(2,1,True).clamp(min=1e-12)
            # print(y.shape,x[:,21:24,:,:].shape)
            # x[:,19:22,:,:]=z/y

        else:
            # print(representation.shape)
            x = F.adaptive_avg_pool2d(representation, (1, 1))
            x = x.view(x.size(0), -1)
            # print(x.shape)
            x = self.fc(x)
            # print(x.shape)
        return x


class Xception(BaseModel):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, tasks=None, num_classes=None, ozan=False, half=False):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        print('half is', half)
        if half == 'Quad':
            print('running quad code')
            self.encoder = EncoderQuad()
        elif half == 'Double':
            self.encoder = EncoderDouble()
        elif half:
            self.encoder = EncoderHalf()
        else:
            self.encoder = Encoder()
        self.tasks = tasks
        self.ozan = ozan
        self.task_to_decoder = {}
        self.task_to_output_channels = {
            'segment_semantic': 18,
            'depth_zbuffer': 1,
            'normal': 3,
            'normal2': 3,
            'edge_occlusion': 1,
            'reshading': 1,
            'keypoints2d': 1,
            'edge_texture': 1,
            'principal_curvature': 2,
            'rgb': 3,
        }
        if tasks is not None:
            for task in tasks:
                output_channels = self.task_to_output_channels[task]
                decoder = Decoder(output_channels, num_classes)
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

    count = 0

    def input_per_task_losses(self, losses):
        # if GradNormRepFunction.inital_task_losses is None:
        #     GradNormRepFunction.inital_task_losses=losses
        #     GradNormRepFunction.current_weights=[1 for i in losses]
        Xception.count += 1
        if Xception.count < 200:
            GradNormRepFunction.inital_task_losses = losses
            GradNormRepFunction.current_weights = [1 for i in losses]
        elif Xception.count % 20 == 0:
            with open("gradnorm_weights.txt", "a") as myfile:
                myfile.write(str(Xception.count) + ': ' + str(GradNormRepFunction.current_weights) + '\n')
        GradNormRepFunction.current_task_losses = losses

    def forward(self, input):
        rep = self.encoder(input)

        if self.tasks is None:
            return self.decoders[0](rep)

        outputs = {'rep': rep}
        if self.ozan == 'gradnorm':
            GradNormRepFunction.n = len(self.decoders)
            rep = gradnorm_rep_function(rep)
            for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
                outputs[task] = decoder(rep[i])
        elif self.ozan:
            OzanRepFunction.n = len(self.decoders)
            rep = ozan_rep_function(rep)
            for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
                outputs[task] = decoder(rep[i])
        else:
            TrevorRepFunction.n = len(self.decoders)
            rep = trevor_rep_function(rep)
            for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
                outputs[task] = decoder(rep)
            # Original loss
            # for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
            #     outputs[task] = decoder(rep)

        return outputs


def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    model = Xception(**kwargs)

    if pretrained:
        # state_dict = model_zoo.load_url(model_urls['xception_taskonomy'])
        # for name,weight in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name]=weight.unsqueeze(-1).unsqueeze(-1)
        #     if 'conv1' in name and len(weight.shape)!=4:
        #         state_dict[name]=weight.unsqueeze(1)
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))
        # if num_classes !=1000:
        #     model.fc = nn.Linear(2048, num_classes)
        # import torch
        # print("writing new state dict")
        # torch.save(model.state_dict(),"xception.pth.tar")
        # print("done")
        # import sys
        # sys.exit(1)

    return model


def xception_ozan(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(ozan=True, **kwargs)

    if pretrained:
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))

    return model


def xception_gradnorm(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(ozan='gradnorm', **kwargs)

    if pretrained:
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))

    return model

def xception_half_gradnorm(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(half=True, ozan='gradnorm', **kwargs)

    return model

def xception_half(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    # try:
    #     num_classes = kwargs['num_classes']
    # except:
    #     num_classes=1000
    # if pretrained:
    #     kwargs['num_classes']=1000
    model = Xception(half=True, **kwargs)

    if pretrained:
        # state_dict = model_zoo.load_url(model_urls['xception_taskonomy'])
        # for name,weight in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name]=weight.unsqueeze(-1).unsqueeze(-1)
        #     if 'conv1' in name and len(weight.shape)!=4:
        #         state_dict[name]=weight.unsqueeze(1)
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))
        # if num_classes !=1000:
        #     model.fc = nn.Linear(2048, num_classes)
        # import torch
        # print("writing new state dict")
        # torch.save(model.state_dict(),"xception.pth.tar")
        # print("done")
        # import sys
        # sys.exit(1)

    return model


def xception_quad(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    # try:
    #     num_classes = kwargs['num_classes']
    # except:
    #     num_classes=1000
    # if pretrained:
    #     kwargs['num_classes']=1000
    print('got quad')
    model = Xception(half='Quad', **kwargs)

    if pretrained:
        # state_dict = model_zoo.load_url(model_urls['xception_taskonomy'])
        # for name,weight in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name]=weight.unsqueeze(-1).unsqueeze(-1)
        #     if 'conv1' in name and len(weight.shape)!=4:
        #         state_dict[name]=weight.unsqueeze(1)
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))
        # if num_classes !=1000:
        #     model.fc = nn.Linear(2048, num_classes)
        # import torch
        # print("writing new state dict")
        # torch.save(model.state_dict(),"xception.pth.tar")
        # print("done")
        # import sys
        # sys.exit(1)

    return model


def xception_double(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    # try:
    #     num_classes = kwargs['num_classes']
    # except:
    #     num_classes=1000
    # if pretrained:
    #     kwargs['num_classes']=1000
    print('got double')
    model = Xception(half='Double', **kwargs)

    if pretrained:
        # state_dict = model_zoo.load_url(model_urls['xception_taskonomy'])
        # for name,weight in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name]=weight.unsqueeze(-1).unsqueeze(-1)
        #     if 'conv1' in name and len(weight.shape)!=4:
        #         state_dict[name]=weight.unsqueeze(1)
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))
        # if num_classes !=1000:
        #     model.fc = nn.Linear(2048, num_classes)
        # import torch
        # print("writing new state dict")
        # torch.save(model.state_dict(),"xception.pth.tar")
        # print("done")
        # import sys
        # sys.exit(1)

    return model


def xception_quad_ozan(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    # try:
    #     num_classes = kwargs['num_classes']
    # except:
    #     num_classes=1000
    # if pretrained:
    #     kwargs['num_classes']=1000
    print('got quad ozan')
    model = Xception(ozan=True, half='Quad', **kwargs)

    if pretrained:
        # state_dict = model_zoo.load_url(model_urls['xception_taskonomy'])
        # for name,weight in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name]=weight.unsqueeze(-1).unsqueeze(-1)
        #     if 'conv1' in name and len(weight.shape)!=4:
        #         state_dict[name]=weight.unsqueeze(1)
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))
        # if num_classes !=1000:
        #     model.fc = nn.Linear(2048, num_classes)
        # import torch
        # print("writing new state dict")
        # torch.save(model.state_dict(),"xception.pth.tar")
        # print("done")
        # import sys
        # sys.exit(1)

    return model


def xception_double_ozan(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    # try:
    #     num_classes = kwargs['num_classes']
    # except:
    #     num_classes=1000
    # if pretrained:
    #     kwargs['num_classes']=1000
    print('got double')
    model = Xception(ozan=True, half='Double', **kwargs)

    if pretrained:
        # state_dict = model_zoo.load_url(model_urls['xception_taskonomy'])
        # for name,weight in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name]=weight.unsqueeze(-1).unsqueeze(-1)
        #     if 'conv1' in name and len(weight.shape)!=4:
        #         state_dict[name]=weight.unsqueeze(1)
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))
        # if num_classes !=1000:
        #     model.fc = nn.Linear(2048, num_classes)
        # import torch
        # print("writing new state dict")
        # torch.save(model.state_dict(),"xception.pth.tar")
        # print("done")
        # import sys
        # sys.exit(1)

    return model


def xception_half_ozan(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(ozan=True, half=True, **kwargs)

    if pretrained:
        # model.load_state_dict(torch.load('xception_taskonomy_small_imagenet_pretrained.pth.tar'))
        model.encoder.load_state_dict(torch.load('xception_taskonomy_small2.encoder.pth.tar'))

    return model
