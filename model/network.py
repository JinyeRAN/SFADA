import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from .grl import GradientReverseFunction
from .models import register_model


class TaskNet(nn.Module):
    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    def __init__(self, num_cls=10, normalize=False, temp=0.05, botten_neck=256):
        super(TaskNet, self).__init__()
        self.vis = False
        self.num_cls = num_cls
        self.botten_neck = botten_neck
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        self.normalize = normalize
        self.temp = temp

    def forward(self, x, with_emb=False, reverse_grad=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        emb = self.fc_params(x)

        if reverse_grad: emb = GradientReverseFunction.apply(emb)
        if self.normalize: emb = F.normalize(emb) / self.temp
        score = self.classifier(emb)

        if with_emb:
            return score, emb
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def parameters_network(self, lr=0.1, lr_scalar=0.1):
        parameter_list = [
            {'params': self.conv_params.parameters(), 'lr': lr * lr_scalar},
            {'params': self.fc_params.parameters(), 'lr': lr},
            {'params': self.classifier.parameters(), 'lr': lr},
        ]
        if self.vis:
            parameter_list.append({'params': self.vis_fc.parameters(), 'lr': lr* lr_scalar},)
        return parameter_list

@register_model('ResNet34Fc')
class ResNet34Fc(TaskNet):
    num_channels = 3
    name = 'ResNet34Fc'

    def setup_net(self):
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, self.num_cls, bias=False)

class BatchNorm1d(nn.Module):
    def __init__(self, dim):
        super(BatchNorm1d, self).__init__()
        self.BatchNorm1d = nn.BatchNorm1d(dim)

    def __call__(self, x):
        if x.size(0) == 1:
            x = torch.cat((x,x), 0)
            x = self.BatchNorm1d(x)[:1]
        else:
            x = self.BatchNorm1d(x)
        return x

@register_model('ResNet50Fc')
class ResNet50Fc(TaskNet):
    num_channels = 3
    name = 'ResNet50Fc'

    def setup_net(self):
        # self.normalize = True
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Sequential(nn.Linear(2048, self.botten_neck), BatchNorm1d(self.botten_neck))
        self.classifier = nn.Linear(self.botten_neck, self.num_cls)

@register_model('SwinFormFc')
class SwinFormFc(TaskNet):
    num_channels = 3
    name = 'SwinFormFc'

    def setup_net(self):
        # self.normalize = True
        model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        inch = model.head.in_features
        model.head = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Sequential(nn.Linear(inch, self.botten_neck), BatchNorm1d(self.botten_neck))

        self.classifier = nn.Sequential(
            nn.Linear(self.botten_neck, self.botten_neck*2),
            BatchNorm1d(self.botten_neck*2),
            nn.Linear(self.botten_neck*2, self.num_cls)
        )

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

@register_model('Generator')
class Generator(nn.Module):
    def __init__(self, input_dim=100, input_size=224, num_cls=10, botten_neck=256):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.class_num = num_cls

        # label embedding
        self.label_emb = nn.Embedding(self.class_num, self.input_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.BatchNorm1d(128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.ReLU(),
        )
        ouch = botten_neck//4
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, ouch, kernel_size=3, stride=1, padding=2), # new
            nn.BatchNorm2d(ouch),# new
            nn.ReLU(),# new
        )
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.mul(self.label_emb(label), input)
        x = self.fc(x)
        x = x.view(-1, 512, (self.input_size // 32), (self.input_size // 32))
        x = self.deconv(x)
        x = x.view(x.size(0), -1)
        return x