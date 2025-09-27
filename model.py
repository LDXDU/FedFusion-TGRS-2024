import torch
from torch import nn
import torch.distributed as dist
from model_utils import svd_gather, gather_trans

class Cross_Fusion_Module(nn.Module):
    def __init__(self, cin, cout):
        super(Cross_Fusion_Module, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv = nn.Conv2d(kernel_size=1, in_channels=cin, out_channels=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1_a = self.relu(self.bn(self.conv(x1)))
        x2_a = self.relu(self.bn(self.conv(x2)))
        x1_o = self.max_pool(x1_a)
        x2_o = self.max_pool(x2_a)
        return x1_o, x2_o


class Fed_Fusion(nn.Module):

    def __init__(self, input_channels1, input_channels2, n_classes):
        super(Fed_Fusion, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.input_channels1 = input_channels1  
        self.input_channels2 = input_channels2

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1_a = nn.Conv2d(input_channels1, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])
        self.conv3_a = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])

        self.conv1_b = nn.Conv2d(input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])
        self.conv3_b = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])

        self.cross_a = Cross_Fusion_Module(cin=filters[2], cout=filters[3])
        self.cross_b = Cross_Fusion_Module(cin=filters[2], cout=filters[3])

        self.conv5 = nn.Conv2d(filters[3] + filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, with_svd = True, k = 4):

        x1 = x1.reshape(-1, self.input_channels1, 7, 7).contiguous()
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))

        x2 = x2.reshape(-1, self.input_channels2, 7, 7).contiguous()
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))
        
        if with_svd:
            t_x1 = svd_gather(x1, k)
            t_x2 = svd_gather(x2, k)
        else:
            t_x1 = gather_trans(x1)
            t_x2 = gather_trans(x2)

        if dist.get_rank() == 0:
            x1 = x1
            x2 = t_x2[1]
        elif dist.get_rank() == 1:
            x1 = t_x1[0]
            x2 = x2

        x11, x21 = self.cross_a(x1, x2)
        x22, x12 = self.cross_b(x2, x1)

        joint_encoder_layer1 = torch.cat([x11 + x21, x22 + x12], 1)
        joint_encoder_layer2 = torch.cat([x11, x12], 1)
        joint_encoder_layer3 = torch.cat([x22, x21], 1)

        fusion1 = self.activation(self.bn5(self.conv5(joint_encoder_layer1)))
        fusion1 = self.activation(self.bn6(self.conv6(fusion1)))
        fusion1 = self.avg_pool(fusion1)
        fusion1 = self.conv7(fusion1)

        fusion2 = self.activation(self.bn5(self.conv5(joint_encoder_layer2)))
        fusion2 = self.activation(self.bn6(self.conv6(fusion2)))
        fusion2 = self.avg_pool(fusion2)
        fusion2 = self.conv7(fusion2)

        fusion3 = self.activation(self.bn5(self.conv5(joint_encoder_layer3)))
        fusion3 = self.activation(self.bn6(self.conv6(fusion3)))
        fusion3 = self.avg_pool(fusion3)
        fusion3 = self.conv7(fusion3)

        fusion1 = torch.squeeze(fusion1) 
        fusion2 = torch.squeeze(fusion2) 
        fusion3 = torch.squeeze(fusion3) 

        l2_loss = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                l2_loss += torch.norm(m.weight, p=2)

        return fusion1, fusion2, fusion3, l2_loss

class Fed_Fusion_Loss(nn.Module):
    def __init__(self, weight):
        super(Fed_Fusion_Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, batch_y, beta_reg = 0.01):
        j1, j2, j3, l2_loss = outputs
        loss = self.ce(j1, batch_y)
        loss += 1 * torch.mean(torch.pow(j2 - j1, 2)) + 1 * torch.mean(torch.pow(j3 - j1, 2)) + beta_reg * l2_loss
        return loss