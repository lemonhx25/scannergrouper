import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.nets.utils import load_checkpoint

momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * (256 // 2), 128)  # 调整 Linear 层的输入维度
        self.num_features = 128

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if x.dim() == 4:
            x = x.squeeze(2)  # 假设输入形状为 [N, C, 1, L]，压缩第2维
        elif x.dim() == 3:
            pass  # 输入形状已经为 [N, C, L]
        else:
            print("Input tensor shape:", x.shape)  # 打印输入张量的形状
            #raise ValueError("Input tensor should have 3 or 4 dimensions")

            if only_fc:
                return self.fc1(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        print(x.shape)#torch.Size([480, 16, 128])
        batch_size = x.size(0)
        num_features = x.size(1) * x.size(2)
        out = x.view(-1,16 * (256 // 2))

        if only_feat:
            return out

        output = self.fc1(out)
        #output = self.relu(x)

        result_dict = {'logits': output, 'feat': out}
        return result_dict
        
# 5 构建网络模型
class Digit(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5) # 3:灰度图片的通道（改为彩色）， 10：输出通道， 5：kernel 5x5
        self.conv2 = nn.Conv2d(10, 10, 3) # 10:输入通道， 20：输出通道， 3：kernel 3x3
        self.fc1 = nn.Linear(160, 128) # 20*10*10:输入通道， 500：输出通道
        #self.fc2 = nn.Linear(50, 14) # 500：输入通道， 10：输出通道

        self.num_features = 128
       
    #前向传播
    def forward(self, x,**kwargs):
        input_size = x.size(0)  # batch_size
        #print('input_size',x.shape)
        x = self.conv1(x) # 输入：batch*1*28*28, 输出：batch*10*12*12 (16 - 5 + 1 = 12)
        x = F.relu(x)  # 激活函数，保持shape不变， 输出batch*10*12*12
        x = F.max_pool2d(x, 2, 2) # 输入：batch*10*12*12 输出：batch*10*6*6
        
        x = self.conv2(x) # 输入：batch*10*6*6, 输出：batch*20*4*4 (6 - 3 + 1 = 4)
        x = F.relu(x)

        x = x.view( -1,input_size,) # 拉平， -1 自动计算维度，20*4*4 = 320
        
        x = self.fc1(x) # 输入：batch*2000，输出：batch*500
        x = F.relu(x)
        
        #x = self.fc2(x) # 输入：batch*500，输出：batch*10
        
        output = F.log_softmax(x, dim=1) # 计算分类后，每个数字的概率值
        
        return output


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.num_features = channels[3]

        # rot_classifier for Remix Match
        # self.is_remix = is_remix
        # if is_remix:
        #     self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.fc(x)
        
        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)

        if only_feat:
            return out
        
        output = self.fc(out)
        result_dict = {'logits':output, 'feat':out}
        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix), blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out#,attention

from torch.nn import functional as F

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, 1)  # 全连接层用于计算注意力分数

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # 将特征图展平成向量
        x_flattened = x.view(batch_size, C, -1)  # B x C x (W*H)
        
        # 计算注意力分数
        attention_scores = self.fc(x_flattened.transpose(1, 2)).squeeze(-1)  # B x (W*H)
        
        # 使用softmax函数计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # B x (W*H)
        
        # 应用注意力权重
        attended_features = (x_flattened * attention_weights.unsqueeze(1)).sum(dim=2)  # B x C
        
        # 将注意力后的特征向量扩展回原始特征图的形状
        attended_features_expanded = attended_features.unsqueeze(2).unsqueeze(3)  # B x C x 1 x 1
        attended_features_expanded = attended_features_expanded.expand_as(x)  # B x C x W x H
        
        return attended_features_expanded
        
class WideResNet2(nn.Module):
        # 1st conv before any network block
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNet2, self).__init__()
        channels = [16, 16 * widen_factor]#, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        #self.attention = nn.Linear(3, 1)  # 定义注意力模块
        #self.attention_layer = SelfAttention(3, 3)
        #self.attention_layer = AttentionLayer(3, 3)
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate)
        # 2nd block
        #self.block2 = NetworkBlock(
        #    n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        #self.block3 = NetworkBlock(
        #    n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[1], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        

        self.fc = nn.Linear(channels[1], num_classes)
        self.channels = channels[1]
        self.num_features = channels[1]

        # rot_classifier for Remix Match
        # self.is_remix = is_remix
        # if is_remix:
        #     self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.fc(x)
            
        #attention_scores = self.attention(x.view(x.size(0), -1))  # 计算注意力分数
        #attention_weights = F.softmax(attention_scores, dim=1)  # 使用softmax函数计算注意力权重
        #attended_features = (x * attention_weights.unsqueeze(2).unsqueeze(3)).sum(dim=(2, 3))  # 加权求和特征
        #x = self.attention_layer(x)

        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        
        
        out = out.view(-1, self.channels)
        if only_feat:
            return out
        
        output = self.fc(out)
        result_dict = {'logits':output, 'feat':out}
        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        #out = self.block2(out)
        #out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix), blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd

def cnn1d_network(pretrained=False, pretrained_path=None, **kwargs):
    model = CNN1D(**kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def digit_network(pretrained=False, pretrained_path=None, **kwargs):
    model = Digit(**kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

def wrn_28_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
    
def wrn_10_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=10, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
def wrn2_10_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet2(first_stride=1, depth=10, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

def wrn2_10_1(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet2(first_stride=1, depth=10, widen_factor=1, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
    
def wrn_10_1(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=10, widen_factor=1, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
    
def wrn_5_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=5, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

def wrn_28_8(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=8, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


if __name__ == '__main__':
    model = wrn_28_2(pretrained=True, num_classes=10)
