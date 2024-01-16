import paddle
import paddle.nn as nn
from cnn_utils import *
import math
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'


class EESP(nn.Layer):
    """
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        """
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'
            ], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(
            n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13
            ): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        
        # print("A", self.k_sizes)
        # self.k_sizes = paddle.to_tensor(self.k_sizes)
        # print("B", self.k_sizes)
        # paddle.sort(x=self.k_sizes), paddle.argsort(x=self.k_sizes)
        self.spp_dw = nn.LayerList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride,
                groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(paddle.concat(x=
            output, axis=1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.shape == input.shape:
            expanded = expanded + input
        return self.module_act(expanded)


class DownSampler(nn.Layer):
    """
    Down-sampling fucntion that has three parallel branches: (1) avg pooling,
    (2) EESP block with stride of 2 and (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    """

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        """
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        """
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim,
            down_method='avg')
        self.avg = nn.AvgPool2D(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(CBR(config_inp_reinf,
                config_inp_reinf, 3, 1), CB(config_inp_reinf, nout, 1, 1))
        self.act = nn.PReLU(nout)

    def forward(self, input, input2=None):
        """
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        """
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = paddle.concat(x=[avg_out, eesp_out], axis=1)
        if input2 is not None:
            w1 = avg_out.shape[2]
            while True:
                input2 = paddle.nn.functional.avg_pool2d(kernel_size=3,
                    padding=1, stride=2, x=input2, exclusive=False)
                w2 = input2.shape[2]
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)
        return self.act(output)


class EESPNet(nn.Layer):
    """
    This class defines the ESPNetv2 architecture for the ImageNet classification
    """

    def __init__(self, classes=1000, s=1):
        """
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        """
        super().__init__()
        reps = [0, 3, 7, 3]
        channels = 3
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s <= 2.0:
            config.append(1280)
        else:
            ValueError('Configuration not supported')
        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim
            ), 'Length of branching factor array and receptive field array should be the same.'
        self.level1 = CBR(channels, config[0], 3, 2)
        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=
            r_lim[0], reinf=self.input_reinforcement)
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=
            r_lim[1], reinf=self.input_reinforcement)
        self.level3 = nn.LayerList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2],
                r_lim=r_lim[2]))
        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=
            r_lim[2], reinf=self.input_reinforcement)
        self.level4 = nn.LayerList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3],
                r_lim=r_lim[3]))
        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=
            r_lim[3])
        self.level5 = nn.LayerList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4],
                r_lim=r_lim[4]))
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))
        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        """
        Function to initialze the parameters
        """
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                paddle.nn.initializer.KaimingNormal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                init_Constant = paddle.nn.initializer.Constant(value=1)
                init_Constant(m.weight)
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
            elif isinstance(m, nn.Linear):
                init_Normal = paddle.nn.initializer.Normal(std=0.001)
                init_Normal(m.weight)
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)

    def forward(self, input, p=0.2):
        """
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        """
        out_l1 = self.level1(input)
        if not self.input_reinforcement:
            del input
            input = None
        out_l2 = self.level2_0(out_l1, input)
        out_l3_0 = self.level3_0(out_l2, input)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, input)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        out_l5_0 = self.level5_0(out_l4)
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)
        output_g = paddle.nn.functional.adaptive_avg_pool2d(x=out_l5,
            output_size=1)
        output_g = paddle.nn.functional.dropout(x=output_g, p=p, training=
            self.training)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        output_1x1 = output_g.view(output_g.shape[0], -1)
        return self.classifier(output_1x1)


if __name__ == '__main__':
    input = paddle.empty(shape=[1, 3, 224, 224])
    model = EESPNet(classes=1000, s=1.0)
    out = model(input)
    print('Output size')
    print(out.shape)
