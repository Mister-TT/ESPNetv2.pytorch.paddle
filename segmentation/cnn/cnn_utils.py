import paddle
__author__ = 'Sachin Mehta'
__version__ = '1.0.1'
__maintainer__ = 'Sachin Mehta'


class PSPModule(paddle.nn.Layer):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = paddle.nn.LayerList(sublayers=[C(features, features, 
            3, 1, groups=features) for size in sizes])
        self.project = CBR(features * (len(sizes) + 1), out_features, 1, 1)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        out = [feats]
        for stage in self.stages:
            feats = paddle.nn.functional.avg_pool2d(kernel_size=3, stride=2,
                padding=1, x=feats, exclusive=False)
            upsampled = paddle.nn.functional.interpolate(x=stage(feats),
                size=(h, w), mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(paddle.concat(x=out, axis=1))


class CBR(paddle.nn.Layer):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = paddle.nn.Conv2D(in_channels=nIn, out_channels=nOut,
            kernel_size=kSize, stride=stride, padding=padding, bias_attr=
            False, groups=groups)
        self.bn = paddle.nn.BatchNorm2D(num_features=nOut)
        self.act = paddle.nn.PReLU(num_parameters=nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(paddle.nn.Layer):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = paddle.nn.BatchNorm2D(num_features=nOut)
        self.act = paddle.nn.PReLU(num_parameters=nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(paddle.nn.Layer):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = paddle.nn.Conv2D(in_channels=nIn, out_channels=nOut,
            kernel_size=kSize, stride=stride, padding=padding, bias_attr=
            False, groups=groups)
        self.bn = paddle.nn.BatchNorm2D(num_features=nOut)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(paddle.nn.Layer):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = paddle.nn.Conv2D(in_channels=nIn, out_channels=nOut,
            kernel_size=kSize, stride=stride, padding=padding, bias_attr=
            False, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(paddle.nn.Layer):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = paddle.nn.Conv2D(in_channels=nIn, out_channels=nOut,
            kernel_size=kSize, stride=stride, padding=padding, bias_attr=
            False, dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilatedB(paddle.nn.Layer):
    """
    This class defines the dilated convolution with batch normalization.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = paddle.nn.Conv2D(in_channels=nIn, out_channels=nOut,
            kernel_size=kSize, stride=stride, padding=padding, bias_attr=
            False, dilation=d, groups=groups)
        self.bn = paddle.nn.BatchNorm2D(num_features=nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        return self.bn(self.conv(input))
