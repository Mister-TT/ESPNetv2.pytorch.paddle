import paddle
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'
from cnn.Model import EESPNet, EESP
import os
from cnn.cnn_utils import *


class EESPNet_Seg(paddle.nn.Layer):

    def __init__(self, classes=20, s=1, pretrained=None, gpus=1):
        super().__init__()
        classificationNet = EESPNet(classes=1000, s=s)
        if gpus >= 1:
            classificationNet = paddle.DataParallel(classificationNet)
        if pretrained:
            if not os.path.isfile(pretrained):
                print(
                    'Weight file does not exist. Training without pre-trained weights'
                    )
            print('Model initialized with pretrained weights')
            classificationNet.set_state_dict(state_dict=paddle.load(path=
                pretrained))
        self.net = classificationNet.module
        del classificationNet
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        if s <= 0.5:
            p = 0.1
        else:
            p = 0.2
        self.proj_L4_C = CBR(self.net.level4[-1].module_act.num_parameters,
            self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2 * self.net.level3[-1].module_act.num_parameters
        self.pspMod = paddle.nn.Sequential(EESP(pspSize, pspSize // 2,
            stride=1, k=4, r_lim=7), PSPModule(pspSize // 2, pspSize // 2))
        self.project_l3 = paddle.nn.Sequential(paddle.nn.Dropout2D(p=p), C(
            pspSize // 2, classes, 1, 1))
        self.act_l3 = BR(classes)
        self.project_l2 = CBR(self.net.level2_0.act.num_parameters +
            classes, classes, 1, 1)
        self.project_l1 = paddle.nn.Sequential(paddle.nn.Dropout2D(p=p), C(
            self.net.level1.act.num_parameters + classes, classes, 1, 1))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = paddle.nn.functional.interpolate(x=x, scale_factor=2, mode=
                'bilinear', align_corners=True)
        return x

    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = paddle.nn.functional.interpolate(x=out_l4_proj,
            scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(paddle.concat(x=[out_l3, up_l4_to_l3],
            axis=1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = paddle.nn.functional.interpolate(x=proj_merge_l3,
            scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(paddle.concat(x=[out_l2, out_up_l3], axis=1)
            )
        out_up_l2 = paddle.nn.functional.interpolate(x=merge_l2,
            scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(paddle.concat(x=[out_l1, out_up_l2], axis=1)
            )
        if self.training:
            return paddle.nn.functional.interpolate(x=merge_l1,
                scale_factor=2, mode='bilinear', align_corners=True
                ), self.hierarchicalUpsample(proj_merge_l3_bef_act)
        else:
            return paddle.nn.functional.interpolate(x=merge_l1,
                scale_factor=2, mode='bilinear', align_corners=True)


if __name__ == '__main__':
    input = paddle.empty(shape=[1, 3, 512, 1024])
    net = EESPNet_Seg(classes=20, s=2)
    out_x_8 = net(input)
    print(out_x_8.shape)
