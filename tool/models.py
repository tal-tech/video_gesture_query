import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
from collections import OrderedDict
import shutil

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neek = Neck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output



def calc_pr_value():

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    CLASS_NUM = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/Yolov4_epoch21.pth'
    test_data_txt = '/workspace/hand/data/gesture_5-30/gesture_c19_shuffle_yolo_style_test.txt'
    test_data_path = '/workspace/hand/data/gesture_5-30/images'
    save_path = '/workspace/hand/data/gesture_5-30/test_results'




    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 


    print('img_names====',len(img_names))


    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 
    img_label={}
    for line in lines:
        line_list = line.strip().split()
        num = len(line_list[1:]) 
        
        img_label[line_list[0]] = []
        
        for i in range(num):
            info = line_list[i + 1].split(',')
            box = [int(info[0]), int(info[1]), int(info[2]), int(info[3])]
            box_label = int(info[4])
            img_label[line_list[0]].append({'box': box, 'box_label': box_label})

    

    model = Yolov4(yolov4conv137weight=None, n_classes=CLASS_NUM, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()


    #最后一个作为背景
    confuse_matrix = np.zeros((CLASS_NUM+1, CLASS_NUM+1), dtype=np.int)

    for i, imgfile in enumerate(img_names):
        
        if len(img_label[imgfile]) ==2:
            print('====2 boxes...continue...', i, imgfile)
            # continue


        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))


        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        boxes = do_detect(model, sized, 0.3, 0.6, use_cuda)

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(img, boxes[0], os.path.join(save_path, imgfile), class_names)



        
        if len(boxes[0]) == 0 :
             
            #没有检测到  认为不全
            continue
        else:
            predict_box = boxes[0][0]
            if len(boxes[0]) > 1:#检测到多个框，取最大置信度的框
                for box in boxes[0]:
                    if box[5] > predict_box[5]:
                        predict_box = box
            
            predict_class = predict_box[6]

        gt= img_label[imgfile][0]['box_label']
        confuse_matrix[gt][predict_class] += 1

    
    sum_column = []
    for i,row  in enumerate(confuse_matrix):
        sum_column.append(sum(row))
    confuse_matrix = np.insert(confuse_matrix, len(confuse_matrix[0]), sum_column, axis=1)
    sum_row = []
    for col_index in range(len(confuse_matrix[0])):
        sum_row.append(sum(confuse_matrix[:,col_index]))
    confuse_matrix = np.insert(confuse_matrix, len(confuse_matrix), sum_row, axis= 0)

    print(confuse_matrix)
    print('======================p=r===================')
    p_list = [.0] * CLASS_NUM
    r_list = [.0] * CLASS_NUM

    for i in range(CLASS_NUM):
        p_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[CLASS_NUM][i]+0.0000001)
        r_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[i][CLASS_NUM]+0.0000001)
        print(i,'p: {:.4f} r:{:.4f}'.format(p_list[i],r_list[i]))
    
    print('====================avg total p=r=============')
    correct_sum = 0
    pred_sum = 0
    label_sum = 0
    for i in range(CLASS_NUM):
        label_sum += confuse_matrix[i][CLASS_NUM]
        pred_sum += confuse_matrix[CLASS_NUM][i]
        correct_sum += confuse_matrix[i][i]
    print('correct_sum:',correct_sum, 'pred_sum:',pred_sum, 'label_sum:', label_sum)
    print('p: {:.4f} r:{:.4f}'.format(correct_sum*1.0 / pred_sum , correct_sum*1.0 / label_sum ))



def calc_pr_value_new():

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect
    from tool.utils_iou  import bb_intersection_over_union

    CLASS_NUM = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/Yolov4_epoch31.pth'
    test_data_txt = '/workspace/hand/data/gesture_5-30/gesture_c19_shuffle_yolo_style_test.txt'
    test_data_path = '/workspace/hand/data/gesture_5-30/images'
    save_path = '/workspace/hand/data/gesture_5-30/test_results'




    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 


    print('img_names====',len(img_names))


    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 
    img_label={}
    for line in lines:
        line_list = line.strip().split()
        num = len(line_list[1:]) 
        
        img_label[line_list[0]] = []
        
        for i in range(num):
            info = line_list[i + 1].split(',')
            box = [int(info[0]), int(info[1]), int(info[2]), int(info[3])]
            box_label = int(info[4])
            img_label[line_list[0]].append({'box': box, 'box_label': box_label})

    

    model = Yolov4(yolov4conv137weight=None, n_classes=CLASS_NUM, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    IOU_THRESH = 0.5
    #多加1行为bg，多加两列分别为bg和ground_truth
    BG_CLASS = CLASS_NUM 
    GT_INDEX = CLASS_NUM + 1
    confuse_matrix = np.zeros((CLASS_NUM+1, CLASS_NUM+2), dtype=np.int)

    for i, imgfile in enumerate(img_names):
        
        if len(img_label[imgfile]) ==2:
            print('====2 boxes...continue...', i, imgfile)
            #continue
        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))


        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        boxes = do_detect(model, sized, 0.3, IOU_THRESH, use_cuda)

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(img, boxes[0], os.path.join(save_path, imgfile), class_names)


        #记录ground truth box是否已经被匹配
        gt_matched_list = []
        pb_matched_list = []
        
        
        for pb_i, predict_box in enumerate(boxes[0]):
            predict_class = predict_box[6]

            for gt_i, gt_box_dict in enumerate(img_label[imgfile]):
                gt_box = gt_box_dict["box"]

                #已经被正确匹配过了，
                if gt_i in gt_matched_list:
                    continue
                iou = bb_intersection_over_union(predict_box, gt_box)
                print('====iou,',iou, predict_box, gt_box)
                if iou > IOU_THRESH:
                    pb_matched_list.append(pb_i)
                    
                    gt_class = gt_box_dict['box_label']
                    if gt_class == predict_class:
                        gt_matched_list.append(gt_i)
                    
                    confuse_matrix[gt_class][predict_class] += 1
                    

            #该框没有与gt框匹配，属于多检或者iou<0.5
            if pb_i not in pb_matched_list:
                confuse_matrix[BG_CLASS][predict_class] += 1

        
        for gt_i, gt_box_dict in enumerate(img_label[imgfile]):
            gt_class = gt_box_dict['box_label']
            confuse_matrix[gt_class][GT_INDEX] += 1
            #没被正确匹配过
            if gt_i not in gt_matched_list:
                confuse_matrix[gt_class][BG_CLASS] += 1

    sum_row = []
    for col_index in range(len(confuse_matrix[0])):
        sum_row.append(sum(confuse_matrix[:,col_index]))
    confuse_matrix = np.insert(confuse_matrix, len(confuse_matrix), sum_row, axis= 0)

    print(confuse_matrix)
    print('======================p=r===================')
    p_list = [.0] * CLASS_NUM
    r_list = [.0] * CLASS_NUM

    for i in range(CLASS_NUM):
        p_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[CLASS_NUM+1][i]+0.0000001)
        r_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[i][CLASS_NUM+2-1]+0.0000001)
        print(i+1,'p: {:.4f} r:{:.4f}'.format(p_list[i],r_list[i]))
    
    print('====================avg total p=r=============')
    correct_sum = 0
    pred_sum = 0
    label_sum = 0
    for i in range(CLASS_NUM):
        label_sum += confuse_matrix[i][GT_INDEX]
        pred_sum += confuse_matrix[CLASS_NUM+1][i]
        correct_sum += confuse_matrix[i][i]
    print('correct_sum:',correct_sum, 'pred_sum:',pred_sum, 'label_sum:', label_sum)
    print('p: {:.4f} r:{:.4f}'.format(correct_sum*1.0 / pred_sum , correct_sum*1.0 / label_sum ))

#数据处理方式不一样,按比例resize
def calc_pr_value_new_new_model():

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect
    from tool.utils_iou  import bb_intersection_over_union

    CLASS_NUM = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/1017_data_preprocess_resume/Yolov4_epoch_25.pth'
    test_data_txt = '/workspace/hand/data/gesture_5-30/gesture_c19_shuffle_yolo_style_test.txt'
    test_data_path = '/workspace/hand/data/gesture_5-30/images'
    save_path = '/workspace/hand/data/gesture_5-30/test_results'




    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 


    print('img_names====',len(img_names))


    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 
    img_label={}
    for line in lines:
        line_list = line.strip().split()
        num = len(line_list[1:]) 
        
        img_label[line_list[0]] = []
        
        for i in range(num):
            info = line_list[i + 1].split(',')
            box = [int(info[0]), int(info[1]), int(info[2]), int(info[3])]
            box_label = int(info[4])
            img_label[line_list[0]].append({'box': box, 'box_label': box_label})

    

    model = Yolov4(yolov4conv137weight=None, n_classes=CLASS_NUM, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    IOU_THRESH = 0.5
    #多加1行为bg，多加两列分别为bg和ground_truth
    BG_CLASS = CLASS_NUM 
    GT_INDEX = CLASS_NUM + 1
    confuse_matrix = np.zeros((CLASS_NUM+1, CLASS_NUM+2), dtype=np.int)

    for i, imgfile in enumerate(img_names):
        
        if len(img_label[imgfile]) ==2:
            print('====2 boxes...continue...', i, imgfile)
            #continue
        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))



        img = cv2.imread(os.path.join(test_data_path, imgfile))

        h,w,c = img.shape
        if h > w:
            padding_img = np.ones([h, h, 3])
            padding_img[:, :, ] = np.mean(img, axis=(0, 1))
            padding_img[:h, (h - w)// 2 : (w+(h - w)// 2)] = img[:h, :w] 

        elif h < w:
            padding_img = np.ones([w, w, 3])
            padding_img[:, :, ] = np.mean(img, axis=(0, 1))
            padding_img[(w - h)// 2 : (h+(w - h)// 2)] = img[:h, :w] 

        
        padding_img[:h, :w] = img[:h, :w] 



        sized = cv2.resize(padding_img, (width, height))


        boxes = do_detect(model, sized, 0.3, IOU_THRESH, use_cuda)

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(img, boxes[0], os.path.join(save_path, imgfile), class_names)


        #记录ground truth box是否已经被匹配
        gt_matched_list = []
        pb_matched_list = []
        
        
        for pb_i, predict_box in enumerate(boxes[0]):
            predict_class = predict_box[6]

            for gt_i, gt_box_dict in enumerate(img_label[imgfile]):
                gt_box = gt_box_dict["box"]

                #已经被正确匹配过了，
                if gt_i in gt_matched_list:
                    continue
                iou = bb_intersection_over_union(predict_box, gt_box)
                print('====iou,',iou, predict_box, gt_box)
                if iou > IOU_THRESH:
                    pb_matched_list.append(pb_i)
                    
                    gt_class = gt_box_dict['box_label']
                    if gt_class == predict_class:
                        gt_matched_list.append(gt_i)
                    
                    confuse_matrix[gt_class][predict_class] += 1
                    

            #该框没有与gt框匹配，属于多检或者iou<0.5
            if pb_i not in pb_matched_list:
                confuse_matrix[BG_CLASS][predict_class] += 1

        
        for gt_i, gt_box_dict in enumerate(img_label[imgfile]):
            gt_class = gt_box_dict['box_label']
            confuse_matrix[gt_class][GT_INDEX] += 1
            #没被正确匹配过
            if gt_i not in gt_matched_list:
                confuse_matrix[gt_class][BG_CLASS] += 1

    sum_row = []
    for col_index in range(len(confuse_matrix[0])):
        sum_row.append(sum(confuse_matrix[:,col_index]))
    confuse_matrix = np.insert(confuse_matrix, len(confuse_matrix), sum_row, axis= 0)

    print(confuse_matrix)
    print('======================p=r===================')
    p_list = [.0] * CLASS_NUM
    r_list = [.0] * CLASS_NUM

    for i in range(CLASS_NUM):
        p_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[CLASS_NUM+1][i]+0.0000001)
        r_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[i][CLASS_NUM+2-1]+0.0000001)
        print(i+1,'p: {:.4f} r:{:.4f}'.format(p_list[i],r_list[i]))
    
    print('====================avg total p=r=============')
    correct_sum = 0
    pred_sum = 0
    label_sum = 0
    for i in range(CLASS_NUM):
        label_sum += confuse_matrix[i][GT_INDEX]
        pred_sum += confuse_matrix[CLASS_NUM+1][i]
        correct_sum += confuse_matrix[i][i]
    print('correct_sum:',correct_sum, 'pred_sum:',pred_sum, 'label_sum:', label_sum)
    print('p: {:.4f} r:{:.4f}'.format(correct_sum*1.0 / pred_sum , correct_sum*1.0 / label_sum ))

label_list = ['1_Point', '2_Palm', '3_Fist', '4_Ok', '5_Prayer', '6_Congratulation', '7_Honour', 
                    '8_Heart_single', '9_Thumb_up', '10_Thumb_down', '11_Rock', '12_Palm_up', 
                    '13_Other', '14_Heart_1', '15_Heart_2', '16_Heart_3', '17_Two', '18_Three', '19_Four']
#数据处理方式还是mosic方式，推理时候直接resize
def calc_pr_value_mosic():

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect
    from tool.utils_iou  import bb_intersection_over_union

    CLASS_NUM = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/6_1026_1_axer_train_only_dada_all/Yolov4_epoch_98.pth'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/ali_epoch_256.pth'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common/Yolov4_epoch_33.pth'#p: 0.8347 r:0.7028
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common/Yolov4_epoch_90.pth_bak'#p: 0.8639 r:0.7488
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common/Yolov4_epoch_92.pth_bak'#p: 0.8470 r:0.8031
    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common/Yolov4_epoch_33.pth_bak'#
 
    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common/Yolov4_epoch_214.pth'#p: 0.8814 r:0.7276
 

    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/15_1031_3_axer_train_only_dada_ratio_0.1_finetue_common/Yolov4_epoch_43.pth_bak'#p: 0.8393 r:0.7700
    # weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/15_1031_3_axer_train_only_dada_ratio_0.1_finetue_common/Yolov4_epoch_50.pth_bak'#p: 0.8036 r:0.7960
    # weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/15_1031_3_axer_train_only_dada_ratio_0.1_finetue_common/Yolov4_epoch_115.pth_bak'#p: 0.8382 r:0.7818
    # weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/15_1031_3_axer_train_only_dada_ratio_0.1_finetue_common/Yolov4_epoch_160.pth_bak'#p: p: 0.8378 r:0.8042
    # weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/15_1031_3_axer_train_only_dada_ratio_0.1_finetue_common/Yolov4_epoch_166.pth_bak'#p: 0.8476 r:0.7936
    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/15_1031_3_axer_train_only_dada_ratio_0.1_finetue_common/Yolov4_epoch_280.pth'# 


    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/20_1104_1_same16_axer_train_only_dada_ratio_0.19_finetue_common/20_1104_1_same16_axer_train_only_dada_ratio_0.19_finetue_common55.pth'#p: 0.8929 r:0.7398
    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/19_1104_2_axer_train_only_dada_ratio_0.3_finetue_common/19_1104_2_axer_train_only_dada_ratio_0.3_finetue_common92.pth'# p: 0.9178 r:0.7362
    #weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/21_1104_3_axer_train_only_dada_ratio_0.19_finetue_common_anchor/21_1104_3_axer_train_only_dada_ratio_0.19_finetue_common_anchor76.pth'#


    test_data_path = '/workspace/hand/data/dada/captures/captures_500_2'
    test_data_txt = '/workspace/hand/data/dada/captures/captures_500_videos_2_test_1500.txt'
    save_path = '/workspace/hand/data/dada/result_data'

    # test_data_txt = '/workspace/hand/data/gesture_5-30/gesture_c19_shuffle_yolo_style_test.txt'
    # test_data_path = '/workspace/hand/data/gesture_5-30/images'
    # save_path = '/workspace/hand/data/gesture_5-30/test_results'


    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 

    print('img_names====',len(img_names))

    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 


    # test_data_path = '/workspace/hand/data/dada/dada_img'
    # img_names = os.listdir(test_data_path)

    img_label={}
    for line in lines:
        line_list = line.strip().split()
        num = len(line_list[1:])   
        img_label[line_list[0]] = [] 
        for i in range(num):
            info = line_list[i + 1].split(',')
            box = [int(info[0]), int(info[1]), int(info[2]), int(info[3])]
            box_label = int(info[4])
            img_label[line_list[0]].append({'box': box, 'box_label': box_label})

    model = Yolov4(yolov4conv137weight=None, n_classes=CLASS_NUM, inference=True)

    
    checkpoint = torch.load(weightfile, map_location=torch.device('cuda'))
    
    pretrained_dict = checkpoint['model_state_dict']
    
    #pretrained_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    

    use_cuda = True
    if use_cuda:
        model.cuda()

    IOU_THRESH = 0.5
    #多加1行为bg，多加两列分别为bg和ground_truth
    BG_CLASS = CLASS_NUM 
    GT_INDEX = CLASS_NUM + 1
    confuse_matrix = np.zeros((CLASS_NUM+1, CLASS_NUM+2), dtype=np.int)

    for i, imgfile in enumerate(img_names):
        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))

        h,w,c = img.shape
        if h > w:
            padding_img = np.ones([h, h, 3])
        elif h < w:
            padding_img = np.ones([w, w, 3])

        padding_img[:, :, ] = np.mean(img, axis=(0, 1))
        padding_img[:h, :w] = img[:h, :w] 

        sized = cv2.resize(padding_img, (width, height))
        # sized = cv2.resize(padding_img, (352, 352))
        
        # t = np.ones([608, 608, 3])
        # t[:, :, ] = np.mean(img, axis=(0, 1))
        # t[:h, :w] = img[:h, :w] 
        # sized=t.copy()
        # cv2.imwrite('t.jpg', sized)
        sized = sized[..., ::-1].copy() # 
        boxes = do_detect(model, sized, 0.3, IOU_THRESH, use_cuda)
        # boxes = do_detect(model, sized, 0.5, IOU_THRESH, use_cuda)

        class_names = load_class_names(namesfile)
        #plot_boxes_cv2(img, boxes[0], os.path.join(save_path, imgfile), class_names)
        img_save = plot_boxes_cv2(img, boxes[0],class_names=class_names)
        #continue
        

        
        #记录ground truth box是否已经被匹配
        gt_matched_list = []
        pb_matched_list = []
        
        
        for pb_i, predict_box in enumerate(boxes[0]):
            predict_class = predict_box[6]

            for gt_i, gt_box_dict in enumerate(img_label[imgfile]):
                gt_box = gt_box_dict["box"]

                #已经被正确匹配过了，
                if gt_i in gt_matched_list:
                    continue
                iou = bb_intersection_over_union(predict_box, gt_box)
                #print('====iou,',iou, predict_box, gt_box)
                if iou > IOU_THRESH:
                    pb_matched_list.append(pb_i)
                    
                    gt_class = gt_box_dict['box_label']
                    if gt_class == predict_class:
                        gt_matched_list.append(gt_i)
                    
                    confuse_matrix[gt_class][predict_class] += 1

                    if not os.path.exists(os.path.join(save_path, label_list[gt_class])):
                        os.makedirs(os.path.join(save_path, label_list[gt_class]))
                    cv2.imwrite(os.path.join(save_path, label_list[gt_class], imgfile), img_save)
                    

            #该框没有与gt框匹配，属于多检或者iou<0.5
            if pb_i not in pb_matched_list:
                confuse_matrix[BG_CLASS][predict_class] += 1
                
                
                cv2.imwrite(os.path.join(save_path, imgfile), img_save)

        
        for gt_i, gt_box_dict in enumerate(img_label[imgfile]):
            gt_class = gt_box_dict['box_label']
            confuse_matrix[gt_class][GT_INDEX] += 1
            #没被正确匹配过
            if gt_i not in gt_matched_list:
                confuse_matrix[gt_class][BG_CLASS] += 1

    sum_row = []
    for col_index in range(len(confuse_matrix[0])):
        sum_row.append(sum(confuse_matrix[:,col_index]))
    confuse_matrix = np.insert(confuse_matrix, len(confuse_matrix), sum_row, axis= 0)

    print(confuse_matrix)
    print('======================p=r===================')
    p_list = [.0] * CLASS_NUM
    r_list = [.0] * CLASS_NUM

    for i in range(CLASS_NUM):
        p_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[CLASS_NUM+1][i]+0.0000001)
        r_list[i] = confuse_matrix[i][i] *1.0 / (confuse_matrix[i][CLASS_NUM+2-1]+0.0000001)
        print(i+1,'p: {:.4f} r:{:.4f}'.format(p_list[i],r_list[i]))
    
    print('====================avg total p=r=============')
    correct_sum = 0
    pred_sum = 0.0000001
    label_sum = 0.0000001
    for i in range(CLASS_NUM):
        label_sum += confuse_matrix[i][GT_INDEX]
        pred_sum += confuse_matrix[CLASS_NUM+1][i]
        correct_sum += confuse_matrix[i][i]
    print('correct_sum:',correct_sum, 'pred_sum:',pred_sum, 'label_sum:', label_sum)
    print('p: {:.4f} r:{:.4f}'.format(correct_sum*1.0 / pred_sum , correct_sum*1.0 / label_sum ))


    print('====================avg dada p=r=============')
    dada_index=[1,2,7,8,13,14,15]#分别为
    ges_name=['2_palm', '3_fist', '8_single_heart', '9_thumb_up', '14_heart_1', '15_heart_2', '16_heart_3']
    dada_index=[1,2,8]#分别为
    ges_name=['2_palm', '3_fist',  '9_thumb_up']
    
    correct_sum = 0
    pred_sum = 0.0000001
    label_sum = 0.0000001
    for i in dada_index:
        print(i+1, ges_name[dada_index.index(i)], confuse_matrix[i])

        label_sum += confuse_matrix[i][GT_INDEX]
        pred_sum += confuse_matrix[CLASS_NUM+1][i]
        correct_sum += confuse_matrix[i][i]
    print('correct_sum:',correct_sum, 'pred_sum:',pred_sum, 'label_sum:', label_sum)
    print('p: {:.4f} r:{:.4f}'.format(correct_sum*1.0 / pred_sum , correct_sum*1.0 / label_sum ))




def test_on_img():
    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    n_classes = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    # weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/Yolov4_epoch31.pth'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/6_1026_1_axer_train_only_dada_all/Yolov4_epoch_80.pth'
    
    # test_data_txt = '/workspace/hand/data/gesture_5-30/gesture_c19_shuffle_yolo_style_test.txt'
    # test_data_path = '/workspace/hand/data/gesture_5-30/images'
    # save_path = '/workspace/hand/data/gesture_5-30/test_results'

    test_data_path = '/workspace/hand/data/dada/dada_img'
    # save_path = '/workspace/hand/data/dada/result_data'

    test_data_txt = '/workspace/hand/data/dada/captures/captures_500_videos_2_test_1500.txt'
    #test_data_path = '/workspace/hand/data/dada/captures/captures_500_2'
    save_path = '/workspace/hand/data/dada/result_data'



    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 

    img_names = os.listdir(test_data_path)



    print('img_names====',len(img_names))


    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    
    checkpoint = torch.load(weightfile, map_location=torch.device('cuda'))
    
    pretrained_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    

    use_cuda = True
    if use_cuda:
        model.cuda()




    for i, imgfile in enumerate(img_names):

        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))


        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        boxes = do_detect(model, sized, 0.4, 0.5, use_cuda)

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(img, boxes[0], os.path.join(save_path, imgfile), class_names)


def test_on_img_new_model():
    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    n_classes = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/1017_data_preprocess_resume/Yolov4_epoch_25.pth'
    test_data_txt = '/workspace/hand/data/gesture_5-30/gesture_c19_shuffle_yolo_style_test.txt'
    # test_data_path = '/workspace/hand/data/gesture_5-30/images'
    # save_path = '/workspace/hand/data/gesture_5-30/test_results'

    test_data_path = '/workspace/hand/data/dada/dada_img'
    save_path = '/workspace/hand/data/dada/result_data'



    lines = open(test_data_txt, 'r').readlines()
    img_names = [line.strip().split()[0] for line in lines] 

    print('img_names====',len(img_names))


    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

   
    checkpoint = torch.load(weightfile, map_location=torch.device('cuda'))
    
    pretrained_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    

    use_cuda = True
    if use_cuda:
        model.cuda()


    # lines = open(test_data_txt, 'r').readlines()
    # img_names = [line.strip().split()[0] for line in lines] 
    img_names = os.listdir(test_data_path)


    for i, imgfile in enumerate(img_names):

        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))
        
        h,w,c = img.shape
        if h > w:
            padding_img = np.ones([h, h, 3])
            padding_img[:, :, ] = np.mean(img, axis=(0, 1))
            padding_img[:h, (h - w)// 2 : (w+(h - w)// 2)] = img[:h, :w] 

        elif h < w:
            padding_img = np.ones([w, w, 3])
            padding_img[:, :, ] = np.mean(img, axis=(0, 1))
            padding_img[(w - h)// 2 : (h+(w - h)// 2)] = img[:h, :w] 

        
        padding_img[:h, :w] = img[:h, :w] 



        sized = cv2.resize(padding_img, (width, height))
        

        # sized = cv2.resize(img, (width, height))
        # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        boxes = do_detect(model, sized, 0.4, 0.5, use_cuda)

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(img, boxes[0], os.path.join(save_path, imgfile), class_names)


def filter():
    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    n_classes = 19
    height = 608
    width = 608
    namesfile = '/workspace/hand/code/pytorch-YOLOv4/data/hand.names'
    weightfile = '/workspace/hand/code/pytorch-YOLOv4/checkpoints/Yolov4_epoch29.pth'
    test_data_txt = '/workspace/hand/data/dada/model_filter_local.txt'
    test_data_path = '/workspace/hand/data/dada/model_chosen'
    save_path = '/workspace/hand/data/dada/result_data'



    lines = open(test_data_txt, 'r').readlines()
    # img_names = [line.strip() for line in lines] 
    img_names = [line.strip().split()[1] for line in lines]

    print('img_names====',len(img_names))


    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
     
    #model.load_state_dict(pretrained_dict)

    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    

    use_cuda = True
    if use_cuda:
        model.cuda()


    

    
    for i, imgfile in enumerate(img_names):

        print(i, os.path.join(test_data_path, imgfile))

        img = cv2.imread(os.path.join(test_data_path, imgfile))

        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(model, sized, 0.1, 0.6, use_cuda)

        if len(boxes[0]) > 0:
            with open('model_filter_axer.txt', 'a') as f:
                f.write(str(i) + ' ' + imgfile +'\n')

        print(i,imgfile)


if __name__ == "__main__":
    import sys
    import cv2

    if sys.argv[1] == "test":
        test_on_img()
        #test_on_img_new_model()#没有马赛克增广的
        exit() 

    if sys.argv[1] == "calc":
        #calc_pr_value_new()
        #calc_pr_value_new_new_model()
        calc_pr_value_mosic()
        exit() 

    if sys.argv[1] == "filter":
        filter()
        exit() 

    filter()
    exit() 

    namesfile = None
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
        namesfile = sys.argv[6]
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    # pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))

    #model.load_state_dict(pretrained_dict)

    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    

    use_cuda = True
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    for i in range(1):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        print('predicting-----')
        # boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        boxes = do_detect(model, sized, 0.1, 0.6, use_cuda)

    if namesfile == None:
        if n_classes == 20:
            namesfile = 'data/voc.names'
        elif n_classes == 80:
            namesfile = 'data/coco.names'
        else:
            print("please give namefile")

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes[0], 'predictions_2.jpg', class_names)
