# -*- coding: utf-8 -*-
'''
@Time          : 2020/10/28 18:49
@Author        : jiawenhao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import cv2
import numpy as np
import os
from collections import OrderedDict
import time
import argparse
import torch
from tool.torch_utils import do_detect
#from models import Yolov4
from tool.models import Yolov4


class Det_dada_hand(object):
    
    def __init__(self):
        
        print('------init---------')
        self.model = Yolov4(yolov4conv137weight=None, n_classes=19, inference=True)
        weightfile = './tool/model/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common_epoch_92_release_1.0.0.pth'
        checkpoint = torch.load(weightfile, map_location=torch.device('cuda'))
        #checkpoint = torch.load(weightfile, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['model_state_dict']


        # checkpoint_dict = {'epoch': 300, 
        #            'model_state_dict': checkpoint['model_state_dict'], 
        #            'optim_state_dict': '', 
        #            'criterion_state_dict': ''}

        # torch.save(checkpoint_dict, '14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common_epoch_92_release_1.0.0.pth')



        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.hand_gestures = ['1_Point', '2_Palm', '3_Fist', '4_Ok', '5_Prayer', '6_Congratulation', '7_Honour', 
                    '8_Heart_single', '9_Thumb_up', '10_Thumb_down', '11_Rock', '12_Palm_up', 
                    '13_Other', '14_Heart_1', '15_Heart_2', '16_Heart_3', '17_Two', '18_Three', '19_Four']

        print('------init done-----')
    '''
    输入：img，ndarray类型，要求为3维，依次代表图片高，图片宽，图片通道数，其中图片通道数应为3
    输出：
    RET_CODE:状态码，int类型，
            0代表正常检测到手，BOXES有意义。
            1代表没有检测到手，BOXES为[]
            2代表其他异常情况，比如img为空等，BOXES为[],

    BOXES:检测框，list类型，包含一系列box，
          每个box同样为list类型，包含4个值，分别为框的左上角坐标x,y 和框的宽高w,h 以及该box置信度p和预测类别c,类似[x,y,w,h,p,c]
          其中x y w h c为int类型
          p为float类型

    '''
    def detect_paas(self, img):
        RET_CODE=2
        BOXES=[]

        if img is None:
            return RET_CODE, BOXES
        h,w,c = img.shape
        if c != 3:
            return RET_CODE, BOXES
        
        if h > w:
            padding_img = np.ones([h, h, 3])
        elif h <= w:
            padding_img = np.ones([w, w, 3])
        
        padding_img[:, :, ] = np.mean(img, axis=(0, 1))
        padding_img[:h, :w] = img[:h, :w]


        sized = cv2.resize(padding_img, (608, 608))
        sized = sized[..., ::-1].copy() 

        boxes = do_detect(self.model, sized, 0.3, 0.5, use_cuda=True)

        if len(boxes[0]) ==0:
            RET_CODE = 1
            return RET_CODE, BOXES  

        for i in range(len(boxes[0])):
            box = boxes[0][i]
            if h > w:
                x1 = int(box[0] * h)
                y1 = int(box[1] * h)
                x2 = int(box[2] * h)
                y2 = int(box[3] * h)
            else:
                x1 = int(box[0] * w)
                y1 = int(box[1] * w)
                x2 = int(box[2] * w)
                y2 = int(box[3] * w)

            BOXES.append([x1, y1, x2-x1, y2-y1, box[5], box[6]])

        RET_CODE=0
        return RET_CODE, BOXES  

        
    #算法测试接口
    def detect_test(self, test_data_path, save_path=None):

        img_names = os.listdir(test_data_path)

        print('imgs====',len(img_names))

        cnt = 0
        time_list =[]
        for i, imgfile in enumerate(img_names):

            print(i, os.path.join(test_data_path, imgfile))
        
            img = cv2.imread(os.path.join(test_data_path, imgfile))

            start_time = time.time()
            ret_code, boxes = self.detect_paas(img)
            end_time = time.time()
            if i:
                time_list.append((end_time - start_time)*1000)
            if ret_code==1 :
                print('no hand')
                cnt += 1
                continue
            if ret_code ==2:
                print('error !!!!!')
                continue
            
            if save_path:

                for box in boxes:
                    x, y, w, h, cls_conf, cls_id = box[0], box[1], box[2], box[3], box[4], box[5]
                    img = cv2.putText(img, '%s:%.2f' % (self.hand_gestures[cls_id], cls_conf), (x, y + h - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
                
                cv2.imwrite(os.path.join(save_path, imgfile), img)

        if len(time_list)> 0:
            print(cnt, 'avg time: {:.2f} ms'.format(sum(time_list) / len(time_list)) )



def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-weightfile', type=str,
                        default='./tool/model/14_1031_2_axer_train_only_dada_ratio_0.19_finetue_common_epoch_92_release_1.0.0.pth',
                        help='path of trained model.', dest='weightfile')

    # parser.add_argument('-test_data_path', type=str,
    #                 default='/workspace/hand/data/dada/test_dataset',
    #                 help='path of your image file.', dest='test_data_path')
    #
    # parser.add_argument('-save_path', type=str,
    #                 default='/workspace/hand/data/dada/result_data',
    #                 help='path of your result path.', dest='save_path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    hand_gestures = {'1_Point': '1', '2_Palm': '5', '3_Fist': 'Fist', '4_Ok': 'Ok', '5_Prayer': 'Prayer',
                     '6_Congratulation': 'other', '7_Honour': 'Honour',
                     '8_Heart_single': 'other', '9_Thumb_up': 'Thumb_up', '10_Thumb_down': 'other', '11_Rock': 'other',
                     '12_Palm_up': 'other',
                     '13_Other': 'other', '14_Heart_1': 'other', '15_Heart_2': 'other', '16_Heart_3': 'other',
                     '17_Two': '2', '18_Three': '3', '19_Four': '4'}
    
    args = get_args()

    folder_path = 'util/'
    det=Det_dada_hand()
    image_path = 'util/test2.png'
    img = cv2.imread(image_path)
    a,b = det.detect_paas(img)
    gesture_pic = det.hand_gestures[b[0][-1]]
    found = False
    # 遍历文件夹中的所有视频文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  # 假设所有视频都是MP4格式
            video_path = os.path.join(folder_path, filename)
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            while True:
                # 读取一帧图像，返回值ret为True或False代表有无下一帧图像；frame就是每一帧的图像，与imread()函数返回类型相同
                ret, frame = cap.read()

                if not ret:  # 如果没有下一帧了，则跳出循环
                    break
                
                #  det = Det_dada_hand(weightfile=args.weightfile)
                a,b = det.detect_paas(frame)
                items =[]
                for box in b:
                    bbox = box[0:4]
                    gesture = det.hand_gestures[box[-1]]
                    item = {'hand_gesture': gesture, 'bbox': bbox}
                    items.append(item)
                print(items)
                for item in items:
                    item['hand_gesture'] = hand_gestures[item['hand_gesture']]
                    if gesture_pic == item['hand_gesture']:
                        found=True
                        print("Found matching gesture!")
                        break
                print(items)
                
            
            cap.release()  # 关闭视频文件
            if found:
                break
            else :
                print("no found")

  
    #det.detect_test(test_data_path=args.test_data_path, save_path=args.save_path)
