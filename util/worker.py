#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import cv2
import base64
import requests
from util.read_lock import ReadLock
from util.util import g_logger, DownloadTimeOutEx
from demo import get_args, Det_dada_hand


g_download_lock = ReadLock(4)
g_pro_lock = ReadLock(1)


class Worker(object):

    def __init__(self, file_name,model,url, idx, b64_data,
                 items=None, flow_items=None, hands_bboxes=None, recognize_status=None):

        self.idx = idx
        self.file_name = file_name
        self.model = model
        self.url = url
        self.b64_data = b64_data
        self.items = items
        self.flow_items = flow_items
        self.hands_bboxes = hands_bboxes
        self.recognize_status = recognize_status
        self.size = 4 * 1024 * 1024

    # def hand_gesture(self, image):
    #     try:
    #         items = []
    #         for i, hands_bbox in enumerate(self.hands_bboxes):
    #             x_min, y_min, x_max, y_max = hands_bbox
    #             pre_gesture = self.hands_gesture[i]
    #             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (85, 25, 158), 2)
    #             cv2.putText(image, str(pre_gesture), (x_min, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
    #                         2)
    #
    #
    #             items.append(item)
    #     except Exception as e:
    #         items = {}
    #         g_logger.error('{} Recognition Error: {}'.format(self.idx, e))
    #
    #     self.items = items

    def download(self):

        try:
            r = requests.get(self.url, stream=True, timeout=(0.5, 1))
            down_chunk = 0
            f = open(self.file_name, 'wb')
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if down_chunk > self.size:
                        f.close()
                        raise DownloadTimeOutEx
                    f.write(chunk)
                    down_chunk += 1024
                    f.flush()
            f.close()
            return 'success'
        except DownloadTimeOutEx:
            return 'illegal size'
        except requests.ConnectionError:
            return 'illegal url'
        except Exception as e:
            g_logger.error('{} Download Error:{}'.format(self.idx, e))
            return 'download error'

    def recognition(self, image):
        try:
            items = []
            a, b = self.model.detect_paas(image)
            if a == 2:
                self.recognize_status = 'recognition error'
            elif a == 1:
                self.items=[]
            else:
                for box in b:
                    bbox = box[0:4]
                    gesture = self.model.hand_gestures[box[-1]]
                    item = {'hand_gesture':gesture,'bbox':bbox}
                    items.append(item)
            self.items = items
        except Exception as e:
            g_logger.error('{} recognition error:{}'.format(self.idx, e))


    def get_b64data(self):

        try:
            if int(len(self.b64_data) * 0.75) > self.size:
                return 'illegal image size'

            f = open(self.file_name, 'wb')
            f.write(base64.b64decode(self.b64_data.encode()))
            f.flush()
            f.close()
            return 'success'
        except Exception as e:
            g_logger.error('{} illegal base64: {}'.format(self.idx, e))

        return 'illegal base64'

    def start_work(self):
        if self.model is None:
            return '','','load model error'
        if self.b64_data is None:
            # g_download_lock.acquire()
            msg = self.download()
            # g_download_lock.release()
        else:
            # g_download_lock.acquire()
            msg = self.get_b64data()
            # g_download_lock.release()
        if msg != 'success':
            self.items = False
        else:
            try:
                # g_pro_lock.acquire()
                image = cv2.imread(self.file_name)
                if image is None:
                    return '', '', 'illegal image type'
                h, w, c = image.shape
                if max(h, w) > 1920 or min(h, w) > 1080:
                    return '', '', 'illegal image size'
                self.recognition(image)
                # self.hand_gesture(image)
            except Exception as e:
                g_logger.error('{} Model error: {}'.format(self.idx, e))
            # finally:
            #     g_pro_lock.release()
        return self.items, msg, self.recognize_status
