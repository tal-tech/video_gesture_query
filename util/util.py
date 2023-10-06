#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import json
import logging
import sys
# import requests
from flask import Response
from settings import Config
from flask import Flask
import copy
import requests

hand_gestures = {'1_Point':'1', '2_Palm':'5', '3_Fist':'Fist', '4_Ok':'Ok', '5_Prayer':'Prayer', '6_Congratulation':'other', '7_Honour':'Honour',
                    '8_Heart_single':'other', '9_Thumb_up':'Thumb_up', '10_Thumb_down':'other', '11_Rock':'other', '12_Palm_up':'other',
                    '13_Other':'other', '14_Heart_1':'other', '15_Heart_2':'other', '16_Heart_3':'other', '17_Two':'2', '18_Three':'3', '19_Four':'4'}

class Status:
    SUCCESS = 'success'
    PARAMETERS_MISS = 'parameters missing'
    INTERNAL_ERR = 'internal error'
    ILLEGAL_URL = 'illegal url'
    ILLEGAL_SIZE = 'illegal image size'
    ILLEGAL_IMAGE = 'illegal image type'
    BAD_REQUEST = 'analysis json failure'
    ILLEGAL_PICTURE = 'illegal picture'
    DOWNLOAD_ERR = 'download error'
    LOAD_MODEL_ERR = 'load model error'
    RECOGNIZE_ERR = 'recognition error'
    DOWNLOAD_TIMEOUT = 'download timeout'
    WRITE_FILE_ERR = 'illegal base64'


err_code = {
    Status.SUCCESS: (200, 20000),
    Status.PARAMETERS_MISS: (200, 3005034000),
    Status.ILLEGAL_URL: (200, 3005034001),
    Status.ILLEGAL_SIZE: (200, 3005034002),
    Status.ILLEGAL_IMAGE: (200, 3005034003),
    Status.BAD_REQUEST: (200, 3005034004),
    Status.ILLEGAL_PICTURE: (200, 3005034005),
    Status.DOWNLOAD_ERR: (200, 3005035001),
    Status.INTERNAL_ERR: (200, 3005035002),
    Status.LOAD_MODEL_ERR: (200, 3005035003),
    Status.RECOGNIZE_ERR: (200, 3005035004),
    Status.DOWNLOAD_TIMEOUT: (200, 3005035005),
    Status.WRITE_FILE_ERR: (200, 3005035006)
}

app = Flask(__name__)
g_logger = logging.getLogger(__name__)


def init_logger():
    ch = logging.StreamHandler()
    # ch.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s [%(thread)d] - %(module)s %(lineno)d - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    g_logger.setLevel(Config.LOG_LEVEL)
    g_logger.propagate = False
    g_logger.addHandler(ch)


init_logger()


def make_response(msg, idx, text=None):
    if text is None:
        flow_data = []
    else:
        flow_data = copy.deepcopy(text)
        for item in flow_data:
            item['hand_gesture'] = hand_gestures[item['hand_gesture']]
    return Response(json.dumps({"msg": msg, "data": flow_data, "code": err_code[msg][1]}),
                    content_type='application/json', status=err_code[msg][0])


def make_dict(msg, req_id, items=None):
    if items is None:
        items = {}
    return {"msg": msg, "data": {"items": items},
            "code": err_code[msg][1], "requestId": req_id}


class DownloadTimeOutEx(Exception):

    def __init__(self):
        Exception.__init__(self, 'download timeout')


class RecognitionEx(Exception):

    def __init__(self):
        Exception.__init__(self, 'recognition error')


class IllegalTypeEx(Exception):

    def __init__(self):
        Exception.__init__(self, 'illegal image type')


class DownErrorEx(Exception):

    def __init__(self):
        Exception.__init__(self, 'download error')


class LoadModelEx(Exception):

    def __init__(self):
        Exception.__init__(self, 'load model error')


def check_url(url):
    ip_pattern = re.compile(
        r'^(?:(?:1[0-9][0-9]\.)|(?:2[0-4][0-9]\.)|(?:25[0-5]\.)|(?:[1-9][0-9]\.)|(?:[0-9]\.)){3}'
        r'(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|(?:25[0-5])|(?:[1-9][0-9])|(?:[0-9]))$')
    url_pattern = re.compile(
        r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    try:
        if ip_pattern.match(url) or url_pattern.match(url):
            return True
        else:
            return False
    except Exception as e:
        return False


def alert_msg(msg):
    return g_logger.error("{} - {}".format(Config.ALARM_CRITICAL, msg))
