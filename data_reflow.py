#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
import json
import requests
import traceback
import base64
from settings import Config
from c_apollo import g_apollo
from util.util import g_logger
from kafka import KafkaProducer
from util.util import err_code, alert_msg


def send_mq(idx, msg='success', req_time=0, input_data=None, url=None, img_data=None, output_data=None):
    try:
        if not g_apollo:
            g_logger.error('not apollo')
            return
        kafka_url = g_apollo.get_value(
            Config.APOLLO_KAFKA,
            namespace=Config.APOLLO_NAMESPACE)
        kafka_topic = g_apollo.get_value(
            Config.APOLLO_TOPIC,
            namespace=Config.APOLLO_NAMESPACE)
        g_logger.info('request:{} kafka:{}, {}'.format(idx, kafka_url, kafka_topic))
        if isinstance(input_data, dict) and 'picture' in input_data:
            input_data.pop('picture')
        d = {
            "apiType": Config.API_TYPE,
            "bizType": Config.BIZ_TYPE,
            "requestId": idx,
            "url": Config.APP_URL,
            "responseTime": int(time.time() * 1000),
            "sendTime": int(time.time() * 1000),
            "sourceInfos": [{
                "id": idx,
                "sourceType": "base64" if img_data else "url",
                "content": img_data if img_data else get_url(url, idx) if url else '',
            }],
            "data": output_data,
            "version": Config.VERSION,
            "sourceRemark": input_data,
            "code": err_code[msg][1],
            "msg": msg,
            "errMsg": 'success',
            'errCode': 200,
            "appKey": input_data.get('appKey') if input_data else '',
            "requestTime": req_time,
        }
        d['duration'] = d['responseTime'] - d['requestTime']
        mq_url = kafka_url.split(',')[0]
        # g_logger.debug('{} - request time:{}'.format(idx, d['duration']))
        kafka = KafkaProducer(bootstrap_servers=mq_url, max_request_size=10*1024*1024)
        msg = json.dumps(d)
        kafka.send(kafka_topic, msg.encode('utf-8'))
        kafka.close()
    except Exception as e:
        g_logger.error(Config.ALARM_DATA + ' - {} 数据回流失败:{}'.format(
                idx, traceback.format_exception(
                    type(e), e, e.__traceback__)))


def get_url(url, idx):

    d = {
        'urls': [url],
        'requestId': idx,
        'sendTime': int(round(time.time() * 1000))
    }

    try:
        t1 = time.time()
        ret = requests.post(url=Config.DATA_CHANGE_URL, json=d)
        t2 = time.time()
        g_logger.info('{} - url change time: {}'.format(idx, t2 - t1))
        ret_json = ret.json()
        if ret_json.get('code') != 2000000:
            ch_url = ret_json.get('resultBean')[0].get('innerUrl')
        else:
            ch_url = url
    except Exception as e:
        g_logger.error("{} - error in change url:{}".format(idx, e))
        ch_url = url

    return ch_url
