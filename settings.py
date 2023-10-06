#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import logging


class Config:

    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    TEMP_FOLDER = os.path.join(BASEDIR, 'temp')
    LOG_LEVEL = logging.DEBUG
    if os.environ.get('LOG_LEVEL') == 'INFO':
        LOG_LEVEL = logging.INFO
    # YAML_FILE_PATH = os.path.join(BASEDIR, 'tools/config.yaml')
    DOWNLOAD_TIMEOUT = 3
    DEPLOY_ENV = os.environ.get('DEPLOY_ENV') or 'local'

    # PAAS
    SERVER_HOST = os.environ.get('SERVER_HOST') or 'hands-gesture-server'
    SERVER_PORT = 8080
    APP_NAME = 'HANDS-GESTURE-SERVER'
    HEART_BEAT_INTERVAL = 3

    # 数据回流
    APP_URL = os.environ.get('APP_URL') or "/aiimage/novabell/hands-gesture"
    APOLLO_URL = os.environ.get('APOLLO_URL') or "http://godhand-apollo-config:8080"
    APOLLO_NAMESPACE = os.environ.get('APOLLO_NAMESPACE') or "datawork-common"
    if DEPLOY_ENV == 'test':
        DATA_CHANGE_URL = "http://internal.gateway-godeye-test.facethink.com/ossurl/material/ossconverts"
    elif DEPLOY_ENV == 'pre':
        DATA_CHANGE_URL = "http://internal.gateway.facethink.com/ossurl/material/ossconverts"
    elif DEPLOY_ENV == 'prod':
        DATA_CHANGE_URL = "http://internal.openai.100tal.com/ossurl/material/ossconverts"

    API_TYPE = 0  # 同步
    BIZ_TYPE = 'datawork-image'
    VERSION = os.environ.get('VERSION') or '2.0'
    APOLLO_ID = "novabell"
    APOLLO_TOPIC = 'image'
    APOLLO_KAFKA = 'kafka-bootstrap-servers'

    APM_URL = os.environ.get('APM_URL')
    ALARM_CRITICAL = os.environ.get('ALARM_CRITICAL') or "alertcode:910010001, " \
                                                         "alerturl:/aiimage/novabell/hands-gesture, alertmsg:" \
                                                         " NovabellWarning 告警"

    ALARM_DATA = "alertcode:910010001, alerturl:/aiimage/novabell/hands-gesture, alertmsg: DataReflow 告警"


config = Config()
if __name__ == '__main__':
    print(dir(config))
    print(config.APP_NAME)
