import importlib
import logging
import os
import sys

import src.config.base_config as config


class TrainModelMain(object):

    def __init__(self, **param):
        self.train_algo_outcome_dt_end = param.get('train_algo_outcome_dt_end')
        self.params_item = param.get('params_item')
        self.logger = param.get('logger')

    def train_model_generality(self, algo_name: str, algo_main_class: str, alog_config_name: str):


        self.logger.info('----------------------------------------开始训练模型， 任务： %s ----------------------------------------' % algo_name)
        algo_module = importlib.import_module(algo_name)
        alog_config = importlib.import_module(alog_config_name)
        main_class = getattr(algo_module, algo_main_class)
        main_class_instance = main_class(self.params_item)
        valid_algo_running_dt_end = None
        valid_algo_interval = None
        main_class_instance.train(self.train_algo_outcome_dt_end, alog_config.TrainModuleConfig.OUTCOME_OFFSET.value,
                                  alog_config.TrainModuleConfig.OUTCOME_WINDOW_LEN.value,
                                  alog_config.TrainModuleConfig.TRAIN_ALGO_INTERVAL.value,
                                  valid_algo_running_dt_end, valid_algo_interval, self.params_item)
        self.logger.info('----------------------------------------任务： %s 模型训练完成----------------------------------------' % algo_name)

    def train_all_model(self):
        self.logger.info("----------------------------------------开始训练模型，时间: %s ----------------------------------------" % self.train_algo_outcome_dt_end)
        algo_list = config.ModulePath.algo_list.value
        for algo_module in algo_list:
            self.train_model_generality(algo_module['algo_name'], algo_module['algo_main_class'],
                                        algo_module['algo_config'])
        self.logger.info("----------------------------------------模型生成完成，进入预测模块----------------------------------------")