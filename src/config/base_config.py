from enum import Enum

import socket

from src.util import os_util

local_ip = socket.gethostbyname(socket.gethostname())
if local_ip == '10.11.41.17':
    base_dir = "/data0/hy_data_mining/poultry_breed_customer_precision_sale/"  # 沙箱测试
elif local_ip == '10.11.21.201':
    base_dir = "/data0/hy_data_mining/poultry_breed_customer_precision_sale/"  # 生产机器
else:
    base_dir = ".."


RAW_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/raw']))
INTERIM_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/interim']))
FEATURE_STORE_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/interim/feature_store']))
MODEL_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/model']))
EXTERNAL_DIR = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/external']))
KEY_TAB_FILE = '/'.join([EXTERNAL_DIR, "wens.keytab"])


class RawData(Enum):
    # 销售品种库存天表
    SALES_SHIPPING_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_SALES_SHIPPING.csv'])
    # 默认日耗料
    AGE_FEED_QTY_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_AGE_FEED_QTY.csv'])
    # 「鸡群巡查记录」供给预测数据
    CHE_INSDATA_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_CHE_INSDATA.csv'])
    # 死淘数据
    DEATH_CONFIRM_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_DEATH_CONFIRM.csv'])
    # 饲料耗用成本表
    FEED_PRICE_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_FEED_PRICE.csv'])
    # 鸡群回收
    REARER_LANDP_RETRIEVE_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_REARER_LANDP_RETRIEVE.csv'])
    # 库存天表
    REARER_POP_SALE_FCST_DAY_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_REARER_POP_SALE_FCST_DAY.csv'])
    # 订购记录
    SALE_ORDER_INFO_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_SALE_ORDER_INFO.csv'])
    # 销售定价
    ADS_AI_MRT_SALE_PRICE_INFO_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_SALE_PRICE_INFO.csv'])
    # 合约客户签约量
    SALES_AGREEMENT_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_SALES_AGREEMENT.csv'])
    # 合约客户签约量-天粒度
    SALES_AGREEMENT_DAY_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_SALES_AGREEMENT_DAY.csv'])
    # 销售品种档次对照表
    DIM_BREEDS_FILECLASS_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_DIM_BREEDS_CLASS.csv'])
    # 内外部客户标志
    DIM_CUST_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_DIM_CUST.csv'])
    # 「生产组织对照表」维度信息表
    DIM_ORG_INV_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_DIM_ORG_INV.csv'])
    # 「鸡群档案」供给预测数据
    REARER_POP_DATA_PATH = "/".join([RAW_DATA_ROOT, 'DT03_REARER_POP_SALE_FCST.csv'])
    # 「品种对照表」维度信息表
    FACT_YQ_FORSALE_BREEDS_RELATION_DATA_PATH = "/".join([RAW_DATA_ROOT, 'FACT_YQ_FORSALE_BREEDS_RELATION.csv'])
    # 客户表
    ADS_AI_MRT_ACTIVE_CUST_DATA_PATH = "/".join([RAW_DATA_ROOT, 'ADS_AI_MRT_ACTIVE_CUST.csv'])

class ModulePath(Enum):
    # 数据预处理模块
    pre_process_data_class = 'PreProcessDataset'
    # 特征模块
    """
        todo 详细说明每个属性的含义 
    """
    feature_dataset_list = [
        # 鸡群属性特征
        {'dataset_name': 'dataset.chicken_attributes',
         'file_type': 'csv',
         'main_class_name': 'ChickenAttributesDataSet',
         'params': {}}
    ]
    # 训练模块
    algo_list = [
        # 任务1 均重预测
        {'algo_name': 'algorithm.weight_prediction_nfm_algo',
         'algo_main_class': 'WeightPredictionNfmAlgo',
         'algo_config': 'config.weight_prediction_config'},

        # 任务2 采购量预测
        # {'algo_name': 'algorithm.sale_qty_prediction_algo',
        #  'algo_main_class': 'SaleQtyPredictionAlgo',
        #  'algo_config': 'config.sale_qty_prediction_config'},
    ]
    # 预测模块
    algo_predict = [
        # 任务1-均重预测
        {'algo_name': 'algorithm.weight_prediction_nfm_algo',
         'algo_main_class': 'WeightPredictionNfmAlgo',
         'algo_config': 'config.weight_prediction_config'},
    ]

    # 测试模块
    eval_list = [
        # 任务1-均重预测
        {'predict_class_name': 'algorithm.weight_prediction_nfm_algo',
         'predict_main_class_name': 'WeightPredictionNfmAlgo',
         'eval_class_name': 'eval.main',
         'eval_main_class_name': 'AvgWtEval',
         'alog_config_name': 'config.weight_prediction_config',
         'params': {'observation_window_len': 14}},

    ]


