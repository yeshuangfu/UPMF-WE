from src.util import os_util
import src.config.base_config as base_config
from enum import Enum

algo_interim_dir = os_util.create_dir_if_not_exist('/'.join([base_config.INTERIM_DATA_ROOT, 'weight_prediction']))
algo_model_dir = os_util.create_dir_if_not_exist('/'.join([base_config.MODEL_DATA_ROOT, 'weight_prediction']))

predict_data_dt_path = "/".join([algo_interim_dir, "WeightPredictionNfm.model.predict_dt.csv"])
predict_data_dt_path_type = 'csv'


class TrainModuleConfig(Enum):
    # 训练时间间隔
    TRAIN_ALGO_INTERVAL = 30
    # 表现期长度
    OUTCOME_WINDOW_LEN = None
    # 表现期偏移量
    OUTCOME_OFFSET = None

    # 训练参数
    num_iter = 50
    batch_size = 128
    lr = 5e-3
    momentum = 0.9
    l2_decay = 1e-2
    seed = 2023
    retrieve_days_interval = (0, 130)
    avg_wt_interval = (0, 8)

    # 最原始版本
    # num_iter = 10
    # batch_size = 128
    # lr = 5e-6
    # momentum = 0.9
    # l2_decay = 1e-2
    # seed = 2023
    # retrieve_days_interval = (50, 130)
    # avg_wt_interval = (2, 8)
    # BASE_EMB_DIM = 64


class PredictModuleConfig(Enum):
    # 预测时间偏移量
    PREDICT_OUTCOME_OFFSET = None
    # 预测时间间隔
    PREDICT_RUNNING_DT_INTERVAL = 1

    predict_days_interval = (50, 130)


class EvalModuleConfig(Enum):
    # 测试时间偏移量
    OUTCOME_OFFSET = 1
    # 测试时间间隔
    EVAL_INTERVAL  = 1
    # 测试时间窗口长度
    OUTCOME_WINDOW_LEN = 1


BASE_EMB_DIM = 128
EMB_DIM = {
    "l3_breeds_class_nm": BASE_EMB_DIM,
    "feed_breeds_nm": BASE_EMB_DIM,
    "gender": BASE_EMB_DIM,
    "breeds_class_nm": BASE_EMB_DIM,
    "rearer_dk": 16,
    "org_inv_dk": BASE_EMB_DIM,
    "l3_org_inv_nm": BASE_EMB_DIM,
    "l4_org_inv_nm": BASE_EMB_DIM,
    "tech_bk": 32,
}
