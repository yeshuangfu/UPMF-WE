import argparse
from datetime import datetime, timedelta
from itertools import product

from util.logger import init_logger

from module import feature_generate_main, train_model_main, eval_main
import torch


if __name__ == '__main__':
    logger = init_logger('../log/job_eval')

    dates_string = '2023-09-05'

    logger.info('----------------------------------------初始化参数----------------------------------------')

    logger.info('----------------------------------------开始执行测试流程----------------------------------------')

    # 所有模块运行时间的日期边界
    train_algo_outcome_dt_end = '2023-09-07'
    eval_running_dt_end = '2023-08-23'

    # 特征预计算模块
    # feature_module_instance = feature_generate_main.FeatureGenerateMain(running_dt=running_dt,
    #                                                                     origin_feature_precompute_interval=origin_feature_precompute_interval,
    #                                                                     logger=logger)
    # feature_module_instance.generate_feature()

    # 定义超参数的搜索空间num_iter batch_size lr momentum l2_decay seed retrieve_days_interval avg_wt_interval BASE_EMB_DIM
    # batch_size
    param_batch64_values = [10, 128, 5e-3, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_batch128_values = [10, 256, 5e-3, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_batch256_values = [10, 512, 5e-3, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_batch512_values = [10, 1024, 5e-3, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_batch1024_values = [10, 2048, 5e-3, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]

    # lr
    param_lr02_values = [10, 128, 0.0030, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_lr03_values = [10, 128, 0.0035, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_lr04_values = [10, 128, 0.0040, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_lr05_values = [10, 128, 0.0045, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    param_lr06_values = [10, 128, 0.0050, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]

    # momentum
    momentum_70 = [10, 128, 0.0050, 0.70, 1e-2, 2023, (50, 130), (2, 8), 128]
    momentum_75 = [10, 128, 0.0050, 0.75, 1e-2, 2023, (50, 130), (2, 8), 128]
    momentum_80 = [10, 128, 0.0050, 0.80, 1e-2, 2023, (50, 130), (2, 8), 128]
    momentum_85 = [10, 128, 0.0050, 0.85, 1e-2, 2023, (50, 130), (2, 8), 128]
    momentum_90 = [10, 128, 0.0050, 0.90, 1e-2, 2023, (50, 130), (2, 8), 128]

    # BASE_EMB_DIM
    BASE_EMB_DIM_16 = [10, 256, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 16]
    BASE_EMB_DIM_32 = [10, 256, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 32]
    BASE_EMB_DIM_64 = [10, 256, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 64]
    BASE_EMB_DIM_128 = [10, 256, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 128]
    BASE_EMB_DIM_256 = [10, 256, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 256]

    # south_best_params = [[10, 64, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 128]]
    # center_best_params = [[10, 256, 5e-3, 0.90, 1e-2, 2023, (50, 130), (2, 8), 128]]
    # east_best_params = [[10, 128, 5e-3, 0.80, 1e-2, 2023, (50, 130), (2, 8), 32]]
    south_east_best_params = [[10, 1024, 5e-3, 0.80, 1e-2, 2023, (50, 130), (2, 8), 32]]

    params_batchsize_list = [param_batch64_values, param_batch128_values, param_batch256_values, param_batch512_values, param_batch1024_values]
    # params_batchsize_list = [param_batch64_values]
    params_lr_list = [param_lr02_values, param_lr03_values, param_lr04_values, param_lr05_values, param_lr06_values]
    params_momentum_list = [momentum_70, momentum_75,momentum_80, momentum_85, momentum_90]
    params_embeddim_list = [BASE_EMB_DIM_16, BASE_EMB_DIM_32, BASE_EMB_DIM_64, BASE_EMB_DIM_128, BASE_EMB_DIM_256]

    # 模型训练模
    for params_item in south_east_best_params:

        train_main_instance = train_model_main.TrainModelMain(train_algo_outcome_dt_end=train_algo_outcome_dt_end,params_item=params_item,
                                                              logger=logger)
        train_main_instance.train_all_model()

        # 测试评估
        eval_main_instance = eval_main.EvaltMain(eval_running_dt_end=eval_running_dt_end, logger=logger,params_item=params_item,
                                                 eval_interval=1)
        eval_main_instance.eval_data()
        logger.info('----------------------------------------测试流程运行结束----------------------------------------')
