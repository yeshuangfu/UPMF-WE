import importlib
import pandas as pd

import src.config.base_config as config

class EvaltMain(object):

    def __init__(self, **param):
        self.eval_running_dt_end = param.get('eval_running_dt_end')
        self.eval_interval = param.get('eval_interval')
        self.params_item = param.get('params_item')
        self.logger = param.get('logger')

    # 方案一-根据测试生成的测试集，推理数据
    def eval_with_index_sample(self, predict_class_name: str, predict_main_class_name: str,
                               eval_class_name: str, eval_main_class_name: str, alog_config_name: str,
                               eval_running_dt_end: str, params: dict):

        # 构造类对象
        self.logger.info('测试索引样本数据集 %s' % eval_main_class_name)
        # 导包
        predict_class_module = importlib.import_module(predict_class_name)
        eval_class_module = importlib.import_module(eval_class_name)
        alog_config = importlib.import_module(alog_config_name)
        # 获取类
        predict_main_class = getattr(predict_class_module, predict_main_class_name)
        eval_main_class = getattr(eval_class_module, eval_main_class_name)
        # 实例化类
        predict_main_class_cls_instance = predict_main_class(self.params_item)
        eval_main_class_instance = eval_main_class()

        # eval_main_class_instance.build_eval_set(eval_running_dt_end,
        #                                         alog_config.EvalModuleConfig.OUTCOME_OFFSET.value,
        #                                         alog_config.EvalModuleConfig.OUTCOME_WINDOW_LEN.value,
        #                                         15)
        index_sample = eval_main_class_instance.get_eval_index_sample()

        pred_result_index = predict_main_class_cls_instance.batch_predict_with_index_sample(index_sample, {})

        eval_result_index = eval_main_class_instance.eval_with_index_sample(predict_result=pred_result_index,
                                                                            save_flag=False, params=self.params_item)

        # 将数据转换为 DataFrame
        new_df = pd.DataFrame(eval_result_index)

        # 读取现有的 Excel 文件
        existing_df = pd.read_excel('experiments.xlsx', sheet_name='Sheet1')

        # 将新数据追加到现有 DataFrame 中
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # 将整个 DataFrame 重新写回到 Excel 文件中
        combined_df.to_excel('experiments.xlsx', sheet_name='Sheet1', index=False)

        self.logger.info(eval_result_index)
        self.logger.info('索引样本数据集测试完成')

    def eval_data(self):
        self.logger.info("开始测试数据，时间: %s " % self.eval_running_dt_end)
        eval_class_list = config.ModulePath.eval_list.value
        for eval_module in eval_class_list:
            self.eval_with_index_sample(eval_module['predict_class_name'], eval_module['predict_main_class_name'],
                                        eval_module['eval_class_name'], eval_module['eval_main_class_name'],
                                        eval_module['alog_config_name'], self.eval_running_dt_end,
                                        eval_module['params'])
        self.logger.info("测试完成")
