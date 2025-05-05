import os, sys, logging
import pandas as pd

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from src.util import serialize


base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


@dataclass_json
@dataclass
class BaseAlgoMixin(object):


    def train(self,
              outcome_offset: int
              # 确定样本的「预测表现期边界」与 「样本运行时间」之间的关系，即 predict_dt = algo_train_running_dt + outcome_offset，确定预测任务类型未T+k（具体的k）
              , outcome_window_len: int  # 确定样本的「预测表现期时间跨度」
              , train_algo_outcome_dt_end: str  # 训练样本算法运行时间，用于确定训练集表现期时间边界，默认训练样本表现期边界为运行时间-1天，即能拿到最新数据的日期。
              , train_algo_interval: int  # 训练样本观察期覆盖天数，确定训练样本时间跨度
              , valid_algo_running_dt: str = None  # 验证样本算法运行时间，用于确定训练样本观察时间边界，默认训练样本观察期边界为 algo_train_running_dt -1
              , valid_algo_interval: int = None  # 训练样本观察期覆盖天数，确定训练样本时间跨度
              , param: dict = None):  # 用于后续参数扩展
        pass

    def batch_predict_with_dt(self,
                              predict_running_dt_end: str  # 预测「样本运行时间」，用于确定预测结观察时间边界，默认预测样本观察期边界为 running_dt_end -1
                              , predict_running_dt_interval: int  # 本观察期覆盖天数，确定预测集样本时间跨度
                              , predict_outcome_offset: int
                              , param: dict = None):  # 用于后续参数扩展

        pass

    def batch_predict_with_index_sample(self,
                                        index_sample: pd.DataFrame
                                        , param: dict = None):  # 用于后续参数扩展
        pass

    def batch_predict_with_dt_and_post_process(self,
                                               predict_running_dt_end: str
                                               # 预测「样本运行时间」，用于确定预测结观察时间边界，默认预测样本观察期边界为 running_dt_end -1
                                               , predict_running_dt_interval: int  # 本观察期覆盖天数，确定预测集样本时间跨度
                                               , predict_outcome_offset: int
                                               , param: dict = None):

        return self.batch_predict_with_dt(predict_running_dt_end,
                                          predict_running_dt_interval,
                                          predict_outcome_offset,
                                          param)
        pass

    def dump(self, root_path: str):  #
        # 1. 保存模型、transform
        # 2. (TO DO 待实验 self.to_json). 保持 Algo 的关键参数，用于load后预测使用，例如 outcome_offset, outcome_window_len，模型路径
        pass

    def load(self, root_path: str):
        # 1. (TO DO 待实验 self.from_json()). 保持 Algo 的关键参数，用于load后预测使用，例如 outcome_offset, outcome_window_len，模型路径
        # 2. 加载模型、transform
        pass

    def dump_data(self, data: pd.DataFrame, dataset_file, file_type='csv'):
        serialize.dataframe_dump(data, dataset_file, file_type)

    def load_data(self, dataset_file, file_type='csv'):
        return serialize.dataframe_read(dataset_file, file_type)


if __name__ == "__main__":
    pass
