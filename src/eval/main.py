import pandas as pd
import logging
import os.path
import sys
import warnings

from src.eval.build_eval_dataset import avg_wt_index_sample
from src.eval.eval_base import EvalBaseMixin
from src.eval.evaluation_test import EvalDate
import src.config.eval_save_path_config as config

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
warnings.filterwarnings("ignore")


class AvgWtEval(EvalBaseMixin):
    def __init__(self):
        self.index_sample = pd.DataFrame()
        self.index_ground_truth = pd.DataFrame()
        self.dt_ground_truth = pd.DataFrame()
        self.outcome_dt_end = None




    def build_eval_set(self, eval_running_dt_end, outcome_offset: int, outcome_window_len: int,
                       eval_interval: int):
        eval_running_dt_end = pd.to_datetime(eval_running_dt_end)
        all_index_sample, all_index_ground_truth, all_dt_ground_truth = [], [], []
        for i in range(eval_interval):
            running_date = eval_running_dt_end - pd.Timedelta(days=i)
            outcome_dt_start = running_date + pd.Timedelta(days=outcome_offset)
            outcome_dt_end = running_date + pd.Timedelta(days=outcome_offset + outcome_window_len - 1)

            single_index_sample, single_index_ground_truth = avg_wt_index_sample(running_date, outcome_dt_start,
                                                                                 outcome_dt_end)

            all_index_sample.append(single_index_sample)
            all_index_ground_truth.append(single_index_ground_truth)
        self.index_sample = pd.concat(all_index_sample, axis=0)
        self.index_ground_truth = pd.concat(all_index_ground_truth, axis=0)
        self.index_ground_truth.to_csv('index_ground_truth.csv')
        self.index_sample.to_csv('index_sample.csv')
        self.outcome_dt_end = eval_running_dt_end + pd.Timedelta(days=outcome_offset + outcome_window_len - 1)


    def get_eval_index_sample(self):
        return pd.read_csv('../data/interim/weight_prediction/index_sample.csv')

    def eval_with_dt(self, predict_result: pd.DataFrame, save_flag: bool = False):
        eval_dt = EvalDate()
        result = eval_dt.pred_avg_wt_eval(predict_result, self.dt_ground_truth, self.outcome_dt_end)
        org_inv_result = eval_dt.pred_avg_wt_org_inv_eval(predict_result, self.dt_ground_truth)
        rearer_pop_result = eval_dt.pred_avg_wt_rearer_pop_eval(predict_result, self.dt_ground_truth)
        groud_truth_dis = eval_dt.groud_truth_distribution_eval(self.dt_ground_truth)

        if save_flag:
            eval_data_path = config.weight_predict_eval_result_dt
            with pd.ExcelWriter(eval_data_path) as writer:
                result.to_excel(writer, sheet_name='基础指标', index=False)
                org_inv_result.to_excel(writer, sheet_name='各公司评价指标', index=False)
                rearer_pop_result.to_excel(writer, sheet_name='各鸡群销售品种评价指标', index=False)
                groud_truth_dis.to_excel(writer, sheet_name='只均重分布', index=False)
        return {'result': result}

    def eval_with_index_sample(self, predict_result: pd.DataFrame, save_flag: bool = False, params: dict={}):
        predict_result = predict_result.dropna(subset=['pred_avg_wt'])
        eval_dt = EvalDate()
        index_ground_truth = pd.read_csv('../data/interim/weight_prediction/index_ground_truth.csv')
        result = eval_dt.pred_avg_wt_eval(predict_result, index_ground_truth, self.outcome_dt_end, params)
        # org_inv_result = eval_dt.pred_avg_wt_org_inv_eval(predict_result, self.index_ground_truth)
        # rearer_pop_result = eval_dt.pred_avg_wt_rearer_pop_eval(predict_result, self.index_ground_truth)
        # groud_truth_dis = eval_dt.groud_truth_distribution_eval(self.index_ground_truth)

        # if save_flag:
        #     eval_data_path = config.weight_predict_eval_result_index
        #     with pd.ExcelWriter(eval_data_path) as writer:
        #         result.to_excel(writer, sheet_name='基础指标', index=False)
        #         org_inv_result.to_excel(writer, sheet_name='各公司评价指标', index=False)
        #         rearer_pop_result.to_excel(writer, sheet_name='各鸡群销售品种评价指标', index=False)
        #         groud_truth_dis.to_excel(writer, sheet_name='只均重分布', index=False)
        return result

