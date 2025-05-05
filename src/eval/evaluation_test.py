import csv
import datetime
import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.spatial as T
from math import sqrt

import sklearn
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_absolute_percentage_error, precision_recall_curve, \
    precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.preprocessing import normalize

import src.config.base_config as config

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
warnings.filterwarnings("ignore")


# 把外部客户按一年内的出勤次数来分高中低频
def get_user_group_day(outcome_dt_end):
    user_data = pd.read_csv(config.RawData.DIM_CUST_DATA_PATH.value)
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data = shipping_data.loc[:, ['shipping_dt', 'cust_dk']].drop_duplicates()
    start_date = outcome_dt_end - relativedelta(years=1)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    shipping_data = shipping_data[
        (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
    outer_user_shipping_data = shipping_data[
                        shipping_data["cust_dk"].isin(user_data[user_data["inner_company_ind"] == '外部']["cust_dk"])]

    outer_user_frequency = outer_user_shipping_data.groupby("cust_dk").size().reset_index(name="count").sort_values(
                           by="count").reset_index(drop=True)
    counts = outer_user_shipping_data.groupby("cust_dk").size()
    sorted_counts = counts.sort_values()
    cumsums = sorted_counts.agg(np.cumsum)
    bins = pd.cut(cumsums, 3, labels=["低频", "中频", "高频"])
    outer_user_frequency['group'] = outer_user_frequency['cust_dk'].map(bins)
    return outer_user_frequency


# 把外部客户按一年内的购买次数来分高中低频
def get_user_group_count(outcome_dt_end):
    user_data = pd.read_csv(config.RawData.DIM_CUST_DATA_PATH.value)
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    start_date = outcome_dt_end - relativedelta(years=1)
    shipping_data = shipping_data[
        (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
    shipping_data = shipping_data.loc[:, ['shipping_dk', 'cust_dk', 'forsale_breeds_dk']].drop_duplicates()
    outer_user_shipping_data = shipping_data[
                        shipping_data["cust_dk"].isin(user_data[user_data["inner_company_ind"] == '外部']["cust_dk"])]

    outer_user_frequency = outer_user_shipping_data.groupby("cust_dk").size().reset_index(name="count").sort_values(
                           by="count").reset_index(drop=True)
    counts = outer_user_shipping_data.groupby("cust_dk").size()
    sorted_counts = counts.sort_values()
    cumsums = sorted_counts.agg(np.cumsum)
    bins = pd.cut(cumsums, 3, labels=["低频", "中频", "高频"])
    outer_user_frequency['group'] = outer_user_frequency['cust_dk'].map(bins)
    return outer_user_frequency


# 获取客户一年内去过的三级公司
def get_user_l3(outcome_dt_end):
    org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    org_inv_data = org_inv_data.loc[org_inv_data["l3_org_inv_nm"] != '佳润公司']
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    start_date = outcome_dt_end - relativedelta(years=1)
    shipping_data = shipping_data[
        (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
    data = pd.merge(shipping_data, org_inv_data, how='inner', on='org_inv_dk')
    user_l3 = data.loc[:, ['cust_dk', 'l3_org_inv_nm']].drop_duplicates()
    return user_l3


# 计算任务2.1每组的评价指标dt
def get_metrics_order_prob_first(group):
    ground_truth_data = group.dropna(subset=['order_days'])
    pred_data = group.dropna(subset=['pred_order_prob'])
    pred_data['order_prob'] = np.where(pred_data['order_days'].isna(), 0, 1)
    data = group.dropna(subset=['order_days', 'pred_order_prob'])
    if len(data) == 0:
        return pd.Series({'Precision': 0, 'Recall': 0, 'AUC': None, 'positive_sample_num': len(ground_truth_data)})
    recall = len(data[data['decision'] == 1]) / len(ground_truth_data)
    if len(pred_data[pred_data['decision'] == 1]) == 0:
        precision = None
    else:
        precision = len(data[data['decision'] == 1]) / len(pred_data[pred_data['decision'] == 1])
    if pred_data["order_prob"].nunique() == 1:
        auc = None
    else:
        auc = roc_auc_score(pred_data['order_prob'], pred_data['pred_order_prob'])
    return pd.Series({'Precision': precision, 'Recall': recall, 'AUC': auc,
                      'positive_sample_num': len(ground_truth_data)})


# 获得目标精度对应的召回率、f1分数和阈值
def get_recall_f1(data, target_precision):
    precisions, recalls, thresholds = precision_recall_curve(data["order_prob"], data["pred_order_prob"])
    index = (np.abs(precisions - target_precision)).argmin()
    precision = precisions[index]
    recall = recalls[index]
    if recall == 1 or precision == 1:
        threshold = None
    else:
        threshold = thresholds[index]
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, threshold


# 计算任务2.1每组的评价指标index
def get_metrics_order_prob_precision(group, target_precision):
    precision, recall, f1, threshold = get_recall_f1(group, target_precision)
    auc = roc_auc_score(group['order_prob'], group['pred_order_prob'])
    return pd.Series({'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': auc})


# 计算任务3每组的评价指标dt
def get_metrics_order_sale_prob_dt(group):
    ground_truth_data = group.dropna(subset=['order_sale_prob'])
    pred_data = group.dropna(subset=['pred_order_sale_prob'])
    data = group.dropna(subset=['order_sale_prob', 'pred_order_sale_prob'])
    if len(data) == 0:
        return pd.Series({'Precision': 0, 'Recall': 0, 'positive_sample_num': len(ground_truth_data)})
    recall = len(data[data['decision'] == 1]) / len(ground_truth_data)
    if len(pred_data[pred_data['decision'] == 1]) == 0:
        precision = None
    else:
        precision = len(data[data['decision'] == 1]) / len(pred_data[pred_data['decision'] == 1])
    return pd.Series({'Precision': precision, 'Recall': recall, 'positive_sample_num': len(ground_truth_data)})


# 计算任务3每组的评价指标index
def get_metrics_order_sale_prob_index(group):
    precision = metrics.precision_score(group['order_sale_prob'], group['decision'], average=None)
    recall = metrics.recall_score(group['order_sale_prob'], group['decision'], average=None)
    f1 = metrics.f1_score(group['order_sale_prob'], group['decision'], average=None)
    acc = accuracy_score(group['order_sale_prob'], group['decision'])
    auc = roc_auc_score(group['order_sale_prob'], group['pred_order_sale_prob'])
    return pd.Series({'Precision_0': precision[0], 'Precision_1': precision[1], 'Recall_0': recall[0],
                      'Recall_1': recall[1], 'F1_0': f1[0], 'F1_1': f1[1], 'ACC': acc, 'AUC': auc})


# 判断list1和list2是否有交集
def has_interaction(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return int(bool(set1 & set2))


# 计算主流销售品种
def MainBreeds(outcome_dt_end):
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    start_date = outcome_dt_end - relativedelta(years=1)
    shipping_data = shipping_data[
        (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
    sum_serise = shipping_data.groupby('forsale_breeds_dk')['sale_qty'].sum().sort_values(ascending=False)
    main_breeds_index = sum_serise[sum_serise > 1000000].index
    return main_breeds_index

# 选择多分类概率最大的类别
def get_max_label(lst):
    return lst.index(max(lst))


# 计算ACC变体的评价指标
def get_acc_variant(data):
    diff = abs(data['order_qty_prob'] - data['pred_order_qty'])
    mask1 = diff == 1
    mask0 = diff == 0
    return (mask0.sum() + mask1.sum() / 2) / len(data['order_qty_prob'])


# 计算任务4每组的评价指标index
def get_metrics_sale_qty_index(group):
    p_weighted = precision_score(group['order_qty_prob'], group['pred_order_qty'], average='weighted')
    r_weighted = recall_score(group['order_qty_prob'], group['pred_order_qty'], average='weighted')
    f1_weighted = f1_score(group['order_qty_prob'], group['pred_order_qty'], average='weighted')
    if group['pred_order_qty_prob'].str.len().max() == group['order_qty_prob'].nunique():
        auc_weighted = roc_auc_score(group['order_qty_prob'], group['pred_order_qty_prob'].tolist(), multi_class='ovr',
                                     average='weighted')
    else:
        auc_weighted = None
    acc_weighted = accuracy_score(group['order_qty_prob'], group['pred_order_qty'])
    acc_variant = get_acc_variant(group)
    kappa = cohen_kappa_score(group['order_qty_prob'], group['pred_order_qty'])
    return pd.Series({'Precision': p_weighted, 'Recall': r_weighted, 'F1': f1_weighted, 'ACC': acc_weighted,
                      'ACC_variant': acc_variant, 'AUC': auc_weighted, 'Kappa': kappa})


# 计算任务4每组的评价指标dt
def get_metrics_sale_qty_dt(group):
    p_weighted = precision_score(group['order_qty_prob'], group['pred_order_qty'], average='weighted')
    r_weighted = recall_score(group['order_qty_prob'], group['pred_order_qty'], average='weighted')
    f1_weighted = f1_score(group['order_qty_prob'], group['pred_order_qty'], average='weighted')
    acc_weighted = accuracy_score(group['order_qty_prob'], group['pred_order_qty'])
    acc_variant = get_acc_variant(group)
    kappa = cohen_kappa_score(group['order_qty_prob'], group['pred_order_qty'])
    return pd.Series({'Precision': p_weighted, 'Recall': r_weighted, 'F1': f1_weighted, 'ACC': acc_weighted,
                      'ACC_variant': acc_variant, 'Kappa': kappa})


# Date预测结果评估类
class EvalDate():
    # 计算任务1基础指标
    def pred_avg_wt_eval(self, pred_data: pd.DataFrame,
                         ground_truth_data: pd.DataFrame,
                         outcome_dt_end: str,
                         params: dict={}):
        columns = ['rearer_pop_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days']
        # pred_data.drop(columns=['Unnamed: 0'], inplace=True)
        # ground_truth_data.drop(columns=['Unnamed: 0'], inplace=True)
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=columns)

        mae = mean_absolute_error(data['avg_wt'], data['pred_avg_wt'])
        # 计算平方误差
        squared_error = (data['avg_wt'] - data['pred_avg_wt']) ** 2

        # 计算均方误差（MSE）
        mse = np.mean(squared_error)
        # 计算均方根误差（RMSE）
        rmse = np.sqrt(mse)

        result = pd.DataFrame({
            "batch_size": params[1],
            "lr": params[2],
            "momentum": params[3],
            "BASE_EMB_DIM": params[8],
            "MAE": [round(mae, 4)],
            "RMSE": [round(rmse, 4)]})
        return result

    # 计算任务1三级公司评价指标
    def pred_avg_wt_org_inv_eval(self, pred_data, ground_truth_data):
        """
        Parameters
        pred_data_path: 预测输出文件
        ground_truth_data_path：ground_truth文件
        raw_data_root：原始文件目录

        Return
        org_inv_result：公司评价指标，running_date|l3_org_inv_nm|l4_org_inv_nm|MAE|MAPE

        Description
        calculation of basic evaluation metrics for each l4_org_inv
        """
        rearer_pop_data = pd.read_csv(config.RawData.REARER_POP_DATA_PATH.value)
        org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
        columns = ['rearer_pop_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days']
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=columns)
        rearer_pop_data = rearer_pop_data.loc[:, ['rearer_pop_dk', 'forsale_breeds_dk', 'org_inv_dk']]
        data = data.merge(rearer_pop_data, on=['rearer_pop_dk', 'forsale_breeds_dk'], how='left')
        data = data.merge(org_inv_data, on='org_inv_dk', how='left')

        org_inv_l3_result = data.groupby(["l3_org_inv_nm"]).apply(
            lambda x: pd.Series({"MAE": mean_absolute_error(x["avg_wt"], x["pred_avg_wt"]),
                                 "MAPE": mean_absolute_percentage_error(x["avg_wt"], x["pred_avg_wt"])})).reset_index()
        org_inv_l3_result["l4_org_inv_nm"] = 'all'
        org_inv_l3_result = org_inv_l3_result.sort_values(by=['MAPE'])

        org_inv_l4_result = data.groupby(["l3_org_inv_nm", "l4_org_inv_nm"]).apply(
            lambda x: pd.Series({"MAE": mean_absolute_error(x["avg_wt"], x["pred_avg_wt"]),
                                 "MAPE": mean_absolute_percentage_error(x["avg_wt"], x["pred_avg_wt"])})).reset_index()
        org_inv_l4_result = org_inv_l4_result.groupby(['l3_org_inv_nm']).apply(
                            lambda x: x.sort_values('MAPE'))
        org_inv_result = pd.concat([org_inv_l4_result, org_inv_l3_result], axis=0)
        return org_inv_result

    # 计算任务1各鸡群销售品种评价指标
    def pred_avg_wt_rearer_pop_eval(self, pred_data, ground_truth_data):
        """
        Parameters
        pred_data_path: 预测输出文件
        ground_truth_data_path：ground_truth文件

        Return
        rearer_pop_result：鸡群评价指标，rearer_pop_dk|forsale_breeds_nm|forsale_breeds_dk|MAE|MAPE|days

        Description
        calculation of basic evaluation metrics for each rearer_pop and forsale_breeds
        """
        columns = ['rearer_pop_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days']
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=columns)
        rearer_pop_result = data.groupby(["rearer_pop_dk", "forsale_breeds_nm", "forsale_breeds_dk"]).apply(
            lambda x: pd.Series({"MAE": mean_absolute_error(x["avg_wt"], x["pred_avg_wt"]),
                                 "MAPE": mean_absolute_percentage_error(x["avg_wt"], x["pred_avg_wt"]),
                                 "days": x["avg_wt"].count()})).reset_index()
        return rearer_pop_result

    # 计算任务1实际均重分布
    def groud_truth_distribution_eval(self, ground_truth_data):
        """
        Parameters
        ground_truth_data_path：ground_truth文件

        Return
        groud_truth_dis：真实值分布，groud_truth_mean|groud_truth_var|groud_truth_median

        Description
        calculate the mean、variance and median of the true values
        """
        last_col_name = ground_truth_data.columns[-1]
        groud_truth_mean = ground_truth_data[last_col_name].mean()
        groud_truth_var = ground_truth_data[last_col_name].var()
        groud_truth_median = ground_truth_data[last_col_name].median()
        groud_truth_dis = pd.DataFrame({"groud_truth_mean": [groud_truth_mean], "groud_truth_var": [groud_truth_var],
                                        "groud_truth_median": [groud_truth_median]})
        return groud_truth_dis

    # 计算任务2.1基础指标
    def pred_order_prob_first_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk'])
        pred_data = pred_data.merge(ground_truth_data, how='left', on=['running_date', 'outcome_date', 'cust_dk'])
        pred_data['order_prob'] = np.where(pred_data['order_days'].isna(), 0, 1)

        recall = len(data[data['decision'] == 1]) / len(ground_truth_data)
        precision = len(data[data['decision'] == 1]) / len(pred_data[pred_data['decision'] == 1])
        auc = roc_auc_score(pred_data['order_prob'], pred_data['pred_order_prob'])
        result = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'AUC': [auc],
                               'positive_sample_num': [len(ground_truth_data)]})
        return result

    # 计算任务2.1出勤客户群评价指标
    def pred_order_prob_first_cust_group_day_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        user_group = get_user_group_day(outcome_dt_end)
        data = pd.merge(ground_truth_data, pred_data, how='outer', on=['running_date', 'outcome_date', 'cust_dk'])
        data = pd.merge(data, user_group, how='inner', on=['cust_dk'])
        ground_truth_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        data = data[data['group'].isin(ground_truth_data['group'])]
        cust_group_result = data.groupby(['group']).apply(get_metrics_order_prob_first).reset_index()
        return cust_group_result

    # 计算任务2.1三级公司评价指标
    def pred_order_prob_first_org_inv_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        user_l3 = get_user_l3(outcome_dt_end)
        data = pd.merge(ground_truth_data, pred_data, how='outer', on=['running_date', 'outcome_date', 'cust_dk'])
        data = pd.merge(data, user_l3, how='inner', on=['cust_dk'])
        ground_truth_data = pd.merge(ground_truth_data, user_l3, how='inner', on=['cust_dk'])
        data = data[data['l3_org_inv_nm'].isin(ground_truth_data['l3_org_inv_nm'])]
        org_inv_result = data.groupby(['l3_org_inv_nm']).apply(get_metrics_order_prob_first).reset_index()
        return org_inv_result

    # 计算任务2.2基础指标
    def pred_order_prob_second_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk'])
        if len(data) == 0:
            result = pd.DataFrame({"Recall_cust": [0], "Recall_l4": [0],
                                   "positive_sample_num": [len(ground_truth_data)]})
            return result
        data['pred_correct'] = data.apply(lambda x: has_interaction(x['l4_org_inv_nm'], x['pred_l4_org_inv_nm']),
                                          axis=1)
        recall_cust = len(data) / len(ground_truth_data)
        recall_l4 = data['pred_correct'].sum() / len(ground_truth_data)
        result = pd.DataFrame({"Recall_cust": [recall_cust], "Recall_l4": [recall_l4],
                               "positive_sample_num": [len(ground_truth_data)]})
        return result

    # 计算任务2.2出勤客户群评价指标
    def pred_order_prob_second_cust_group_day_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        user_group = get_user_group_day(outcome_dt_end)
        user_group_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        data = pd.merge(user_group_data, pred_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk'])
        if len(data) == 0:
            result = pd.DataFrame({"Recall_cust": [0], "Recall_l4": [0],
                                   "positive_sample_num": [len(ground_truth_data)]})
            return result
        data['pred_correct'] = data.apply(lambda x: has_interaction(x['l4_org_inv_nm'], x['pred_l4_org_inv_nm']),
                                          axis=1)
        cust_group_result = data.groupby(['group']).apply(lambda x: pd.Series(
            {"Recall_cust": x["pred_correct"].count(), "Recall_l4": x["pred_correct"].sum()})).reset_index()
        tp_fn = user_group_data.groupby(['group']).size().reset_index(name="positive_sample_num")
        cust_group_result = pd.merge(cust_group_result, tp_fn, on=['group'], how='outer')
        cust_group_result['Recall_cust'] = cust_group_result['Recall_cust'] / cust_group_result['positive_sample_num']
        cust_group_result['Recall_l4'] = cust_group_result['Recall_l4'] / cust_group_result['positive_sample_num']
        return cust_group_result

    # 计算任务2.2三级公司评价指标
    def pred_order_prob_second_org_inv_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        user_l3 = get_user_l3(outcome_dt_end)
        user_l3_data = pd.merge(ground_truth_data, user_l3, how='inner', on=['cust_dk'])
        data = pd.merge(user_l3_data, pred_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk'])
        if len(data) == 0:
            result = pd.DataFrame({"Recall_cust": [0], "Recall_l4": [0],
                                   "positive_sample_num": [len(ground_truth_data)]})
            return result
        data['pred_correct'] = data.apply(lambda x: has_interaction(x['l4_org_inv_nm'], x['pred_l4_org_inv_nm']),
                                          axis=1)

        org_inv_result = data.groupby(['l3_org_inv_nm']).apply(lambda x: pd.Series(
            {"Recall_cust": x["pred_correct"].count(), "Recall_l4": x["pred_correct"].sum()})).reset_index()
        tp_fn = user_l3_data.groupby(['l3_org_inv_nm']).size().reset_index(name="positive_sample_num")
        org_inv_result = pd.merge(org_inv_result, tp_fn, on=['l3_org_inv_nm'], how='outer')
        org_inv_result['Recall_cust'] = org_inv_result['Recall_cust'] / org_inv_result['positive_sample_num']
        org_inv_result['Recall_l4'] = org_inv_result['Recall_l4'] / org_inv_result['positive_sample_num']
        return org_inv_result

    # 计算任务3基础指标
    def pred_order_sale_prob_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk', 'forsale_breeds_nm', 'forsale_breeds_dk'])
        ground_truth_list = ground_truth_data.groupby(['cust_dk', 'running_date',
                                                       'outcome_date'])['forsale_breeds_dk'].unique().reset_index()
        pred_top3_list = pred_data.sort_values('pred_order_sale_prob', ascending=False).groupby(
                         ['cust_dk', 'running_date', 'outcome_date']).head(3)
        pred_top3_list = pred_top3_list.groupby(['cust_dk', 'running_date', 'outcome_date'])['forsale_breeds_dk'].apply(
                         list).reset_index(name='pred_forsale_breeds_dk')
        list_data = pd.merge(pred_top3_list, ground_truth_list, how='inner',
                             on=['running_date', 'outcome_date', 'cust_dk'])

        precision = len(data[data['decision'] == 1]) / len(pred_data[pred_data['decision'] == 1])
        recall = len(data[data['decision'] == 1]) / len(ground_truth_data)
        if len(list_data) == 0:
            recall_top3 = 0
        else:
            list_data['pred_correct'] = list_data.apply(lambda x: has_interaction(x['forsale_breeds_dk'],
                                                                                  x['pred_forsale_breeds_dk']), axis=1)
            recall_top3 = list_data['pred_correct'].sum() / len(ground_truth_list)
        result = pd.DataFrame({'Precision': [precision], 'Recall': [recall],
                               "positive_sample_num": [len(ground_truth_data)], 'Recall_top3': [recall_top3],
                               'cust_num': [len(ground_truth_list)]})
        return result

    # 计算任务3购买次数客户群评价指标
    def pred_order_sale_prob_cust_group_count_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        user_group = get_user_group_count(outcome_dt_end)
        data = pd.merge(pred_data, ground_truth_data, how='outer',
                        on=['running_date', 'outcome_date', 'cust_dk', 'forsale_breeds_nm', 'forsale_breeds_dk'])
        data = pd.merge(data, user_group, how='inner', on=['cust_dk'])
        ground_truth_list = ground_truth_data.groupby(['cust_dk', 'running_date', 'outcome_date'])[
            'forsale_breeds_dk'].unique().reset_index()
        ground_truth_list = pd.merge(ground_truth_list, user_group, how='inner', on=['cust_dk'])
        pred_top3_list = pred_data.sort_values('pred_order_sale_prob', ascending=False).groupby(
            ['cust_dk', 'running_date', 'outcome_date']).head(3)
        pred_top3_list = pred_top3_list.groupby(['cust_dk', 'running_date', 'outcome_date'])['forsale_breeds_dk'].apply(
            list).reset_index(name='pred_forsale_breeds_dk')
        list_data = pd.merge(ground_truth_list, pred_top3_list, how='inner',
                             on=['running_date', 'outcome_date', 'cust_dk'])

        data = data[data['group'].isin(ground_truth_list['group'])]
        result = data.groupby(['group']).apply(get_metrics_order_sale_prob_dt).reset_index()
        tp_fn = ground_truth_list.groupby(['group']).size().reset_index(name="cust_num")
        if len(list_data) == 0:
            result['Recall_top3'] = 0
            cust_group_result = pd.merge(result, tp_fn, on=['group'], how='outer')
            return cust_group_result
        list_data['Recall_top3'] = list_data.apply(lambda x: has_interaction(x['forsale_breeds_dk'],
                                                                             x['pred_forsale_breeds_dk']), axis=1)
        cust_group_result = list_data.groupby(['group'])["Recall_top3"].sum().reset_index()
        cust_group_result = pd.merge(cust_group_result, tp_fn, on=['group'], how='inner')
        cust_group_result['Recall_top3'] = cust_group_result['Recall_top3'] / cust_group_result['cust_num']
        cust_group_result = pd.merge(result, cust_group_result, on=['group'], how='outer')
        return cust_group_result

    # 计算任务3三级公司评价指标
    def pred_order_sale_prob_org_inv_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        user_l3 = get_user_l3(outcome_dt_end)
        data = pd.merge(pred_data, ground_truth_data, how='outer',
                        on=['running_date', 'outcome_date', 'cust_dk', 'forsale_breeds_nm', 'forsale_breeds_dk'])
        data = pd.merge(data, user_l3, how='inner', on=['cust_dk'])
        ground_truth_list = ground_truth_data.groupby(['cust_dk', 'running_date', 'outcome_date'])[
            'forsale_breeds_dk'].unique().reset_index()
        ground_truth_list = pd.merge(ground_truth_list, user_l3, how='inner', on=['cust_dk'])
        pred_top3_list = pred_data.sort_values('pred_order_sale_prob', ascending=False).groupby(
            ['cust_dk', 'running_date', 'outcome_date']).head(3)
        pred_top3_list = pred_top3_list.groupby(['cust_dk', 'running_date', 'outcome_date'])['forsale_breeds_dk'].apply(
            list).reset_index(name='pred_forsale_breeds_dk')
        list_data = pd.merge(ground_truth_list, pred_top3_list, how='inner',
                             on=['running_date', 'outcome_date', 'cust_dk'])

        data = data[data['l3_org_inv_nm'].isin(ground_truth_list['l3_org_inv_nm'])]
        result = data.groupby(['l3_org_inv_nm']).apply(get_metrics_order_sale_prob_dt).reset_index()
        tp_fn = ground_truth_list.groupby(['l3_org_inv_nm']).size().reset_index(name="cust_num")
        if len(list_data) == 0:
            result['Recall_top3'] = 0
            org_inv_result = pd.merge(result, tp_fn, on=['l3_org_inv_nm'], how='outer')
            return org_inv_result
        list_data['Recall_top3'] = list_data.apply(lambda x: has_interaction(x['forsale_breeds_dk'],
                                                                             x['pred_forsale_breeds_dk']), axis=1)
        org_inv_result = list_data.groupby(['l3_org_inv_nm'])["Recall_top3"].sum().reset_index()
        org_inv_result = pd.merge(org_inv_result, tp_fn, on=['l3_org_inv_nm'], how='inner')
        org_inv_result['Recall_top3'] = org_inv_result['Recall_top3'] / org_inv_result['cust_num']
        org_inv_result = pd.merge(result, org_inv_result, on=['l3_org_inv_nm'], how='outer')
        return org_inv_result

    # 计算任务4基础评价指标
    def pred_sale_qty_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                                                                       'forsale_breeds_nm', 'forsale_breeds_dk'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))

        p_weighted = precision_score(data['order_qty_prob'], data['pred_order_qty'], average='weighted')
        r_weighted = recall_score(data['order_qty_prob'], data['pred_order_qty'], average='weighted')
        f1_weighted = f1_score(data['order_qty_prob'], data['pred_order_qty'], average='weighted')
        acc_weighted = accuracy_score(data['order_qty_prob'], data['pred_order_qty'])
        acc_variant = get_acc_variant(data)
        kappa = cohen_kappa_score(data['order_qty_prob'], data['pred_order_qty'])
        all_result = pd.DataFrame({'Precision': p_weighted, 'Recall': r_weighted, 'F1': f1_weighted,
                                   'ACC': acc_weighted, 'ACC_variant': acc_variant,
                                   'Kappa': kappa}, index=['all'])
        all_result['sample_num'] = len(ground_truth_data)
        all_result['recognition_num'] = len(data)
        all_result['recognition_rate'] = all_result['recognition_num'] / all_result['sample_num']

        p = pd.DataFrame(precision_score(data['order_qty_prob'], data['pred_order_qty'], average=None))
        r = pd.DataFrame(recall_score(data['order_qty_prob'], data['pred_order_qty'], average=None))
        f = pd.DataFrame(f1_score(data['order_qty_prob'], data['pred_order_qty'], average=None))
        acc = data.groupby(['order_qty_prob']).apply(lambda x: accuracy_score(x['order_qty_prob'],
                                                                              x['pred_order_qty']))
        acc_variant = data.groupby(['order_qty_prob']).apply(get_acc_variant)
        sample_num = ground_truth_data.groupby(['order_qty_prob']).size()
        recognition_num = data.groupby(['order_qty_prob']).size()

        result = pd.concat([p, r, f, acc, acc_variant, sample_num, sample_num, recognition_num], axis=1)
        result.columns = ['Precision', 'Recall', 'F1', 'ACC', 'ACC_variant', 'sample_proportion', 'sample_num',
                          'recognition_num']
        result['sample_proportion'] = result['sample_proportion'] / len(ground_truth_data)
        result['recognition_rate'] = result['recognition_num'] / result['sample_num']
        result = pd.concat([all_result, result], axis=0).reset_index()
        return result

    # 计算任务4混淆矩阵
    def pred_sale_qty_confu_matrix_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                                                                       'forsale_breeds_nm', 'forsale_breeds_dk'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))
        cm = confusion_matrix(data['order_qty_prob'], data['pred_order_qty'])
        cm = normalize(cm, norm='l1', axis=1)
        label = pd.Series(sorted(set(data['order_qty_prob']).union(set(data['pred_order_qty']))))
        true_label = label.astype(str) + '-true'
        pred_label = label.astype(str) + '-pred'
        cm = pd.DataFrame(cm, columns=pred_label, index=true_label)
        return cm

    # 计算任务4客户购买次数评价指标
    def pred_sale_qty_cust_group_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                                                                       'forsale_breeds_nm', 'forsale_breeds_dk'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))
        user_group = get_user_group_count(outcome_dt_end)
        user_group_data = pd.merge(data, user_group, how='inner', on=['cust_dk'])
        result = user_group_data.groupby(['group']).apply(get_metrics_sale_qty_dt).reset_index()
        all_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        recognition_data = user_group_data.groupby(['group']).size().reset_index(name="recognition_num")
        all_data = all_data.groupby(['group']).size().reset_index(name="sample_num")
        cust_group_recognition = pd.merge(recognition_data, all_data, how='inner', on=['group'])
        cust_group_result = pd.merge(result, cust_group_recognition, how='inner', on=['group'])
        cust_group_result['recognition_rate'] = cust_group_result['recognition_num'] / cust_group_result['sample_num']
        return cust_group_result

    # 计算任务4销售品种评价指标
    def pred_sale_qty_forsale_breeds_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
        shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
        start_date = outcome_dt_end - relativedelta(years=1)
        shipping_data = shipping_data[
            (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
        shipping_data = shipping_data[(shipping_data['sale_qty'] > 0) & (shipping_data['sale_price'] > 0)]
        sale_qty = shipping_data.groupby('forsale_breeds_dk')['sale_qty'].sum().reset_index()
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                                                                       'forsale_breeds_nm', 'forsale_breeds_dk'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))

        forsale_breeds_result = data.groupby(['forsale_breeds_nm', 'forsale_breeds_dk']).apply(
            get_metrics_sale_qty_dt).reset_index()
        forsale_breeds_result = forsale_breeds_result.merge(sale_qty, on='forsale_breeds_dk', how='left')
        forsale_breeds_result.sort_values(by=['sale_qty'], ascending=False, inplace=True)
        return forsale_breeds_result


# index sample预测结果评估类
class EvalIndex():
    # 计算任务2.1基础指标
    def pred_order_prob_first_eval(self, pred_data, ground_truth_data, target_precision):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk'])

        precision, recall, f1, threshold = get_recall_f1(data, target_precision)
        auc = roc_auc_score(data['order_prob'], data['pred_order_prob'])
        result = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'F1': [f1], 'AUC': [auc]})
        result['positive_sample_num'] = len(ground_truth_data[ground_truth_data['order_prob'] == 1])
        result['negative_sample_num'] = len(ground_truth_data[ground_truth_data['order_prob'] == 0])
        result['sample_num'] = len(ground_truth_data)
        result['recognition_num'] = len(data)
        result['recognition_rate'] = result['recognition_num'] / result['sample_num']
        return result

    # 计算任务2.1出勤客户群评价指标
    def pred_order_prob_first_cust_group_day_eval(self, pred_data, ground_truth_data, outcome_dt_end, target_precision):
        user_group = get_user_group_day(outcome_dt_end)
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk'])
        user_group_data = pd.merge(data, user_group, how='inner', on=['cust_dk'])

        result = user_group_data.groupby(['group']).apply(get_metrics_order_prob_precision,
                                                          target_precision).reset_index()
        all_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        recognition_data = user_group_data.groupby(["group"]).size().reset_index(name="recognition_num")
        all_data = all_data.groupby(["group"]).size().reset_index(name="sample_num")
        cust_group_recognition = pd.merge(recognition_data, all_data, how='inner', on=['group'])
        cust_group_result = pd.merge(result, cust_group_recognition, how='inner', on=['group'])
        cust_group_result['recognition_rate'] = cust_group_result['recognition_num'] / cust_group_result['sample_num']
        prob_dis = user_group_data.groupby(['group'])['pred_order_prob'].describe(
            percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
        cust_group_result = cust_group_result.merge(prob_dis, on='group', how='left')
        cust_group_result.drop(columns=['count'], inplace=True)
        return cust_group_result

    # 计算任务2.1三级公司评价指标
    def pred_order_prob_first_org_inv_eval(self, pred_data, ground_truth_data, outcome_dt_end, target_precision):
        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk'])
        cust_l3 = get_user_l3(outcome_dt_end)
        data = pd.merge(cust_l3, data, how='inner', on=['cust_dk'])

        l3 = data.groupby('l3_org_inv_nm')["order_prob"].nunique()
        l3 = l3[l3 > 1].index.to_frame()
        data = data[data["l3_org_inv_nm"].isin(l3["l3_org_inv_nm"])]

        org_inv_result = data.groupby(["l3_org_inv_nm"]).apply(get_metrics_order_prob_precision,
                                                               target_precision).reset_index()
        return org_inv_result

    # 计算任务3基础指标
    def pred_order_sale_prob_eval(self, pred_data, ground_truth_data, flag):
        if flag == 'pseudo':
            ground_truth_data = ground_truth_data.loc[ground_truth_data['pseudo'] == 1]
        elif flag == 'real':
            ground_truth_data = ground_truth_data.loc[ground_truth_data['pseudo'] == 0]
        else:
            pass

        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk', 'org_inv_dk', 'forsale_breeds_nm',
                            'forsale_breeds_dk', 'retrieve_days', 'sale_price'])

        precision = metrics.precision_score(data['order_sale_prob'], data['decision'], average=None)
        recall = metrics.recall_score(data['order_sale_prob'], data['decision'], average=None)
        f1 = metrics.f1_score(data['order_sale_prob'], data['decision'], average=None)
        acc = accuracy_score(data['order_sale_prob'], data['decision'])
        auc = roc_auc_score(data['order_sale_prob'], data['pred_order_sale_prob'])
        result = pd.DataFrame({'Precision_0': [precision[0]], 'Precision_1': [precision[1]], 'Recall_0': [recall[0]],
                               'Recall_1': [recall[1]], 'F1_0': [f1[0]], 'F1_1': [f1[1]], 'ACC': [acc], 'AUC': [auc]})
        result['sample_num'] = len(ground_truth_data)
        result['recognition_num'] = len(data)
        result['recognition_rate'] = result['recognition_num'] / result['sample_num']
        return result

    # 计算任务3出勤客户群评价指标
    def pred_order_sale_prob_cust_group_day_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk', 'org_inv_dk', 'forsale_breeds_nm',
                            'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        user_group = get_user_group_day(outcome_dt_end)
        user_group_data = pd.merge(data, user_group, how='inner', on=['cust_dk'])
        result = user_group_data.groupby(['pseudo', 'group']).apply(get_metrics_order_sale_prob_index).reset_index()
        all_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        recognition_data = user_group_data.groupby(['pseudo', 'group']).size().reset_index(name="recognition_num")
        all_data = all_data.groupby(['pseudo', 'group']).size().reset_index(name="sample_num")
        cust_group_recognition = pd.merge(recognition_data, all_data, how='inner', on=['pseudo', 'group'])
        cust_group_result = pd.merge(result, cust_group_recognition, how='inner', on=['pseudo', 'group'])
        cust_group_result['recognition_rate'] = cust_group_result['recognition_num']/cust_group_result['sample_num']
        return cust_group_result

    # 计算任务3购买次数客户群评价指标
    def pred_order_sale_prob_cust_group_count_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk', 'org_inv_dk', 'forsale_breeds_nm',
                            'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        user_group = get_user_group_count(outcome_dt_end)
        user_group_data = pd.merge(data, user_group, how='inner', on=['cust_dk'])
        result = user_group_data.groupby(['pseudo', 'group']).apply(get_metrics_order_sale_prob_index).reset_index()
        all_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        recognition_data = user_group_data.groupby(['pseudo', 'group']).size().reset_index(name="recognition_num")
        all_data = all_data.groupby(['pseudo', 'group']).size().reset_index(name="sample_num")
        cust_group_recognition = pd.merge(recognition_data, all_data, how='inner', on=['pseudo', 'group'])
        cust_group_result = pd.merge(result, cust_group_recognition, how='inner', on=['pseudo', 'group'])
        cust_group_result['recognition_rate'] = cust_group_result['recognition_num']/cust_group_result['sample_num']
        return cust_group_result

    # 计算任务3三级公司、四级公司评价指标
    def pred_order_sale_prob_org_inv_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk', 'org_inv_dk', 'forsale_breeds_nm',
                            'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        # 关联三级、四级公司
        org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
        org_inv_data = org_inv_data.loc[:, ["org_inv_dk", "l3_org_inv_nm", "l4_org_inv_nm"]].drop_duplicates()
        data = pd.merge(data, org_inv_data, how='inner', on=['org_inv_dk'])
        # 过滤出在正样本里出现过的三级公司的数据
        l3 = data.groupby('l3_org_inv_nm')["order_sale_prob"].nunique()
        l3 = l3[l3 > 1].index.to_frame()
        data = data[data["l3_org_inv_nm"].isin(l3["l3_org_inv_nm"])]
        org_inv_result = data.groupby(['pseudo',
                                       'l3_org_inv_nm']).apply(get_metrics_order_sale_prob_index).reset_index()
        # 过滤出在正负样本里都出现过的四级公司的数据
        org_inv_result['l4_org_inv_nm'] = 'all'
        l4 = data.groupby('l4_org_inv_nm')["order_sale_prob"].nunique()
        l4 = l4[l4 > 1].index.to_frame()
        data = data[data["l4_org_inv_nm"].isin(l4["l4_org_inv_nm"])]
        org_inv_l4_result = data.groupby(['pseudo', 'l3_org_inv_nm',
                                          'l4_org_inv_nm']).apply(get_metrics_order_sale_prob_index).reset_index()
        org_inv_result = pd.concat([org_inv_l4_result, org_inv_result], axis=0)
        return org_inv_result

    # 计算任务3各销售品种评价指标
    def pred_order_sale_prob_forsale_breeds_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
        start_date = outcome_dt_end - relativedelta(years=1)
        shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
        shipping_data = shipping_data[
            (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
        shipping_data = shipping_data[(shipping_data.sale_qty > 0) & (shipping_data.sale_price > 0)]
        data = pd.merge(pred_data, ground_truth_data, how='inner',
                        on=['running_date', 'outcome_date', 'cust_dk', 'org_inv_dk', 'forsale_breeds_nm',
                            'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        # 过滤出在正负样本里都出现过的销售品种的数据
        forsale_breeds = data.groupby('forsale_breeds_dk')["order_sale_prob"].nunique()
        forsale_breeds = forsale_breeds[forsale_breeds > 1].index.to_frame()
        data = data[data["forsale_breeds_dk"].isin(forsale_breeds["forsale_breeds_dk"])]

        forsale_breeds_result = data.groupby(['pseudo', 'forsale_breeds_nm', 'forsale_breeds_dk']).apply(
            get_metrics_order_sale_prob_index).reset_index()
        sale_qty = shipping_data.groupby('forsale_breeds_dk')['sale_qty'].sum().reset_index()
        forsale_breeds_result = forsale_breeds_result.merge(sale_qty, on='forsale_breeds_dk', how='left')
        forsale_breeds_result.sort_values(by=['pseudo', 'sale_qty'], ascending=False, inplace=True)
        return forsale_breeds_result

 # 计算任务4基础评价指标
    def pred_sale_qty_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                        'org_inv_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))

        p_weighted = precision_score(data['order_qty_prob'], data['pred_order_qty'], average='weighted')
        r_weighted = recall_score(data['order_qty_prob'], data['pred_order_qty'], average='weighted')
        f1_weighted = f1_score(data['order_qty_prob'], data['pred_order_qty'], average='weighted')
        if data['pred_order_qty_prob'].str.len().max() == data['order_qty_prob'].nunique():
            auc_weighted = roc_auc_score(data['order_qty_prob'].tolist(), data['pred_order_qty_prob'].tolist(),
                                         multi_class='ovr', average='weighted')
            auc = pd.DataFrame(roc_auc_score(data['order_qty_prob'], data['pred_order_qty_prob'].tolist(),
                                             multi_class='ovr', average=None))
        else:
            auc_weighted = None
            auc = pd.DataFrame([None])
        acc_weighted = accuracy_score(data['order_qty_prob'], data['pred_order_qty'])
        acc_variant = get_acc_variant(data)
        kappa = cohen_kappa_score(data['order_qty_prob'], data['pred_order_qty'])
        all_result = pd.DataFrame({'Precision': p_weighted, 'Recall': r_weighted, 'F1': f1_weighted,
                                   'ACC': acc_weighted, 'ACC_variant': acc_variant, 'AUC': auc_weighted,
                                   'Kappa': kappa}, index=['all'])
        all_result['sample_num'] = len(ground_truth_data)
        all_result['recognition_num'] = len(data)
        all_result['recognition_rate'] = all_result['recognition_num'] / all_result['sample_num']

        p = pd.DataFrame(precision_score(data['order_qty_prob'], data['pred_order_qty'], average=None))
        r = pd.DataFrame(recall_score(data['order_qty_prob'], data['pred_order_qty'], average=None))
        f = pd.DataFrame(f1_score(data['order_qty_prob'], data['pred_order_qty'], average=None))
        acc = data.groupby(['order_qty_prob']).apply(lambda x: accuracy_score(x['order_qty_prob'],
                                                                              x['pred_order_qty']))
        acc_variant = data.groupby(['order_qty_prob']).apply(get_acc_variant)
        sample_num = ground_truth_data.groupby(['order_qty_prob']).size()
        recognition_num = data.groupby(['order_qty_prob']).size()

        result = pd.concat([p, r, f, acc, acc_variant, auc, sample_num, sample_num, recognition_num], axis=1)
        result.columns = ['Precision', 'Recall', 'F1', 'ACC', 'ACC_variant', 'AUC', 'sample_proportion', 'sample_num',
                          'recognition_num']
        result['sample_proportion'] = result['sample_proportion'] / len(ground_truth_data)
        result['recognition_rate'] = result['recognition_num'] / result['sample_num']
        result = pd.concat([all_result, result], axis=0).reset_index()#.reset_index(names='Label')
        return result

    # 计算任务4混淆矩阵
    def pred_sale_qty_confu_matrix_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                                                                       'org_inv_dk', 'forsale_breeds_nm',
                                                                       'forsale_breeds_dk', 'retrieve_days',
                                                                       'sale_price'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))
        cm = confusion_matrix(data['order_qty_prob'], data['pred_order_qty'])
        cm = normalize(cm, norm='l1', axis=1)
        label = pd.Series(sorted(set(data['order_qty_prob']).union(set(data['pred_order_qty']))))
        true_label = label.astype(str) + '-true'
        pred_label = label.astype(str) + '-pred'
        cm = pd.DataFrame(cm, columns=pred_label, index=true_label)
        return cm

    # 计算任务4各购买次数客户群评价指标
    def pred_sale_qty_cust_group_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                        'org_inv_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))
        user_group = get_user_group_count(outcome_dt_end)
        user_group_data = pd.merge(data, user_group, how='inner', on=['cust_dk'])

        result = user_group_data.groupby(['group']).apply(get_metrics_sale_qty_index).reset_index()
        all_data = pd.merge(ground_truth_data, user_group, how='inner', on=['cust_dk'])
        recognition_data = user_group_data.groupby(['group']).size().reset_index(name="recognition_num")
        all_data = all_data.groupby(['group']).size().reset_index(name="sample_num")
        cust_group_recognition = pd.merge(recognition_data, all_data, how='inner', on=['group'])
        cust_group_result = pd.merge(result, cust_group_recognition, how='inner', on=['group'])
        cust_group_result['recognition_rate'] = cust_group_result['recognition_num'] / cust_group_result['sample_num']
        return cust_group_result

    # 计算任务4各公司评价指标
    def pred_sale_qty_org_inv_eval(self, pred_data, ground_truth_data):
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                        'org_inv_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))
        org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
        org_inv_data = org_inv_data[["org_inv_dk", "l3_org_inv_nm"]].drop_duplicates()
        data = pd.merge(data, org_inv_data, how='inner', on=['org_inv_dk'])

        result = data.groupby(['l3_org_inv_nm']).apply(get_metrics_sale_qty_index).reset_index()
        all_data = pd.merge(ground_truth_data, org_inv_data, how='inner', on=['org_inv_dk'])
        recognition_data = data.groupby(['l3_org_inv_nm']).size().reset_index(name="recognition_num")
        all_data = all_data.groupby(['l3_org_inv_nm']).size().reset_index(name="sample_num")
        org_inv_recognition = pd.merge(recognition_data, all_data, how='inner', on=['l3_org_inv_nm'])
        org_inv_result = pd.merge(result, org_inv_recognition, how='inner', on=['l3_org_inv_nm'])
        org_inv_result['recognition_rate'] = org_inv_result['recognition_num'] / org_inv_result['sample_num']
        return org_inv_result

    # 计算任务4各销售品种评价指标
    def pred_sale_qty_forsale_breeds_eval(self, pred_data, ground_truth_data, outcome_dt_end):
        shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
        shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
        start_date = outcome_dt_end - relativedelta(years=1)
        shipping_data = shipping_data[
            (shipping_data['shipping_dt'] > start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
        shipping_data = shipping_data[(shipping_data['sale_qty'] > 0) & (shipping_data['sale_price'] > 0)]
        sale_qty = shipping_data.groupby('forsale_breeds_dk')['sale_qty'].sum().reset_index()
        data = pd.merge(pred_data, ground_truth_data, how='inner', on=['running_date', 'outcome_date', 'cust_dk',
                        'org_inv_dk', 'forsale_breeds_nm', 'forsale_breeds_dk', 'retrieve_days', 'sale_price'])
        data['pred_order_qty'] = data['pred_order_qty_prob'].apply(lambda x: get_max_label(x))

        forsale_breeds_result = data.groupby(['forsale_breeds_nm', 'forsale_breeds_dk']).apply(
            get_metrics_sale_qty_index).reset_index()
        forsale_breeds_result = forsale_breeds_result.merge(sale_qty, on='forsale_breeds_dk', how='left')
        forsale_breeds_result.sort_values(by=['sale_qty'], ascending=False, inplace=True)
        return forsale_breeds_result