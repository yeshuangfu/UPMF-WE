import copy

import numpy as np
import pandas as pd
import logging
import os.path
import sys
import warnings
from dateutil.relativedelta import relativedelta
import src.config.base_config as config
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
warnings.filterwarnings("ignore")


# 去除数据中佳润公司和内部客户的数据
def drop_jiarun_inner_data(data):
    user_data = pd.read_csv(config.RawData.DIM_CUST_DATA_PATH.value)
    org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
    org_inv_data = org_inv_data.loc[org_inv_data["l3_org_inv_nm"] != '佳润公司']
    outer_data = data[data["cust_dk"].isin(user_data[user_data["inner_company_ind"] == '外部']["cust_dk"])]
    drop_jiarun_inner_data = outer_data[outer_data["org_inv_dk"].isin(org_inv_data["org_inv_dk"])]
    return drop_jiarun_inner_data


# 关联发货数据表和回收表
def shipping_retrieve_table(outcome_dt_start, outcome_dt_end):
    """
    Description
    connecting shipping_data and retrieve_data
    """
    columns = ['shipping_dk', 'org_inv_dk', 'forsale_breeds_nm', 'forsale_breeds_dk']
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    shipping_data = shipping_data[
        (shipping_data['shipping_dt'] >= outcome_dt_start) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
    fg_grade_1 = shipping_data.groupby(columns)["fg_grade_nm"].nunique().reset_index(name='fg_grade')
    fg_grade_1 = fg_grade_1[fg_grade_1['fg_grade'] == 1]
    shipping_data = fg_grade_1.merge(shipping_data, how='left', on=columns)
    shipping_data = shipping_data[shipping_data['fg_grade_nm'] == '1级品']
    shipping_data = shipping_data[(shipping_data.sale_qty > 0) & (shipping_data.sale_price > 0)]
    shipping_data['GMV'] = shipping_data['sale_price'] * shipping_data['sale_qty']
    shipping_data = shipping_data.groupby(by=columns, as_index=False).agg({
        'shipping_dt': 'first', 'shipping_dk': 'first', 'org_inv_dk': 'first', 'cust_dk': 'first',
        'forsale_breeds_nm': 'first', 'forsale_breeds_dk': 'first', 'sale_qty': 'sum', 'GMV': 'sum'})
    shipping_data['sale_price'] = shipping_data['GMV'] / shipping_data['sale_qty']

    retrieve_data = pd.read_csv(config.RawData.REARER_LANDP_RETRIEVE_DATA_PATH.value)
    retrieve_data['retrieve_dt'] = pd.to_datetime(retrieve_data['retrieve_dt'])
    retrieve_data = retrieve_data[
        (retrieve_data['retrieve_dt'] >= (outcome_dt_start - pd.Timedelta(days=7))) &
        (retrieve_data['retrieve_dt'] <= outcome_dt_end)]
    retrieve_data = retrieve_data[retrieve_data['retrieve_qty'] > 0]
    retrieve_data['DMV'] = retrieve_data['retrieve_days'] * retrieve_data['retrieve_qty']
    retrieve_data = retrieve_data.groupby(by=columns, as_index=False).agg({'shipping_dk': 'first',
        'forsale_breeds_nm': 'first', 'forsale_breeds_dk': 'first', 'retrieve_qty': 'sum', 'DMV': 'sum'})
    retrieve_data['retrieve_days'] = retrieve_data['DMV'] / retrieve_data['retrieve_qty']

    data = shipping_data.merge(retrieve_data, how='left', on=columns)

    columns = ['shipping_dt', 'org_inv_dk', 'cust_dk', 'forsale_breeds_nm', 'forsale_breeds_dk']
    non_empty_data = data.dropna(subset=['DMV'])
    non_empty_data = non_empty_data.groupby(by=columns, as_index=False).agg({
        'shipping_dt': 'first', 'org_inv_dk': 'first', 'cust_dk': 'first', 'forsale_breeds_nm': 'first',
        'forsale_breeds_dk': 'first', 'sale_qty': 'sum', 'DMV': 'sum', 'GMV': 'sum'})
    non_empty_data['retrieve_days'] = non_empty_data['DMV'] / non_empty_data['sale_qty']
    non_empty_data['sale_price'] = non_empty_data['GMV'] / non_empty_data['sale_qty']
    retrieve_days_data = non_empty_data.drop(['sale_qty', 'sale_price', 'DMV', 'GMV'], axis=1)

    data = data.groupby(by=columns, as_index=False).agg({
        'shipping_dt': 'first', 'org_inv_dk': 'first', 'cust_dk': 'first', 'forsale_breeds_nm': 'first',
        'forsale_breeds_dk': 'first', 'sale_qty': 'sum', 'GMV': 'sum'})
    data['sale_price'] = data['GMV'] / data['sale_qty']
    data = data.drop(['GMV'], axis=1)
    data = data.merge(retrieve_days_data, how='left', on=columns)
    return data, non_empty_data


# 生成伪样本
def make_pseudo_sample(real_sample, time, step):
    pseudo_sample = []
    for i in range(1, time+1):
        tmp = copy.deepcopy(real_sample)
        tmp['sale_price'] = tmp['sale_price'] + i * step
        tmp['pseudo'] = 1
        pseudo_sample.append(tmp)
    return pd.concat(pseudo_sample, axis=0)


# 生成任务1dt的正样本
def avg_wt(running_date, observation_dt_start, observation_dt_end):
    sale_fcst_day_data = pd.read_csv(config.RawData.REARER_POP_SALE_FCST_DAY_DATA_PATH.value)
    sale_fcst_day_data['stat_dt'] = pd.to_datetime(sale_fcst_day_data['stat_dt'])
    pop_breeds = sale_fcst_day_data[
        (sale_fcst_day_data['stat_dt'] >= observation_dt_start) & (sale_fcst_day_data['stat_dt'] <= observation_dt_end)]
    pop_breeds = pop_breeds[['rearer_pop_dk', 'forsale_breeds_nm', 'forsale_breeds_dk']].drop_duplicates()

    retrieve_data = pd.read_csv(config.RawData.REARER_LANDP_RETRIEVE_DATA_PATH.value)
    ground_truth_data = pop_breeds.merge(retrieve_data, on=['rearer_pop_dk', 'forsale_breeds_nm', 'forsale_breeds_dk'],
                                         how='left')
    org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
    org_inv_data = org_inv_data.loc[org_inv_data["l3_org_inv_nm"] != '佳润公司']
    ground_truth_data = ground_truth_data[ground_truth_data["org_inv_dk"].isin(org_inv_data["org_inv_dk"])]
    ground_truth_data['running_date'] = running_date
    ground_truth_data.rename(columns={"retrieve_dt": "outcome_date"}, inplace=True)
    ground_truth_data = ground_truth_data[ground_truth_data['retrieve_qty'] > 0]
    ground_truth_data = ground_truth_data[
        (ground_truth_data['retrieve_days'] >= 50) & (ground_truth_data['retrieve_days'] < 130)]
    ground_truth_data = ground_truth_data[(ground_truth_data['avg_wt'] > 2) & (ground_truth_data['avg_wt'] <= 8)]
    ground_truth_data = ground_truth_data.groupby(["running_date", "outcome_date", "rearer_pop_dk", "forsale_breeds_nm",
                                                   "forsale_breeds_dk", "retrieve_days"]).apply(
        lambda x: np.average(x['avg_wt'], weights=x['retrieve_qty'])).reset_index(name='avg_wt')
    return ground_truth_data


# 生成任务1index的样本索引和ground truth
def avg_wt_index_sample(running_date, outcome_dt_start, outcome_dt_end):
    retrieve_data = pd.read_csv(config.RawData.REARER_LANDP_RETRIEVE_DATA_PATH.value)
    org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
    org_inv_data = org_inv_data.loc[org_inv_data["l3_org_inv_nm"] != '佳润公司']
    # Todo 2 公司的测试样本，不同数据集的测试样本公司记得换

    # 华南区域
    org_inv_data = org_inv_data[(org_inv_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e001-fbc6c0a802deCCE7AED4')
                                      | (org_inv_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-0612c0a802deCCE7AED4')]

    # 华东区域
    # org_inv_data = org_inv_data[(org_inv_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-07d2c0a802deCCE7AED4')
    #                                   | (org_inv_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-02c1c0a802deCCE7AED4')
    #                                   | (org_inv_data['l3_org_inv_dk'] == 'ZaA5bwEfEADgAaRYCgsFAcznrtQ=')]

    # 华中
    # org_inv_data = org_inv_data[(org_inv_data['l3_org_inv_dk'] == 'y22ajAEcEADgGYceCgsFAsznrtQ=')
    #                             ]

    # 西南
    # org_inv_data = org_inv_data[(org_inv_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-0b5ec0a802deCCE7AED4')]

    retrieve_data = retrieve_data[retrieve_data["org_inv_dk"].isin(org_inv_data["org_inv_dk"])]
    retrieve_data['retrieve_dt'] = pd.to_datetime(retrieve_data['retrieve_dt'])
    outcome_data = retrieve_data[
        (retrieve_data['retrieve_dt'] >= outcome_dt_start) & (retrieve_data['retrieve_dt'] <= outcome_dt_end)]
    outcome_data['running_date'] = running_date
    outcome_data.rename(columns={"retrieve_dt": "outcome_date"}, inplace=True)
    outcome_data = outcome_data[outcome_data['retrieve_qty'] > 0]
    ground_truth_data = outcome_data.groupby(["running_date", "outcome_date", "rearer_pop_dk", "forsale_breeds_nm",
                                              "forsale_breeds_dk", "retrieve_days"]).apply(
        lambda x: np.average(x['avg_wt'], weights=x['retrieve_qty'])).reset_index(name='avg_wt')
    sample_data = ground_truth_data.drop(columns=['avg_wt'])
    return sample_data, ground_truth_data


# 生成任务2.1dt的正样本
def order_prob_first(running_date, outcome_dt_start, outcome_dt_end):
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data = drop_jiarun_inner_data(shipping_data)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    shipping_data = shipping_data.loc[:, ['shipping_dt', 'cust_dk']].drop_duplicates()
    outcome_data = shipping_data[
        (shipping_data['shipping_dt'] >= outcome_dt_start) & (shipping_data['shipping_dt'] <= outcome_dt_end)]

    ground_truth_data = outcome_data.groupby(['cust_dk']).size().reset_index(name="order_days")
    ground_truth_data['running_date'] = running_date
    ground_truth_data['outcome_date'] = outcome_dt_start
    return ground_truth_data


# 生成任务2.1index的样本索引和ground truth
def order_prob_first_index_sample(running_date, outcome_dt_start, outcome_dt_end):
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data = drop_jiarun_inner_data(shipping_data)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    user_start_date = outcome_dt_end - relativedelta(years=1)
    shipping_data = shipping_data[
        (shipping_data['shipping_dt'] > user_start_date) & (shipping_data['shipping_dt'] <= outcome_dt_end)]
    shipping_data = shipping_data.loc[:, ['shipping_dt', 'cust_dk']].drop_duplicates()
    outcome_data = shipping_data[
        (shipping_data['shipping_dt'] >= outcome_dt_start) & (shipping_data['shipping_dt'] <= outcome_dt_end)]

    positive_sample = outcome_data.groupby(['cust_dk']).size().reset_index(name="order_days")
    ground_truth_data = shipping_data.loc[:, ['cust_dk']].drop_duplicates()
    ground_truth_data['running_date'] = running_date
    ground_truth_data['outcome_date'] = outcome_dt_start
    ground_truth_data = ground_truth_data.merge(positive_sample, on=['cust_dk'], how='left')
    ground_truth_data = ground_truth_data.fillna({'order_days': 0})
    ground_truth_data['order_prob'] = np.where(ground_truth_data['order_days'] == 0, 0, 1)

    sample_data = ground_truth_data.drop(columns=['order_days', 'order_prob'])
    return sample_data, ground_truth_data


# 生成任务2.2的样本索引和ground truth
def order_prob_second(running_date, outcome_dt_start, outcome_dt_end):
    org_inv_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    shipping_data = drop_jiarun_inner_data(shipping_data)
    org_inv_data = org_inv_data.loc[org_inv_data["l3_org_inv_nm"] != '佳润公司']
    outcome_data = shipping_data[
        (shipping_data['shipping_dt'] >= outcome_dt_start) & (shipping_data['shipping_dt'] <= outcome_dt_end)]

    ground_truth_data = outcome_data.merge(org_inv_data, on="org_inv_dk", how="left")
    ground_truth_data = ground_truth_data.loc[:, ['cust_dk', 'l4_org_inv_nm']].drop_duplicates()
    ground_truth_data['running_date'] = running_date
    ground_truth_data['outcome_date'] = outcome_dt_start
    ground_truth_data = ground_truth_data.groupby(['cust_dk', 'running_date',
                                                   'outcome_date'])['l4_org_inv_nm'].unique().reset_index()

    sample_data = ground_truth_data.drop(columns=['l4_org_inv_nm'])
    return sample_data, ground_truth_data


# 生成任务3dt的正样本
def order_sale_prob(running_date, outcome_dt_start, outcome_dt_end):
    shipping_data = pd.read_csv(config.RawData.SALES_SHIPPING_DATA_PATH.value)
    shipping_data['shipping_dt'] = pd.to_datetime(shipping_data['shipping_dt'])
    shipping_data = drop_jiarun_inner_data(shipping_data)
    outcome_data = shipping_data[
        (shipping_data['shipping_dt'] >= outcome_dt_start) & (shipping_data['shipping_dt'] <= outcome_dt_end)]

    ground_truth_data = outcome_data.loc[:, ['cust_dk', 'forsale_breeds_nm', 'forsale_breeds_dk']].drop_duplicates()
    ground_truth_data['running_date'] = running_date
    ground_truth_data['outcome_date'] = outcome_dt_start
    ground_truth_data['order_sale_prob'] = 1
    return ground_truth_data


# 生成任务3index的样本索引和ground truth
def order_sale_prob_index_sample(running_date, outcome_dt_start, outcome_dt_end):
    org_inv_nm_data = pd.read_csv(config.RawData.DIM_ORG_INV_DATA_PATH.value)
    org_inv_nm_data = org_inv_nm_data.loc[:, ['org_inv_dk', 'l4_org_inv_nm']]
    outcome_data, non_empty_outcome_data = shipping_retrieve_table(outcome_dt_start, outcome_dt_end)
    outcome_date = drop_jiarun_inner_data(outcome_data)
    outcome_date = outcome_date.rename(columns={'shipping_dt': 'outcome_date'})
    outcome_date['running_date'] = running_date
    non_empty_outcome_data = drop_jiarun_inner_data(non_empty_outcome_data)
    non_empty_outcome_data = non_empty_outcome_data.rename(columns={'shipping_dt': 'outcome_date'})
    non_empty_outcome_data['running_date'] = running_date

    positive_sample = outcome_date.drop(columns=['sale_qty'])
    positive_sample['order_sale_prob'] = 1

    columns = ['outcome_date', 'org_inv_dk', 'forsale_breeds_dk']
    negative_sample_pool = non_empty_outcome_data.groupby(by=columns, as_index=False).agg({
        'running_date': 'first', 'outcome_date': 'first', 'org_inv_dk': 'first', 'forsale_breeds_nm': 'first',
        'forsale_breeds_dk': 'first', 'sale_qty': 'sum', 'GMV': 'sum', 'DMV': 'sum'})
    negative_sample_pool['sale_price'] = negative_sample_pool['GMV'] / negative_sample_pool['sale_qty']
    negative_sample_pool['retrieve_days'] = negative_sample_pool['DMV'] / negative_sample_pool['sale_qty']
    negative_sample_pool = negative_sample_pool.drop(['sale_qty', 'GMV', 'DMV'], axis=1)
    negative_sample_pool['order_sale_prob'] = 0

    positive_l4 = pd.merge(positive_sample, org_inv_nm_data, on="org_inv_dk", how="inner")
    negative_pool_l4 = pd.merge(negative_sample_pool, org_inv_nm_data, on="org_inv_dk", how="inner")
    negative_sample = []
    for date in pd.date_range(start=outcome_dt_start, end=outcome_dt_end, freq='D'):
        date_positive = positive_l4.loc[positive_l4['outcome_date'] == date]
        date_negative = negative_pool_l4.loc[negative_pool_l4['outcome_date'] == date]
        for user in list(set(date_positive.loc[:, "cust_dk"])):
            user_date_positive_sample = date_positive[date_positive['cust_dk'] == user]
            user_date_negative_sample = date_negative[
                date_negative['l4_org_inv_nm'].isin(user_date_positive_sample['l4_org_inv_nm']) &
                ~date_negative['forsale_breeds_dk'].isin(user_date_positive_sample['forsale_breeds_dk'])]
            user_date_negative_sample['cust_dk'] = user
            user_date_negative_sample.drop(columns=['l4_org_inv_nm'], inplace=True)
            negative_sample.append(user_date_negative_sample)
    negative_sample = pd.concat(negative_sample, axis=0)
    pseudo_positive_sample = make_pseudo_sample(positive_sample, time=3, step=-0.3)
    pseudo_negative_sample = make_pseudo_sample(negative_sample, time=3, step=0.3)
    ground_truth_data = pd.concat([positive_sample, negative_sample, pseudo_positive_sample, pseudo_negative_sample],
                                  axis=0)
    ground_truth_data.loc[ground_truth_data["pseudo"] != 1, "pseudo"] = 0
    ground_truth_data = ground_truth_data.loc[ground_truth_data['sale_price'] > 0]
    sample_data = ground_truth_data.drop(columns=['order_sale_prob', 'pseudo'])
    return sample_data, ground_truth_data

