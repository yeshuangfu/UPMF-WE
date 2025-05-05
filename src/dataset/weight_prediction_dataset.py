import os
import sys
import logging
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from .base_dataset import BaseDataSet
import src.config.base_config as base_config


class WeightPredictionDataset(BaseDataSet):
    """
    Parameters
    ----------
    retrieve_data_filename: 回收数据文件名。默认为"回收.csv"

    start_date: 开始日期。

    end_date: 结束日期。

    (Optional)retrieve_days_interval: 训练天龄范围，筛选出在该范围的天龄进行训练。默认为(50, 130)

    (Optional)avg_wt_interval: 训练均重范围，筛选出在该范围的均重进行训练。默认为(2, 8)

    Usage
    ------
    1. 在回收数据中索引出"rearer_pop_dk", "forsale_breeds_dk", "retrieve_dt", "retrieve_days", "avg_wt"
    2. 根据"rearer_pop_dk", "forsale_breeds_dk"两个在chicken_attributes中索引出"l3_breeds_class_nm", "feed_breeds_nm",
        "gender", "breeds_class_nm", "rearer_dk", "org_inv_dk", "tech_bk"
    3. 根据"retrieve_dt"获得月份"month"

    Return
    ------
    data/interim/weight_prediction.csv
        l3_breeds_class_nm | feed_breeds_nm | gender | breeds_class_nm | rearer_dk | org_inv_dk | l3_org_inv_nm | l4_org_inv_nm | tech_bk | month | retrieve_days | avg_wt | date

    Example
    -------
    >>> dataset = WeightPredictionDataset()
    >>> dataset.bulid_dataset()
    >>> dataset.dump_dataset("/".join([config.interim_dir, "weight_prediction.csv"])
    """

    def __init__(self, **param):
        super().__init__(param)
        # self.start_date = param.get('start_date', '2023-08-02')
        self.start_date = '2023-05-29'
        self.end_date = '2023-09-15'
        self.min_day, self.max_day = param.get('retrieve_days_interval', (50, 130))
        self.min_wt, self.max_wt = param.get('avg_wt_interval', (2, 8))
        self.raw_feature_names = param.get('raw_feature_names', [])

        retrieve_data = pd.read_csv(base_config.RawData.REARER_LANDP_RETRIEVE_DATA_PATH.value)
        retrieve_data = retrieve_data[
            ["rearer_pop_dk", "forsale_breeds_dk", "retrieve_dt", "retrieve_days", "avg_wt", "retrieve_qty"]]
        retrieve_data = retrieve_data[(retrieve_data['retrieve_dt'] >= self.start_date) &
                                      (retrieve_data['retrieve_dt'] <= self.end_date) &
                                      (retrieve_data['retrieve_days'] >= self.min_day) &
                                      (retrieve_data['retrieve_days'] <= self.max_day) &
                                      (retrieve_data['avg_wt'] >= self.min_wt) &
                                      (retrieve_data['avg_wt'] <= self.max_wt)]
        self.retrieve_data = retrieve_data[
            ["retrieve_dt", "rearer_pop_dk", "forsale_breeds_dk", "retrieve_days", "avg_wt", "retrieve_qty"]]
        self.retrieve_data.to_csv('wwweee.csv', index=False)
        # todo 这个是将同个鸡群同一天的销售数据合并在一起
        # self.retrieve_data = self.retrieve_data.groupby(["retrieve_dt", "rearer_pop_dk",
        #                                       "forsale_breeds_dk", "retrieve_days"]).apply(
        # lambda x: np.average(x['avg_wt'], weights=x['retrieve_qty'])).reset_index(name='avg_wt')
        chicken_attributes = pd.read_csv("/".join([base_config.FEATURE_STORE_ROOT, 'chicken_attributes.csv']))
        self.chicken_attributes = chicken_attributes[
            ["rearer_pop_dk", "forsale_breeds_dk", "l3_breeds_class_nm", "feed_breeds_nm", "gender", "breeds_class_nm",
             "rearer_dk", "org_inv_dk", "tech_bk"]]

        org_data = pd.read_csv(base_config.RawData.DIM_ORG_INV_DATA_PATH.value)
        self.org_data = org_data[["org_inv_dk", "l4_org_inv_nm", "l3_org_inv_nm", "l3_org_inv_dk"]]


    def build_dataset(self, **param):
        def data_cleaning(df):
            df['timedelta'] = pd.to_timedelta(df['retrieve_days'] - 1, unit='D')
            df = df.set_index('timedelta')
            grouped_df = df.groupby(pd.Grouper(freq='1D')).agg({"avg_wt": ['mean', 'std']})
            upper_threshold = grouped_df[('avg_wt', 'mean')] + 3 * (grouped_df[('avg_wt', 'std')])
            lower_threshold = grouped_df[('avg_wt', 'mean')] - 2 * (grouped_df[('avg_wt', 'std')])
            df['upper_threshold'] = upper_threshold
            df['lower_threshold'] = lower_threshold

            df = df[(df['avg_wt'] <= df['upper_threshold']) & (df['avg_wt'] >= df['lower_threshold'])]

            return df

        all_item = self.retrieve_data[["rearer_pop_dk", "forsale_breeds_dk"]].drop_duplicates()
        new_day_wt = pd.DataFrame({
            "retrieve_days": [0.],
            "avg_wt": [0.]
        })
        new_data = all_item.merge(new_day_wt, how='cross')
        self.retrieve_data = pd.concat([self.retrieve_data, new_data])

        self.data = self.retrieve_data.merge(self.chicken_attributes, on=["rearer_pop_dk", "forsale_breeds_dk"],
                                             how='left')
        # todo 1修改隶属公司，按区域划分 西南  e5fb68e5-0109-1000-e002-0b5ec0a802deCCE7AED4 福建e5fb68e5-0109-1000-e002-07d2c0a802deCCE7AED4
        # 华南
        # self.org_data = self.org_data[(self.org_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e001-fbc6c0a802deCCE7AED4')
        #                               | (self.org_data[
        #                               'l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-0612c0a802deCCE7AED4')]

        # 华东
        # self.org_data = self.org_data[(self.org_data['l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-07d2c0a802deCCE7AED4')
        #                               | (self.org_data[
        #                                      'l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-02c1c0a802deCCE7AED4')
        #                               | (self.org_data[
        #                                      'l3_org_inv_dk'] == 'ZaA5bwEfEADgAaRYCgsFAcznrtQ=')
        #                               ]
        # 华中区域
        # self.org_data = self.org_data[(self.org_data['l3_org_inv_dk'] == 'y22ajAEcEADgGYceCgsFAsznrtQ=')
        #                             | (self.org_data[
        #                                      'l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-0b5ec0a802deCCE7AED4')
        #                             |(self.org_data[
        #                                      'l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-0b5ec0a802deCCE7AED4')
        #                                 ]

        # 西南区域
        # self.org_data = self.org_data[(self.org_data[
        #                                      'l3_org_inv_dk'] == 'e5fb68e5-0109-1000-e002-0b5ec0a802deCCE7AED4')
        #                               ]

        self.data = self.data.merge(self.org_data, on='org_inv_dk', how='left')
        self.data = self.data[self.raw_feature_names]
        self.data = self.data.dropna().reset_index(drop=True)



class WeightPredictionPredictIndexDataset(BaseDataSet):

    def __init__(self, **param):
        super().__init__(param)

        self.end_date = param.get('end_date')
        self.start_date = param.get('start_date')
        self.min_days, self.max_days = param.get('predict_days_interval', (50, 130))

        inventory_data = pd.read_csv(base_config.RawData.REARER_POP_SALE_FCST_DAY_DATA_PATH.value)
        inventory_data = inventory_data[(inventory_data['stat_dt'] >= self.start_date) &
                                        (inventory_data['stat_dt'] <= self.end_date)]
        inventory_data = inventory_data[['rearer_pop_dk', 'forsale_breeds_nm', 'forsale_breeds_dk']]
        self.inventory_data = inventory_data.drop_duplicates()

    def build_dataset(self, **param):
        self.data = pd.DataFrame({
            'retrieve_days': list(range(self.min_days, self.max_days + 1))
        })
        self.data = self.data.merge(self.inventory_data, how='cross')


class WeightPredictionPredictDataset(BaseDataSet):
    """
    Parameters
    ----------
    input_dataframe: 测试数据Dataframe
        running_date | outcome_date | rearer_pop_dk | forsale_breeds_dk | retrieve_days
    mask: 是否将rearer_pop_dk赋值为None

    Usage
    ------
    1. 根据"rearer_pop_dk"和"forsale_breeds_dk"从鸡群档案中索引出"rearer_dk", "tech_bk", "org_inv_dk"。
       如果"rearer_pop_dk"为None，或索引不到，则记为"unknown"。
    2. 根据"forsale_breeds_dk"在品种对照表中索引出"l3_breeds_class_nm", "feed_breeds_nm", "gender"
    3. 根据"forsale_breeds_dk"在销售品种档次对照表中索引出"breeds_class_nm"

    Return
    ------
    data/interim/weight_prediction.csv
        l3_breeds_class_nm | feed_breeds_nm | gender | breeds_class_nm | rearer_dk | org_inv_dk | tech_bk | month | retrieve_days

    Example
    -------
    >>> dataset = WeightPredictionDataset()
    >>> dataset.bulid_dataset()
    >>> dataset.dump_dataset("/".join([config.interim_dir, "weight_prediction.csv"])
    """

    def __init__(self, input_dataframe: pd.DataFrame, **param):
        super().__init__(param)
        self.input_dataframe = input_dataframe
        self.data = pd.DataFrame()
        self.chicken_file = pd.read_csv(base_config.RawData.REARER_POP_DATA_PATH.value)[
            ["rearer_pop_dk", "forsale_breeds_dk", "rearer_dk", "tech_bk", "org_inv_dk"]]
        self.breeds_data = pd.read_csv(base_config.RawData.FACT_YQ_FORSALE_BREEDS_RELATION_DATA_PATH.value)[
            ["forsale_breeds_dk", "l3_breeds_class_nm", "feed_breeds_nm", "gender"]]
        self.class_data = pd.read_csv(base_config.RawData.DIM_BREEDS_FILECLASS_DATA_PATH.value)[
            ["forsale_breeds_dk", "breeds_class_nm"]]
        self.org_data = pd.read_csv(base_config.RawData.DIM_ORG_INV_DATA_PATH.value)[
            ["org_inv_dk", "l4_org_inv_nm", "l3_org_inv_nm"]]

        self.chicken_file = self.chicken_file.drop_duplicates(subset=["rearer_pop_dk", "forsale_breeds_dk"])
        self.breeds_data = self.breeds_data.drop_duplicates(subset=["forsale_breeds_dk"])
        self.class_data = self.class_data.drop_duplicates(subset=["forsale_breeds_dk"])
        self.org_data = self.org_data.drop_duplicates(subset=["org_inv_dk"])

        self.mask = param.get("mask", False)
        self.raw_feature_names = param.get('raw_feature_names', [])

    def build_dataset(self, **param):
        self.data["rearer_pop_dk"] = self.input_dataframe["rearer_pop_dk"] if not self.mask else "unknown"
        self.data["forsale_breeds_dk"] = self.input_dataframe["forsale_breeds_dk"]
        self.data['retrieve_days'] = self.input_dataframe['retrieve_days']
        self.data['retrieve_dt'] = self.input_dataframe['outcome_date']
        # todo del
        # self.data['avg_wt'] = self.input_dataframe['avg_wt']

        self.data = self.data.merge(self.chicken_file, how='left', on=["rearer_pop_dk", "forsale_breeds_dk"])
        self.data = self.data.merge(self.breeds_data, how='left', on="forsale_breeds_dk")
        self.data = self.data.merge(self.class_data, how='left', on="forsale_breeds_dk")
        self.data = self.data.merge(self.org_data, how='left', on="org_inv_dk")

        self.data = self.data[self.raw_feature_names]

        self.data.fillna("unknown", inplace=True)
