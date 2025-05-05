import argparse
import os, sys, logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_squared_error as mse
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.dataset.weight_prediction_dataset1 import WeightPredictionDataset, WeightPredictionPredictDataset, \
    WeightPredictionPredictIndexDataset
from src.feature.features import Feature, FeatureType, FeatureDtype, Features
from src.feature.weight_prediction_transform import WeightPredictionTransformPipeline
from src.model.weight_prediction_nfm import WeightPredictionNfm
import src.config.weight_prediction_config as weight_prediction_config
import src.util.os_util as os_util
from src.util import serialize
from src.algorithm.base_algo import BaseAlgoMixin


base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WeightPredictionDataLoader(Dataset):
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        sample = self.data_df.iloc[idx, :-1].values
        label = self.data_df.iloc[idx, -1]

        return sample, label


class WeightPredictionPredictDataLoader(Dataset):
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        sample = self.data_df.iloc[idx, :].values

        return sample


class WeightPredictionNfmAlgo(BaseAlgoMixin):
    """
    Parameter
    ---------
    end_date: 边界日期。训练数据中只会包含end_date之前的数据。

    (Optional)training_interval: 训练日期数。只训练end_date之前若干天的数据。如果不给出则训练end_date之前所有数据。

    (Optional)retrieve_days_interval: 训练天龄范围，筛选出在该范围的天龄进行训练。默认为(50, 130)

    (Optional)avg_wt_interval: 训练均重范围，筛选出在该范围的均重进行训练。默认为(2, 8)

    (Optional)group: 是否将训练数据中同一天、同一鸡群、同一销售品种的训练数据聚合。默认为False。

    (Optional)random_seed: 随机数种子。用于划分训练集和测试集。

    (Optional)torch_seed: pytorch随机数种子，用于torch初始化参数或Dropout等操作。
    """

    def __init__(self, params: dict = {}):
        # super().__init__(param)
        self.dump_path = "./"
        self.file_type = "csv"
        self.origin_dataset = None
        self.model = None
        self.transform = None

        # 训练数据参数
        self.start_date = None
        self.end_date = None
        self.retrieve_days_interval = weight_prediction_config.TrainModuleConfig.retrieve_days_interval.value
        self.avg_wt_interval = weight_prediction_config.TrainModuleConfig.avg_wt_interval.value
        self.origin_dataset = None
        self.training_losses = []
        self.validation_losses = []
        # 模型保存路径
        self.model = None
        self.model_path = "/".join([weight_prediction_config.algo_model_dir, "WeightPredictionNfm.model.pth"])
        # self.new_model_path = "/".join([weight_prediction_config.algo_model_dir, "WeightPredictionNfm.new_model.pth"])
        # transform pipeline
        self.transform = None
        self.transform_path = "/".join([weight_prediction_config.algo_model_dir, "WeightPredictionNfmTransform.json"])

        # 训练过程参数
        self.num_iter = weight_prediction_config.TrainModuleConfig.num_iter.value
        self.batch_size = params[1]
        self.lr = params[2]
        self.momentum = params[3]
        self.l2_decay = params[4]
        self.BASE_EMB_DIM = params[8]
        self.old_param_weight = 0.4

        # 随机数种子
        seed = params[5]
        self.random_seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info("torch_seed={}".format(torch.initial_seed()))
        logger.info("random_seed={}".format(self.random_seed))

        # 原始特征与transform后特征名
        self.raw_feature_names = ["l3_breeds_class_nm", "feed_breeds_nm", "gender",
                                  "breeds_class_nm", "rearer_dk", "org_inv_dk", "l3_org_inv_nm", "l4_org_inv_nm",
                                  "tech_bk", "retrieve_days", "avg_wt"]
        self.transform_feature_names = ["l3_breeds_class_nm", "feed_breeds_nm", "gender",
                                        "breeds_class_nm", "rearer_dk", "org_inv_dk", "l3_org_inv_nm", "l4_org_inv_nm",
                                        "tech_bk", "retrieve_days"]

    def dump_data(self, data: pd.DataFrame, dataset_file):
        serialize.dataframe_dump(data, dataset_file, self.file_type)

    def load_data(self, dataset_file):
        return serialize.dataframe_read(dataset_file, self.file_type)

    def bulid_train_test_data(self, params: dict={}):
        self.origin_dataset = WeightPredictionDataset(start_date=self.start_date, end_date='2023-08-09',
                                                      retrieve_days_interval=params[6],
                                                      avg_wt_interval=params[7],
                                                      raw_feature_names=self.raw_feature_names)

        self.origin_dataset.build_dataset()

        train_test_data = self.origin_dataset.data.copy()
        self.dump_data(train_test_data.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_raw.csv"]))

        logger.info("Data between {} and {}: {}".format(self.start_date, self.end_date, train_test_data.shape))
        if train_test_data.isna().any().any():
            logger.info("!!!Warning: Nan in train_test_data")

        train_test_X = train_test_data.iloc[:, ~train_test_data.columns.isin(['avg_wt'])]
        train_test_y = train_test_data.loc[:, 'avg_wt']

        self.dump_data(train_test_X.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_X_raw.csv"]))
        self.dump_data(train_test_y.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_y_raw.csv"]))

    def split_train_test_data(self):
        train_test_X = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_X_raw.csv"]))
        train_test_y = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_y_raw.csv"]))

        if self.random_seed is None:
            train_X, test_X, train_y, test_y = train_test_split(train_test_X, train_test_y, test_size=0.2)
        else:
            train_X, test_X, train_y, test_y = train_test_split(train_test_X, train_test_y, test_size=0.2,
                                                                random_state=self.random_seed)

        self.dump_data(train_X.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_train_X_raw.csv"]))
        self.dump_data(test_X.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_test_X_raw.csv"]))
        self.dump_data(train_y.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_train_y.csv"]))
        self.dump_data(test_y.reset_index(drop=True),
                       "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_test_y.csv"]))

        logger.info("Training input: {}, Training label: {}".format(train_X.shape, train_y.shape))
        logger.info("Testing input: {}, Testing label: {}".format(test_X.shape, test_y.shape))
        if train_X.isna().any().any():
            logger.info("!!!Warning: Nan in train_X")
        if test_X.isna().any().any():
            logger.info("!!!Warning: Nan in test_X")
        if train_y.isna().any().any():
            logger.info("!!!Warning: Nan in train_y")
        if test_y.isna().any().any():
            logger.info("!!!Warning: Nan in train_y")

    def fit_transform(self):
        self.transform = WeightPredictionTransformPipeline(transform_feature_names=self.transform_feature_names)
        train_X = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_train_X_raw.csv"]))
        test_X = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_test_X_raw.csv"]))

        train_X = self.transform.fit_transform(input_dataset=train_X)
        test_X = self.transform.transform(input_dataset=test_X)
        with open(self.transform_path, "w+") as dump_file:
            dump_file.write(self.transform.to_json())
        logger.info("transformer transfrom_columns size: %d" % len(self.transform.features))
        logger.info("Saved transformer to {}".format(self.transform_path))

        self.dump_data(train_X.reset_index(drop=True), "/".join(
            [weight_prediction_config.algo_interim_dir, "weight_prediction_train_X_transformed.csv"]))
        self.dump_data(test_X.reset_index(drop=True), "/".join(
            [weight_prediction_config.algo_interim_dir, "weight_prediction_test_X_transformed.csv"]))

        logger.info("Transformed training input: {}".format(train_X.shape))
        logger.info("Transformed testing input: {}".format(test_X.shape))
        if train_X.isna().any().any():
            logger.info("!!!Warning: Nan in train_X")
        if test_X.isna().any().any():
            logger.info("!!!Warning: Nan in test_X")

    def load_transform(self):
        if self.transform is None:
            with open(self.transform_path, "r+") as dump_file:
                self.transform = WeightPredictionTransformPipeline.from_json(dump_file.read())
            logger.info("Loading transformer {}".format(self.transform_path))
        logger.info("transformer transfrom_columns size:%d" % len(self.transform.features))

    def build_model(self, load_dict: bool = False):
        self.load_transform()

        feature_dict = self.transform.features.features
        model_param = {}
        for feature in self.transform.features.get_features_keys():
            if feature_dict[feature].feature_type == FeatureType.Categorical or feature_dict[
                feature].feature_type == FeatureType.CategoricalSequence:
                param_name = "num_" + feature
                num_feature = feature_dict[feature].category_encode.size + 1
                model_param[param_name] = num_feature
                logger.info("{} = {}".format(param_name, num_feature))
        logger.info(model_param)
        model_param['BASE_EMB_DIM'] = self.BASE_EMB_DIM
        self.model = WeightPredictionNfm(model_param).to(device)

        if load_dict:
            logger.info('Loading model {}'.format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            for m in self.model.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(m.weight)

    def build_train_test_dataloader(self):
        train_X = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_train_X_transformed.csv"]))
        train_y = self.load_data("/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_train_y.csv"]))

        train_df = pd.concat([train_X, train_y], axis=1)
        train_df = train_df[self.raw_feature_names]
        train_dataset = WeightPredictionDataLoader(train_df)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

        test_X = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_test_X_transformed.csv"]))
        test_y = self.load_data("/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_test_y.csv"]))

        test_df = pd.concat([test_X, test_y], axis=1)
        test_df = test_df[self.raw_feature_names]
        test_dataset = WeightPredictionDataLoader(test_df)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def build_optim(self):
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.l2_decay)

    def train_model(self):
        logger = logging.getLogger(__name__)
        logger.info("Training with Pytorch Automatic Mixed Precision")
        batch_num = len(self.train_loader)
        patience = 5
        counter = 0
        best_loss = None
        
        # 确保在训练开始时清空损失列表
        self.training_losses = []
        self.validation_losses = []

        for epoch in range(self.num_iter):
            self.model.train()
            running_loss = 0.0
            total_batches = 0
            logger.info("Starting epoch [%d/%d]." % (epoch + 1, self.num_iter))

            for i, data in enumerate(self.train_loader, 0):
                input, label = data
                input = input.to(device)
                label = label.to(device)
                if input.shape[0] == 1:
                    continue

                self.optimizer.zero_grad()
                output = self.model(input)
                input_tensor = output.to(torch.float64)
                loss = self.loss_func(input_tensor, label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_batches += 1
                if (i + 1) % 2000 == 0:
                    logger.info('[%d/%d, %d/%d] training loss: %f' % (
                        epoch + 1, self.num_iter, i + 1, batch_num, running_loss / 2000))
                    running_loss = 0.0

            # 计算每个epoch的平均训练损失
            if total_batches > 0:
                epoch_train_loss = running_loss / total_batches
            else:
                epoch_train_loss = running_loss
            self.training_losses.append(epoch_train_loss)
            logger.info('[%d/%d] training loss: %f' % (epoch + 1, self.num_iter, epoch_train_loss))

            valid_loss = self.validation()
            self.validation_losses.append(valid_loss)
            logger.info('[%d/%d] valid loss: %f' % (epoch + 1, self.num_iter, valid_loss))

            # 每个epoch结束后，绘制训练损失曲线
            self.plot_train_loss()

            if best_loss is None or valid_loss < best_loss:
                counter = 0
                best_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info("Saved model to {}".format(self.model_path))
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
                    break

        # 训练结束后绘制完整的损失曲线
        self.plot_losses()

    def validation(self):
        mse_metric = np.zeros(2)
        self.model.eval()
        with torch.no_grad():
            for input, label in self.test_loader:
                input = input.to(device)
                label = label.to(device)
                output = self.model(input)
                loss = self.loss_func(output, label)
                mse_metric += (loss.item() * len(label), len(label))
        return mse_metric[0] / mse_metric[1]

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        print(f'self.training_losses:{self.training_losses}')
        print(f'self.validation_losses:{self.validation_losses}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        
        # 保存图表到文件，使用SVG格式
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/training_validation_loss.svg', format='svg')
        plt.close()
        
        # 单独绘制训练损失
        self.plot_train_loss()

    def plot_train_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, 'b-', label='Training Loss')
        print(f'self.training_losses:{self.training_losses}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        
        # 保存图表到文件，使用SVG格式
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/train_loss.svg', format='svg')
        plt.close()

    def plot_prediction_vs_actual(self, y_true, y_pred, dataset_type='train'):
        """
        绘制预测值与实际值的拟合图
        
        参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        dataset_type: 数据集类型，'train'或'valid'
        """
        # 过滤掉异常值，特别是接近(0,0)的点
        # 创建数据的副本以进行过滤
        y_true_filtered = np.array(y_true)
        y_pred_filtered = np.array(y_pred)
        
        # 找出所有异常点的索引 (例如：值太小的点)
        outlier_indices = []
        for i in range(len(y_true_filtered)):
            # 检测接近原点的异常值
            if y_true_filtered[i] < 0.5 or y_pred_filtered[i] < 0.5:
                outlier_indices.append(i)
        
        # 从数据中移除异常点
        if outlier_indices:
            y_true_filtered = np.delete(y_true_filtered, outlier_indices)
            y_pred_filtered = np.delete(y_pred_filtered, outlier_indices)
            print(f"已移除 {len(outlier_indices)} 个异常点")
        
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        plt.scatter(y_true_filtered, y_pred_filtered, alpha=0.5, label='Predictions')
        plt.rcParams.update({'font.size': 16})
        # 添加理想拟合线 (y=x)
        min_val = min(min(y_true_filtered), min(y_pred_filtered))
        max_val = max(max(y_true_filtered), max(y_pred_filtered))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
        
        # 添加实际拟合线
        z = np.polyfit(y_true_filtered, y_pred_filtered, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(y_true_filtered), p(np.sort(y_true_filtered)), 'g-', label='Actual Fit')
        
        # 计算MAE
        mae_value = mae(y_true_filtered, y_pred_filtered)
        
        # 计算RMSE
        rmse_value = np.sqrt(mse(y_true_filtered, y_pred_filtered))
        
        # 添加文本框只显示MAE和RMSE指标
        plt.annotate(f'MAE = {mae_value:.4f}\nRMSE = {rmse_value:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        title_prefix = 'Training' if dataset_type == 'train' else 'Validation'
        plt.title(f'{title_prefix} Set: Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.legend()
        
        # 保存图表，使用SVG格式
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{dataset_type}_prediction_vs_actual.svg', format='svg')
        plt.close()

    def train_indicator(self):
        self.build_model(load_dict=True)

        labels = np.array([])
        outputs = np.array([])
        with torch.no_grad():
            for input, label in self.train_loader:
                input = input.to(device)
                label = label.to(device)
                output = self.model(input)

                labels = np.append(labels, label.cpu().numpy())
                outputs = np.append(outputs, output.cpu().numpy())
            logger.info("--------weight_prediction_algo train_summary: train_mae: {}, train_mse: {}, train_mape: {}".format(mae(labels, outputs), mse(labels, outputs), mape(labels, outputs)))
        
        # 绘制训练集预测值与实际值拟合图
        self.plot_prediction_vs_actual(labels, outputs, 'train')

        labels = np.array([])
        outputs = np.array([])
        with torch.no_grad():
            for input, label in self.test_loader:
                input = input.to(device)
                label = label.to(device)
                output = self.model(input)

                labels = np.append(labels, label.cpu().numpy())
                outputs = np.append(outputs, output.cpu().numpy())
            logger.info("--------weight_prediction_algo valid_summary: valid_mae: {}, valid_mse: {}, valid_mape: {}".format(mae(labels, outputs), mse(labels, outputs), mape(labels, outputs)))
        
        # 绘制验证集预测值与实际值拟合图
        self.plot_prediction_vs_actual(labels, outputs, 'valid')

    def train(self,
              train_algo_outcome_dt_end: str, outcome_offset: Optional[int],
              outcome_window_len: Optional[int], train_algo_interval: int,
              valid_algo_running_dt_end: Optional[str], valid_algo_interval: Optional[int],
              params_item: dict = {}):

        self.end_date = train_algo_outcome_dt_end
        self.start_date = (
                    datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=train_algo_interval)).strftime(
            "%Y-%m-%d")
        logger.info("Start date: %s" % self.start_date)
        logger.info("End date: %s" % self.end_date)

        # self.bulid_train_test_data(params_item)
        # self.split_train_test_data()
        self.fit_transform()

        self.build_model(load_dict=False)
        self.build_train_test_dataloader()
        self.build_optim()
        self.train_model()
        self.train_indicator()

    def build_predict_index_dataset(self, predict_running_dt_end: str, predict_running_dt_interval: int):
        start_date = (datetime.strptime(predict_running_dt_end, "%Y-%m-%d") - timedelta(
            days=predict_running_dt_interval-1)).strftime("%Y-%m-%d")
        index_dataset = WeightPredictionPredictIndexDataset(end_date=predict_running_dt_end, start_date=start_date,
                                                            predict_days_interval=weight_prediction_config.PredictModuleConfig.predict_days_interval.value)
        index_dataset.build_dataset()
        logger.info("Predict Index Data Size: {}".format(index_dataset.data.shape))
        return index_dataset.data

    def build_predict_dataset(self, input_dataframe: pd.DataFrame, mask: bool = False):
        dataset = WeightPredictionPredictDataset(input_dataframe=input_dataframe, mask=mask,
                                                 raw_feature_names=self.raw_feature_names[:-1])
        dataset.build_dataset()
        dataset.dump_dataset('/'.join([weight_prediction_config.algo_interim_dir, 'weight_prediction_predict_X.csv']))

        predict_transformed_X = self.transform.transform(input_dataset=dataset.data)
        self.dump_data(predict_transformed_X, '/'.join(
            [weight_prediction_config.algo_interim_dir, 'weight_prediction_predict_X_transformed.csv']))

    def build_predict_dataloader(self):
        predict_X = self.load_data(
            "/".join([weight_prediction_config.algo_interim_dir, "weight_prediction_predict_X_transformed.csv"]))
        predict_X = predict_X[self.transform_feature_names]
        predict_dataset = WeightPredictionPredictDataLoader(predict_X)
        self.predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

    def predict(self, input_dataframe: pd.DataFrame):
        avg_wt = np.array([])
        self.model.eval()
        with torch.no_grad():
            for input in self.predict_loader:
                input = input.to(device)
                output = self.model(input)
                avg_wt = np.append(avg_wt, output.cpu().numpy())
        output_dataframe = input_dataframe.copy()
        output_dataframe['pred_avg_wt'] = avg_wt
        return output_dataframe

    def batch_predict_with_dt(self, predict_running_dt_end: str, predict_running_dt_interval: int,
                              predict_outcome_offset: int, mask: bool=False):
        index_sample = self.build_predict_index_dataset(predict_running_dt_end, predict_running_dt_interval)
        logger.info("Mask rearer_pop_dk: {}".format(mask))
        self.load_transform()
        self.build_predict_dataset(input_dataframe=index_sample, mask=mask)
        self.build_predict_dataloader()
        self.build_model(load_dict=True)
        result = self.predict(input_dataframe=index_sample)

        decrease_item = self.test_increase(result)
        if len(decrease_item) > 0:
            logger.info("Error!! Found not increase item!")
            for rearer_pop_dk, forsale_breeds_dk in decrease_item:
                logger.info("rearer_pop_dk={}, forsale_breeds_dk={}".format(rearer_pop_dk, forsale_breeds_dk))
        # 删除
        # result.to_csv("/".join([weight_prediction_config.algo_interim_dir, "WeightPredictionNfm.model.predict_dt.csv"]), index=False)
        return result

    def batch_predict_with_index_sample(self, index_sample: pd.DataFrame, mask: bool=False):
        logger.info("Mask rearer_pop_dk: {}".format(mask))
        self.load_transform()
        self.build_predict_dataset(input_dataframe=index_sample, mask=mask)
        self.build_predict_dataloader()
        self.build_model(load_dict=True)
        result = self.predict(input_dataframe=index_sample)
        return result

    def test_increase(self, df: pd.DataFrame):
        decrease_item = []
        grouped_df = df.groupby(['rearer_pop_dk', 'forsale_breeds_dk'])
        for (rearer_pop_dk, forsale_breeds_dk), group in grouped_df:
            pred_avg_wt = group['pred_avg_wt'].tolist()
            increase = all([wt1<wt2 for wt1, wt2 in zip(pred_avg_wt[:-1], pred_avg_wt[1:])])
            if not increase:
                decrease_item.append((rearer_pop_dk, forsale_breeds_dk))
        return decrease_item

if __name__ == "__main__":
    pass
