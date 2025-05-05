import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config.weight_prediction_config as weight_prediction_config


"""
鸡群均重预测模块输入特征：
- 品类l3_breeds_class_nm: Categorical类。
- 饲养品种feed_breeds_nm: Categorical类。
- 性别gender: Categorical类。
- 档次breeds_class_nm: Categorical类。
- 养户rearer_dk: Categorical类。
- 服务部org_inv_dk: Categorical类。
- 技术员tech_bk: Categorical类。
- 天龄age: Continuous类。
"""

EMB_DIM = weight_prediction_config.EMB_DIM
BASE_EMB_DIM = weight_prediction_config.BASE_EMB_DIM

def get_activate_func(activate_func='relu', param: dict={}):
    if activate_func == 'relu':
        return nn.ReLU()
    elif activate_func == 'leaky_relu':
        negative_slope = param.get('negative_slope', 1e-2)
        return nn.LeakyReLU(negative_slope=negative_slope)
    else:
        raise ValueError("Unknown activate_func")

def make_mlp_layers(mlp_input_dim, hidden_dims, mlp_output_dim, activate_func='relu', **param):
    mlp_layers = nn.Sequential()
    dropout_rate = param.get('dropout_rate', None)
    if len(hidden_dims) == 0:
        mlp_layers.add_module("output", nn.Linear(mlp_input_dim, mlp_output_dim))
    else:
        mlp_layers.add_module("input", nn.Linear(mlp_input_dim, hidden_dims[0]))
        mlp_layers.add_module("activate0", get_activate_func(activate_func, param))
        if dropout_rate is not None:
            mlp_layers.add_module("dropout0", nn.Dropout(p=dropout_rate))

        for i, (input_dim, output_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            mlp_layers.add_module("linear{}".format(i+1), nn.Linear(input_dim, output_dim))
            mlp_layers.add_module("activate{}".format(i+1), get_activate_func(activate_func, param))
            if dropout_rate is not None:
                mlp_layers.add_module("dropout{}".format(i+1), nn.Dropout(p=dropout_rate))

        mlp_layers.add_module("output", nn.Linear(hidden_dims[-1], mlp_output_dim))
    return mlp_layers

class FeatureInteraction(nn.Module):

    def __init__(self, **params):
        super().__init__()

        self.emb_size = params.get('BASE_EMB_DIM')
        self.bn = nn.BatchNorm1d(self.emb_size)

    def forward(self, emb_stk):
        # emb_stk = [field_num, batch_size, emb_size]
        emb_stk = emb_stk.permute((1, 0, 2))
        # emb_stk = [batch_size, field_num, emb_size]
        # todo 二阶交互
        emb_cross = 1 / 2 * (
                torch.pow(torch.sum(emb_stk, dim=1), 2) - torch.sum(torch.pow(emb_stk, 2), dim=1)
        )
        # emb_cross = [batch_size, emb_size]
        return self.bn(emb_cross)

class WeightPredictionNfm(nn.Module):
    """
    Parameter
    ---------
    - num_l3_breeds_class_nm
    - num_feed_breeds_nm
    - num_gender
    - num_breeds_class_nm
    - num_rearer_dk
    - num_org_inv_dk
    - num_l3_org_inv_nm
    - num_l4_org_inv_nm
    - num_tech_bk
    """
    def __init__(self, params: dict={}):
        super().__init__()
        num_l3_breeds_class_nm = params.get('num_l3_breeds_class_nm', 1)
        num_feed_breeds_nm = params.get('num_feed_breeds_nm', 1)
        num_gender = params.get('num_gender', 1)
        num_breeds_class_nm = params.get('num_breeds_class_nm', 1)
        num_rearer_dk = params.get('num_rearer_dk', 1)
        num_org_inv_dk = params.get('num_org_inv_dk', 1)
        num_l3_org_inv_nm = params.get('num_l3_org_inv_nm', 1)
        num_l4_org_inv_nm = params.get('num_l4_org_inv_nm', 1)
        num_tech_bk = params.get('num_tech_bk', 1)

        # 离散特征嵌入
        self.l3_breeds_class_nm_emb = nn.Embedding(num_l3_breeds_class_nm, params.get('BASE_EMB_DIM'))
        self.feed_breeds_nm_emb = nn.Embedding(num_feed_breeds_nm, params.get('BASE_EMB_DIM'))
        self.gender_emb = nn.Embedding(num_gender, params.get('BASE_EMB_DIM'))
        self.breeds_class_nm_emb = nn.Embedding(num_breeds_class_nm, params.get('BASE_EMB_DIM'))
        self.rearer_dk_emb = nn.Embedding(num_rearer_dk, EMB_DIM["rearer_dk"])
        self.org_inv_dk_emb = nn.Embedding(num_org_inv_dk, params.get('BASE_EMB_DIM'))
        self.l3_org_inv_nm_emb = nn.Embedding(num_l3_org_inv_nm, params.get('BASE_EMB_DIM'))
        self.l4_org_inv_nm_emb = nn.Embedding(num_l4_org_inv_nm, params.get('BASE_EMB_DIM'))
        self.tech_bk_emb = nn.Embedding(num_tech_bk, EMB_DIM["tech_bk"])

        self.rearer_dk_fc = nn.Linear(EMB_DIM["rearer_dk"], params.get('BASE_EMB_DIM'))
        self.tech_bk_fc = nn.Linear(EMB_DIM["tech_bk"], params.get('BASE_EMB_DIM'))

        mlp_input_dim = params.get('BASE_EMB_DIM')
        mlp_output_dim = 3
        
        # MLP层，输出三个参数a,b,c
        self.mlp = make_mlp_layers(mlp_input_dim=mlp_input_dim,
                                   hidden_dims=[512, 128, 64, 32, 8],
                                   mlp_output_dim=mlp_output_dim)

        self.fea_interact = FeatureInteraction(BASE_EMB_DIM=params.get('BASE_EMB_DIM'))

    def forward(self, input):
        l3_breeds_class_nm = input[:, 0].to(torch.long)
        feed_breeds_nm = input[:, 1].to(torch.long)
        gender = input[:, 2].to(torch.long)
        breeds_class_nm = input[:, 3].to(torch.long)
        rearer_dk = input[:, 4].to(torch.long)
        org_inv_dk = input[:, 5].to(torch.long)
        l3_org_inv_nm = input[:, 6].to(torch.long)
        l4_org_inv_nm = input[:, 7].to(torch.long)
        tech_bk = input[:, 8].to(torch.long)
        age = input[:, 9]

        emb_list = []
        emb_list.append(self.l3_breeds_class_nm_emb(l3_breeds_class_nm))
        emb_list.append(self.feed_breeds_nm_emb(feed_breeds_nm))
        emb_list.append(self.gender_emb(gender))
        emb_list.append(self.breeds_class_nm_emb(breeds_class_nm))
        emb_list.append(self.rearer_dk_fc(self.rearer_dk_emb(rearer_dk)))
        emb_list.append(self.org_inv_dk_emb(org_inv_dk))
        emb_list.append(self.l3_org_inv_nm_emb(l3_org_inv_nm))
        emb_list.append(self.l4_org_inv_nm_emb(l4_org_inv_nm))
        emb_list.append(self.tech_bk_fc(self.tech_bk_emb(tech_bk)))

        # 原始结构
        emb_stk = torch.stack(emb_list)
        x = self.fea_interact(emb_stk)
        x = self.mlp(x)
        a = nn.Softplus()(x[:, 0])
        b = nn.Softplus()(x[:, 1])
        c = x[:, 2]
        y = b * nn.Sigmoid()(a * age + c)

        # 1.去掉softPlus
        # emb_stk = torch.stack(emb_list)
        # x = self.fea_interact(emb_stk)
        # x = self.mlp(x)
        # a = x[:, 0]
        # b = x[:, 1]
        # c = x[:, 2]
        # y = b * nn.Sigmoid()(a * age + c)

        # 2.去掉二阶交互
        # emb_stk = torch.stack(emb_list)
        # emb_stk = F.avg_pool1d(emb_stk.permute(1, 2, 0), kernel_size=9).squeeze()
        # x = self.mlp(emb_stk)
        # a = nn.Softplus()(x[:, 0])
        # b = nn.Softplus()(x[:, 1])
        # c = x[:, 2]
        # y = b * nn.Sigmoid()(a * age + c)

        # 3.去掉生长曲线 还得修改MLP输出的元素个数 112行
        # emb_stk = torch.stack(emb_list)
        # x = self.fea_interact(emb_stk)
        # x = self.mlp(x)
        # a = nn.Softplus()(x[:, 0])
        # y = a

        # 4.去掉SoftPlus
        # emb_stk = torch.stack(emb_list)
        # x = self.fea_interact(emb_stk)
        # x = self.mlp(x)
        # a = nn.Softplus()(x[:, 0])
        # y = a



        if y.isnan().any():
            print(input)
            print(x)
            print(y)
            assert False
        # if a.min() <= 0:
        #     print("Warning: a must be positive!")
        # if b.min() <= 0:
        #     print("Warning: b must be positive!")
        
        return y
