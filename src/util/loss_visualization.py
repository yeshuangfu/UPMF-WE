import matplotlib.pyplot as plt
import os
import sys
import src.dataset.dataset_config as config

def train_log_visual(train_log_path):
    # print(train_log_path)
    with open(train_log_path, 'r') as f:
        lines = f.readlines()
    loss_list = []
    epoch_list = []
    for line in lines:
        if "Loss" in line:
            words = line.split()  # 按空格分开
            epoch = words[3]  # 找到epoch和loss字段值存入列表
            loss = words[11]
            epoch_list.append(epoch)
            loss_list.append(loss)

    loss_list = [float(value) for value in loss_list]  # 将列表中的值转换为浮点数类型
    epoch_list = sorted(set(epoch_list))  # 将epoch去重后排序
    epoch_list = [int(value) for value in epoch_list]  # epoch列表转为数值型
    # 每个epoch包含43个批次，分批次求每个epoch的平均loss
    step = 22  # 步长 2111%step_size + 1
    averages_loss = []  # 存储平均值的列表

    for i in range(0, len(loss_list), step):
        subset = loss_list[i: i + step]  # 提取当前步长范围内的子列表
        avg = sum(subset) / len(subset)  # 计算子列表的平均值
        avg = round(avg, 4)  # 保留4位小数
        averages_loss.append(avg)  # 将平均值添加到结果列表中

    # 创建图形窗口
    plt.figure()
    # 绘制折线图
    plt.plot(epoch_list, averages_loss, label='Loss')
    # 添加标题和坐标轴标签
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    # 显示图表
    plt.show()

if __name__ == '__main__':
    train_log_visual("/".join([config.log_dir, "deepfm/deepfm_1_1.0.0/2023-07-05_22-27-06.log"]))