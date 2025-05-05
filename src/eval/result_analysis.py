import pandas as pd
import src.config.base_config as config
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

version = 'v2.2.3'
end_date = "2021-06-15"
# end_date = "2021-09-20"
# end_date = "2021-12-15"

def base_metrics(data: pd.DataFrame):
    gt_label = data['order_prob'].to_numpy()
    pred_prob = data['pred_order_prob'].to_numpy()
    pred_label = data['SHIPPING_DATA_N'].to_numpy()

    conf_matrix = confusion_matrix(gt_label, pred_label)
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    # print("TN: {}, FN: {}, TP: {}, FP: {}".format(TN, FN, TP, FP))
    positive_precision = precision_score(gt_label, pred_label, pos_label=1)
    positive_recall = recall_score(gt_label, pred_label, pos_label=1)
    positive_f1 = f1_score(gt_label, pred_label, pos_label=1)
    # print("Positive precision: {}".format(positive_precision))
    # print("Positive recall: {}".format(positive_recall))
    # print("Positive f1: {}".format(positive_f1))
    negative_precision = precision_score(gt_label, pred_label, pos_label=0)
    negative_recall = recall_score(gt_label, pred_label, pos_label=0)
    negative_f1 = f1_score(gt_label, pred_label, pos_label=0)
    # print("Negative precision: {}".format(negative_precision))
    # print("Negative recall: {}".format(negative_recall))
    # print("Negative f1: {}".format(negative_f1))

    auc = roc_auc_score(gt_label, pred_prob)
    # print("AUC: {}".format(auc))

def remove_cold_start_cust(data: pd.DataFrame, sale_data: pd.DataFrame):
    all_cust = set(sale_data['cust_dk'].unique())
    data.loc[:, 'is_cold_start'] = data.apply(lambda row: 0 if row['cust_dk'] in all_cust else 1, axis=1)
    cold_start_user_num = (data['is_cold_start']==1).sum()
    not_cold_start_user_num = (data['is_cold_start']==0).sum()
    # print("Cold start: {}, {}%".format(cold_start_user_num, cold_start_user_num/len(data)*100))
    # print("Not cold start: {}, {}%".format(not_cold_start_user_num, not_cold_start_user_num/len(data)*100))
    return data[data['is_cold_start']==0]

def new_l4(data: pd.DataFrame, sale_data: pd.DataFrame):
    l4_data = sale_data.groupby('cust_dk')['l4_org_inv_nm'].apply(set).reset_index()
    l4_data.rename({'l4_org_inv_nm': 'purchased_l4_org_inv_nm'}, axis=1, inplace=True)
    data = data.merge(l4_data, how='left', on='cust_dk')
    data['is_new_l4'] = data.apply(lambda row: 0 if row['l4_org_inv_nm'] in row['purchased_l4_org_inv_nm'] else 1, axis=1)

    new_l4_num = (data['is_new_l4']==1).sum()
    # print("New l4 org: {}, {}%".format(new_l4_num, new_l4_num/len(data)*100))

    not_new_l4 = data[data['is_new_l4']==0]

    purchase_count = sale_data.groupby(['cust_dk', 'l4_org_inv_nm']).size().reset_index()
    purchase_count.columns = ['cust_dk', 'l4_org_inv_nm', 'l4_purchase_count']
    not_new_l4 = not_new_l4.merge(purchase_count, on=['cust_dk', 'l4_org_inv_nm'], how='left')

    not_new_l4['l4_purchase_count'] = not_new_l4['l4_purchase_count'].astype(int)
    l4_purchase_count = not_new_l4['l4_purchase_count'].value_counts()
    # print(len(l4_purchase_count))
    x = []
    y = []
    for k, v in l4_purchase_count.items():
        x.append(str(k))
        y.append(v)
    plt.cla()
    plt.bar(x, y)
    plt.xlabel('l4_purchase_count')
    plt.ylabel('Count')
    plt.title('l4_purchase_count Histogram')
    plt.savefig("./l4_purchase_count.png")

    return data

def new_l3(data: pd.DataFrame, sale_data: pd.DataFrame):
    l3_data = sale_data.groupby('cust_dk')['l3_org_inv_nm'].apply(set).reset_index()
    l3_data.rename({'l3_org_inv_nm': 'purchased_l3_org_inv_nm'}, axis=1, inplace=True)
    data = data.merge(l3_data, how='left', on='cust_dk')
    data['is_new_l3'] = data.apply(lambda row: 0 if row['l3_org_inv_nm'] in row['purchased_l3_org_inv_nm'] else 1, axis=1)

    new_l3_num = (data['is_new_l3']==1).sum()
    # print("New l3 org: {}, {}%".format(new_l3_num, new_l3_num/len(data)*100))


    not_new_l3 = data[data['is_new_l3']==0]

    purchase_count = sale_data.groupby(['cust_dk', 'l3_org_inv_nm']).size().reset_index()
    purchase_count.columns = ['cust_dk', 'l3_org_inv_nm', 'l3_purchase_count']
    not_new_l3 = not_new_l3.merge(purchase_count, on=['cust_dk', 'l3_org_inv_nm'], how='left')

    not_new_l3['l3_purchase_count'] = not_new_l3['l3_purchase_count'].astype(int)
    l3_purchase_count = not_new_l3['l3_purchase_count'].value_counts()
    # print(len(l3_purchase_count))
    x = []
    y = []
    for k, v in l3_purchase_count.items():
        x.append(str(k))
        y.append(v)
    plt.cla()
    plt.bar(x, y)
    plt.xlabel('l3_purchase_count')
    plt.ylabel('Count')
    plt.title('l3_purchase_count Histogram')
    plt.savefig("./l3_purchase_count.png")

    return data

def purchase_interval(data: pd.DataFrame, interval_data: pd.DataFrame):
    data = data.merge(interval_data, how='left', on=['edge_date', 'cust_dk'])
    data.fillna(0, inplace=True)
    data['interval_from_last_purchase'] = data['interval_from_last_purchase'].astype(int)
    interval_from_last_purchase = data['interval_from_last_purchase'].value_counts()
    # print(len(interval_from_last_purchase))
    x = []
    y = []
    for k, v in interval_from_last_purchase.items():
        if k > 20: 
            continue
        x.append(str(k))
        y.append(v)
    plt.cla()
    # plt.hist(data['interval_from_last_purchase'], bins=len(interval_from_last_purchase))
    plt.bar(x, y)
    plt.xlabel('interval_from_last_purchase')
    plt.ylabel('Count')
    plt.title('interval_from_last_purchase Histogram')
    plt.savefig("./interval_from_last_purchase.png")

    return data

def attendance(data: pd.DataFrame, attendence_data: pd.DataFrame):
    data = data.merge(attendence_data, how='left', on=['edge_date', 'cust_dk'])
    data.fillna(0, inplace=True)

    data['avg_interval_last_7_days'] = data['avg_interval_last_7_days'].astype(int)
    avg_interval_last_7_days = data['avg_interval_last_7_days'].value_counts()
    # print(len(avg_interval_last_7_days))
    x = []
    y = []
    for k, v in avg_interval_last_7_days.items():
        x.append(str(k))
        y.append(v)
    plt.cla()
    # plt.hist(data['avg_interval_last_7_days'], bins=len(avg_interval_last_7_days))
    plt.bar(x, y)
    plt.xlabel('avg_interval_last_7_days')
    plt.ylabel('Count')
    plt.title('avg_interval_last_7_days Histogram')
    plt.savefig("./avg_interval_last_7_days.png")

    return data

def l4_analysis(data: pd.DataFrame, sale_data: pd.DataFrame):
    all_l4 = set(data['l4_org_inv_nm'].unique())
    fn_sale_data = sale_data[sale_data['l4_org_inv_nm'].isin(all_l4)]
    result = fn_sale_data['l4_org_inv_nm'].value_counts().to_numpy()
    hist, bins = np.histogram(result, bins=range(0, 1200, 100))
    plt.bar(bins[:-1], hist, width=70)
    plt.xlabel('purchase count')
    plt.ylabel('num l4 org')
    plt.savefig("./l4_analysis.png")

def false_negative_analysis(data: pd.DataFrame):
    global end_date
    false_negative = data[(data['order_prob']==1.) & (data['decision']==0.)]
    fn_num = false_negative.shape[0]
    # print("False negative: {}".format(fn_num))
    
    end_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=float(100))).strftime("%Y-%m-%d")
    
    cust_label_data_filename = 'DIM_CUST.csv'
    cust_label_data = pd.read_csv("/".join([config.RAW_DATA_ROOT, cust_label_data_filename]))

    sale_data_filename = "ADS_AI_MRT_SALES_SHIPPING.csv"
    sale_data = pd.read_csv("/".join([config.RAW_DATA_ROOT, sale_data_filename]))
    sale_data.rename({"shipping_dt": "date"}, axis=1, inplace=True)
    sale_data = sale_data[(sale_data['guide_price'] > 0) & 
                         (sale_data['sale_price'] > 0) & 
                         (sale_data['sale_qty'] > 0) &
                         (sale_data["date"]>=start_date) &
                         (sale_data["date"]<=end_date)].reset_index(drop=True)
    merge_data = pd.merge(left=sale_data, right=cust_label_data, on='cust_dk', how='left').reset_index(drop=True)
    sale_data = sale_data[merge_data['inner_company_ind'] == '外部']
    sale_data = sale_data[["org_inv_dk", "cust_dk"]]
    sale_data = sale_data.drop_duplicates()

    org_data_filename = "DIM_ORG_INV.csv"
    org_data = pd.read_csv("/".join([config.RAW_DATA_ROOT, org_data_filename]))[["org_inv_dk", "l4_org_inv_nm", "l3_org_inv_nm"]]
    org_data = org_data.drop_duplicates()

    sale_data = sale_data.merge(right=org_data, how='left', on='org_inv_dk')
    org_data = org_data[["l4_org_inv_nm", "l3_org_inv_nm"]].drop_duplicates()
    false_negative = false_negative.merge(right=org_data, how='left', on='l4_org_inv_nm')

    # false_negative = remove_cold_start_cust(false_negative, sale_data)

    # false_negative = new_l4(false_negative, sale_data)
    # false_negative = new_l3(false_negative, sale_data)

    # interval_data = pd.read_csv("/".join([config.INTERIM_DATA_ROOT, "customer_purchase_interval.csv"]))
    # interval_data.rename({'date': 'edge_date'}, axis=1, inplace=True)
    # interval_data['edge_date'] = pd.to_datetime(interval_data['edge_date'])
    # attendance_data = pd.read_csv("/".join([config.INTERIM_DATA_ROOT, "customer_attendance.csv"]))
    # attendance_data.rename({'date': 'edge_date'}, axis=1, inplace=True)
    # attendance_data['edge_date'] = pd.to_datetime(attendance_data['edge_date'])

    # false_negative = purchase_interval(false_negative, interval_data)
    # false_negative = attendance(false_negative, attendance_data)

    false_negative = l4_analysis(data, sale_data)

    


if __name__ == '__main__':
    ground_truth = pd.read_csv("/".join([config.algo_external_dir, "pred_order_prob-{}-ground_truth_data.csv".format(end_date)]))
    pred_data = pd.read_csv("/".join([config.algo_external_dir, "{}-pred_order_prob-{}-pred_data.csv".format(version, end_date)]))
    # print(ground_truth.shape, pred_data.shape)

    data = ground_truth.merge(pred_data, how='left')
    data['running_date'] = pd.to_datetime(data['running_date'])
    data['outcome_date'] = pd.to_datetime(data['outcome_date'])
    data['edge_date'] = data['running_date'] - timedelta(days=1)
    # base_metrics(data)
    false_negative_analysis(data)
