import pandas as pd



def process_data(data: pd.DataFrame):
    df = pd.read_csv('single_gender_chicken.csv')
    df['timedelta'] = pd.to_timedelta(df['retrieve_days'] - 1, unit='D')
    df = df.set_index('timedelta')
    grouped_df = df.groupby(pd.Grouper(freq='1D')).agg({"avg_wt": ['mean', 'std']})
    upper_threshold = grouped_df[('avg_wt', 'mean')] + 3 * (grouped_df[('avg_wt', 'std')])
    lower_threshold = grouped_df[('avg_wt', 'mean')] - 2 * (grouped_df[('avg_wt', 'std')])
    df['upper_threshold'] = upper_threshold
    df['lower_threshold'] = lower_threshold

    df = df[(df['avg_wt'] <= df['upper_threshold']) & (df['avg_wt'] >= df['lower_threshold'])]

# 筛选出指定鸡群
def select_chicken(df: pd.DataFrame):
    df = df[df['rearer_pop_dk'] == '++rfMae/RE2W59qboKLmo8B4fig=']
    return df

if __name__ == '__main__':
    # 筛选指定鸡群 ++rfMae/RE2W59qboKLmo8B4fig=
    df = pd.read_csv('../../data/raw/ADS_AI_MRT_REARER_LANDP_RETRIEVE.csv')
    singel_rearer = select_chicken(df)
    # 筛选指定
