import json
import pandas as pd


# def dump_dict_as_txt(d, out_path):
#     with open(out_path, "w+") as f:
#         for k, v in d.items():
#             # print >> f, k, v


def dump_dict_as_jsom(d, out_path):
    f = open(out_path, 'w+')
    f.write(json.dumps(d))
    f.close()


def load_josn_as_dict(input):
    d = {}
    with open(input, 'r') as f:
        d = json.load(fp=f)
    return d


def dataframe_dump(df: pd.DataFrame, path: str, file_type: str, index=False):
    if file_type == 'csv':
        df.to_csv(path, index=index)
    elif file_type == 'parquet':
        df.to_parquet(path, index=index)
    elif file_type == 'pickle':
        df.to_pickle(path)


def dataframe_read(path: str, file_type='csv', index_col=0):
    if file_type == 'csv':
        return pd.read_csv(path, index_col=None)
    elif file_type == 'parquet':
        return pd.read_parquet(path)
    elif file_type == 'pickle':
        return pd.read_pickle(path)


def dump_str_to_file(line, file_path):
    with open(file_path, 'w+') as f:
        f.write(line)


def read_file(file_path):
    with open(file_path, 'r+') as f:
        return f.read()
