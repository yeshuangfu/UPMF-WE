import os, sys, logging
import pandas as pd

from src.sqlalchemy import create_engine
from src.pyhive import hive

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class HiveDB(object):

    def __init__(self, key_file, database=None):
        # kerberos认证
        os.system('kinit -kt %s wens/dev@WENS.COM.CN' % key_file)
        if database is not None:
            self.s_conn = hive.Connection(host='mammut12.wens.com.cn', port='10500', auth='KERBEROS',
                                          kerberos_service_name='hive', database=database)
        else:
            self.s_conn = hive.Connection(host='mammut12.wens.com.cn', port='10500', auth='KERBEROS',
                                          kerberos_service_name='hive')

    def hive2csv(self, sql, path, index=False, drop_column_prefix=True):

        def _drop_column_prefix(col):
            if "." in col:
                prefix, new_col = col.split(".")
            else:
                new_col = col

            return new_col

        df = pd.read_sql_query(sql, self.s_conn)

        if drop_column_prefix:
            columns = [_drop_column_prefix(col) for col in df.columns]
            df.columns = columns

        df.to_csv(path, index=index)
        logger.info("load dataset count:%d" % len(df))

    def read_table2csv(self, table, path, index=False):
        df = pd.read_sql_table(table, self.s_conn)
        df.to_csv(path, index=index)


class MysqlDB(object):

    def __init__(self, key_file):
        # kerberos认证
        os.system('kinit -kt %s wens/dev@WENS.COM.CN' % key_file)
        self.mysql_conn = create_engine("mysql+pymysql://hydata:hydata@10.11.18.46/hydata")

    def dataframe2table(self, table: str, df: pd.DataFrame):
        logger.info("upload dataset count:%d" % len(df))
        df.to_sql(table, self.mysql_conn, if_exists="replace", index=None)
