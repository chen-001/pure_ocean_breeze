__updated__ = "2023-08-31 19:57:21"

import pandas as pd
from loguru import logger
import datetime
import psycopg2 as pg
import numpy as np
import requests
import os
from typing import Union, Dict, List
from psycopg2.extensions import register_adapter, AsIs
from tenacity import retry, stop_after_attempt, wait_fixed
import questdb.ingress as qdbing
from pure_ocean_breeze.jason.state.homeplace import HomePlace


class MetaSQLDriver(object):
    """所有sql类数据库通用的一些功能"""

    def __init__(
        self, user: str, password: str, host: str, port: str, database: str
    ) -> None:
        """数据库的基本信息

        Parameters
        ----------
        user : str
            用户名
        password : str
            密码
        host : str
            地址
        port : str
            端口
        database : str
            数据库名
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def connect(self):
        ...

    def do_order(self, sql_order: str) -> any:
        """执行任意一句sql语句

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        any
            返回结果
        """
        conn = self.connect()
        cur = conn.cursor()
        return cur.execute(sql_order)

    def add_new_database(self, db_name: str) -> None:
        """添加一个新数据库

        Parameters
        ----------
        db_name : str
            新数据库的名称
        """
        try:
            self.do_order(f"CREATE DATABASE {db_name}")
            logger.success(f"已添加名为{db_name}的数据库")
        except Exception:
            logger.warning(f"已经存在名为{db_name}的数据库，请检查")

    @retry(stop=stop_after_attempt(10))
    def get_data(
        self, sql_order: str, only_array: bool = 0
    ) -> Union[pd.DataFrame, np.ndarray]:
        """以sql命令的方式，从数据库中读取数据

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            读取的结果
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(sql_order)
        df_data = cursor.fetchall()
        if not only_array:
            columns = [i[0] for i in cursor.description]
            df = pd.DataFrame(df_data, columns=columns)
            return df
        else:
            return np.array(df_data)

    def get_data_show_time(self, sql_order: str) -> pd.DataFrame:
        """以sql命令的方式，从数据库中读取数据，并告知所用时间

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        pd.DataFrame
            读取的结果
        """
        a = datetime.datetime.now()
        df = self.get_data(sql_order)
        b = datetime.datetime.now()
        c = b - a
        l = c.seconds + c.microseconds / 1e6
        l = round(l, 2)
        print(f"共用时{l}秒")
        return df

    def get_data_alter(self, sql_order: str) -> pd.DataFrame:
        """专门用于应对get_data函数可能出现的特殊情况，例如宽表

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        pd.DataFrame
            读取的结果
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(sql_order)
        df_data = cursor.fetchall()
        df = pd.DataFrame(df_data)
        df.columns = list(df.iloc[0, :])
        df = df.iloc[1:, :]
        df.index = list(df.iloc[:, 0])
        df = df.iloc[:, 1:]
        return df

    def get_data_old(self, sql_order: str) -> pd.DataFrame:
        """以pandas.read_sql读取数据

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        pd.DataFrame
            读取的结果
        """
        a = pd.read_sql(sql_order, con=self.engine)
        return a

    def get_data_old_show_time(self, sql_order: str) -> pd.DataFrame:
        """以pd.read_sql和sql命令的方式，从数据库中读取数据，并告知所用时间

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        pd.DataFrame
            读取的结果
        """
        a = datetime.datetime.now()
        df = self.get_data_old(sql_order)
        b = datetime.datetime.now()
        c = b - a
        l = c.seconds + c.microseconds / 1e6
        l = round(l, 2)
        print(f"共用时{l}秒")
        return df

    def show_all_codes(self, table_name: str) -> list:
        """返回表（常用于分钟数据）中所有股票的代码

        Parameters
        ----------
        table_name : str
            表名

        Returns
        -------
        list
            全部股票代码
        """
        df = self.get_data(f"select distinct(code) from {table_name}").sort_values(
            "code"
        )
        return list(df.code)

    def show_all_dates(self, table_name: str) -> list:
        """返回表（常用于分钟数据）中所有日期

        Parameters
        ----------
        table_name : str
            表名

        Returns
        -------
        list
            全部日期
        """
        df = self.get_data(f"select distinct(date) from {table_name}").sort_values(
            "date"
        )
        return list(df.date)


class DriverOfPostgre(MetaSQLDriver):
    """能以postgresql和psycopg2驱动连接的数据库"""

    def __init__(
        self, user: str, password: str, host: str, port: str, database: str
    ) -> None:
        """通过postgre的psycopg2驱动连接数据库

        Parameters
        ----------
        user : str
            用户名
        password : str
            密码
        host : str
            地址
        port : str
            端口
        database : str
            数据库名
        """
        super().__init__(user, password, host, port, database)

    def connect(self):
        conn = pg.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        return conn


class Questdb(DriverOfPostgre):
    """Questdb的写入方式都为追加，因此如果想replace之前的数据，请手动删除表格
    Questdb的web console为127.0.0.1:9000，作者已经修改为127.0.0.1:9001"""

    def __init__(
        self,
        user: str = "admin",
        password: str = "quest",
        host: str = "192.168.200.60",
        port: str = "8812",
        database: str = "qdb",
        tmp_csv_path: str = "tmp_dataframe_for_questdb.csv",
        web_port: str = "9000",
    ) -> None:
        """通过postgre的psycopg2驱动连接questdb数据库

        Parameters
        ----------
        user : str, optional
            用户名, by default "admin"
        password : str, optional
            密码, by default "quest"
        host : str, optional
            地址, by default "192.168.200.60"
        port : str, optional
            端口, by default "8812"
        database : str, optional
            数据库, by default "qdb"
        tmp_csv_path : str, optional
            通过csv导入数据时，csv文件的暂存位置, by default "/opt/homebrew/var/questdb/copy_path/tmp_dataframe.csv"
        web_port : str, optional
            questdb控制台的端口号，安装questdb软件时默认为9000，本库默认为9000, by default 9000
        """
        super().__init__(user, password, host, port, database)
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.tmp_csv_path = tmp_csv_path
        self.web_port = web_port

    def __addapt_numpy_float64(self, numpy_float64):
        return AsIs(numpy_float64)

    def __addapt_numpy_int64(self, numpy_int64):
        return AsIs(numpy_int64)

    @retry(stop=stop_after_attempt(3000), wait=wait_fixed(0.01))
    def write_via_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        symbols: Union[str, bool, List[int], List[str]] = None,
        tuple_col: Union[str, List[str]] = None,
    ) -> None:
        """通过questdb的python库直接将dataframe写入quested数据库

        Parameters
        ----------
        df : pd.DataFrame
            要写入的dataframe
        table_name : str
            questdb中该表的表名
        symbols : Union[str, bool, List[int], List[str]], optional
            为symbols的那些列的名称, by default None
        tuple_col : Union[str, List[str]], optional
            数据类型为tuple或list的列的名字, by default None
        """
        if tuple_col is None:
            ...
        elif isinstance(tuple_col, str):
            df[tuple_col] = df[tuple_col].apply(str)
        else:
            for t in tuple_col:
                df[t] = df[t].apply(str)
        if symbols is not None:
            with qdbing.Sender(self.host, 9009) as sender:
            # with qdbing.Sender.from_conf(f"http::addr={self.host}:{self.port};") as sender:
                sender.dataframe(df, table_name=table_name, symbols=symbols)
        else:
            with qdbing.Sender(self.host, 9009) as sender:
                sender.dataframe(df, table_name=table_name)

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(3))
    def get_data_with_tuple(
        self,
        sql_order: str,
        tuple_col: Union[str, List[str]] = "fac",
        without_timestamp: bool = 1,
    ) -> pd.DataFrame:
        """从questdb数据库中，读取那些值中带有元组或列表的表格

        Parameters
        ----------
        sql_order : str
            读取的sql命令
        tuple_col : Union[str, List[str]], optional
            数值类型为元组或列表的那些列的名称, by default 'fac'
        without_timestamp : bool, optional
            读取时是否删去数据库自动加入的名为`timestamp`的列, by default 1

        Returns
        -------
        pd.DataFrame
            读取到的数据
        """
        data = self.get_data(sql_order)

        def eval_it(x):
            if "," in x.iloc[0]:
                x = x.apply(
                    lambda y: [
                        float(i) if y not in ["nan", " nan", "None"] else np.nan
                        for i in y[1:-1].split(",")
                    ]
                )
            else:
                x = x.astype(float)
            return x

        if isinstance(tuple_col, str):
            data[tuple_col] = eval_it(data[tuple_col])
        else:
            for t in tuple_col:
                data[t] = eval_it(data[t])
        if "timestamp" in list(data.columns):
            if without_timestamp:
                data = data.drop(columns=["timestamp"])
        return data

    def write_via_csv(self, df: pd.DataFrame, table: str, index_id: str = None) -> None:
        """以csv中转的方式，将pd.DataFrame写入Questdb，这一方法的速度约为直接写入的20倍以上，建议使用此方法

        Parameters
        ----------
        df : pd.DataFrame
            要存入的pd.DataFrame
        table : str
            表名
        """
        register_adapter(np.float64, self.__addapt_numpy_float64)
        register_adapter(np.int64, self.__addapt_numpy_int64)
        conn = self.connect()
        # SQL quert to execute
        tmp_df = self.tmp_csv_path + str(np.random.randint(100000000))
        if index_id is None:
            df.to_csv(tmp_df, index=None)
        else:
            df.to_csv(tmp_df, index_label=index_id)
        f = open(tmp_df, "r")
        cursor = conn.cursor()
        try:
            csv = {"data": (table, f)}
            server = f"http://{self.host}:{self.web_port}/imp"
            response = requests.post(server, files=csv)
        except (Exception, pg.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
        f.close()
        cursor.close()
        os.remove(tmp_df)

    def show_all_tables(self) -> pd.DataFrame:
        """获取Questdb中所有的表的名称

        Returns
        -------
        pd.DataFrame
            所有表的名称
        """
        return self.get_data("show tables")

    def show_all_codes(self, table_name: str) -> list:
        """返回表（常用于分钟数据）中所有股票的代码

        Parameters
        ----------
        table_name : str
            表名

        Returns
        -------
        list
            全部股票代码
        """
        df = self.get_data(f"select distinct(code) from {table_name}").sort_values(
            "code"
        )
        return list(df.code)

    def show_all_dates(self, table_name: str) -> list:
        """返回表（常用于分钟数据）中所有日期

        Parameters
        ----------
        table_name : str
            表名

        Returns
        -------
        list
            全部日期
        """
        df = self.get_data(f"select distinct(date) from {table_name}").sort_values(
            "date"
        )
        return list(df.date)

    def copy_all_tables(self):
        """下载某个questdb数据库下所有的表格"""
        homeplace = HomePlace()
        path = homeplace.update_data_file + self.host + "_copy/"
        if not os.path.exists(path):
            os.makedirs(path)
        tables = [
            i
            for i in list(self.show_all_tables().table)
            if i
            not in [
                "sys.column_versions_purge_log",
                "telemetry_config",
                "sys.telemetry_wal",
                "telemetry",
            ]
        ]
        logger.info(f"共{len(tables)}个表，分别为{tables}")
        for table in tables:
            logger.info(f"正在备份{table}表……")
            down = self.get_data(f"select * from {table}")
            down.to_parquet(f"{path}{self.host}_{table}.parquet")
            logger.success(f"{table}表备份完成")
        logger.success("所有表备份完成")

    def upload_all_copies(self):
        """上传之前备份在本地的questdb的所有表格"""
        homeplace = HomePlace()
        path = homeplace.update_data_file + self.host + "_copy/"
        files = os.listdir(path)
        files = [i.split(".parquet")[0] for i in files]
        logger.info(f"共{len(files)}个表，分别为{files}")
        for file in files:
            logger.info(f"正在上传{file}表……")
            self.write_via_df(
                pd.read_parquet(path + file + ".parquet"),
                file.split(self.host + "_")[-1],
            )
            logger.success(f"{file}表上传完成")
        logger.success("所有表上传完成")
