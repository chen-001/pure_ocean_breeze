__updated__ = "2025-02-26 15:30:49"

import pandas as pd
import psycopg2 as pg
import numpy as np
from typing import Union
from tenacity import retry, stop_after_attempt, wait_fixed
import questdb.ingress as qdbing


class Questdb(object):
    """Questdb数据库连接和操作类
    
    Questdb的写入方式都为追加，因此如果想replace之前的数据，请手动删除表格
    Questdb的web console为127.0.0.1:9000，作者已经修改为127.0.0.1:9001
    """

    def __init__(
        self,
        user: str = "admin",
        password: str = "quest",
        host: str = "192.168.200.60",
        port: str = "8812",
        database: str = "qdb",
    ) -> None:
        """初始化Questdb连接参数

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
        web_port : str, optional
            questdb控制台的端口号，安装questdb软件时默认为9000，本库默认为9000, by default 9000
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def connect(self):
        """连接数据库

        Returns
        -------
        connection
            数据库连接
        """
        conn = pg.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        return conn

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

    @retry(stop=stop_after_attempt(10))
    def get_data(
        self, sql_order: str, only_array: bool = 0
    ) -> Union[pd.DataFrame, np.ndarray]:
        """以sql命令的方式，从数据库中读取数据

        Parameters
        ----------
        sql_order : str
            sql命令
        only_array : bool, optional
            是否只返回数组, by default 0

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

    @retry(stop=stop_after_attempt(3000), wait=wait_fixed(0.01))
    def write_via_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        symbols: Union[str, bool, list[int], list[str]] = None,
        tuple_col: Union[str, list[str]] = None,
    ) -> None:
        """通过questdb的python库直接将dataframe写入quested数据库

        Parameters
        ----------
        df : pd.DataFrame
            要写入的dataframe
        table_name : str
            questdb中该表的表名
        symbols : Union[str, bool, list[int], list[str]], optional
            为symbols的那些列的名称, by default None
        tuple_col : Union[str, list[str]], optional
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
                sender.dataframe(df, table_name=table_name, symbols=symbols)
        else:
            with qdbing.Sender(self.host, 9009) as sender:
                sender.dataframe(df, table_name=table_name)




    