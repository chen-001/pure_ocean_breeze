__updated__ = "2022-11-05 00:12:35"

import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sqlalchemy import FLOAT, INT, VARCHAR, BIGINT
from loguru import logger
import datetime
import psycopg2 as pg
import psycopg2.extras as extras
import numpy as np
import requests
import os
from typing import Union
from psycopg2.extensions import register_adapter, AsIs
from pure_ocean_breeze.legacy_version.v3p4.state.states import STATES


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


class sqlConfig(object):
    def __init__(
        self,
        db_name: str = None,
        db_user: str = STATES["db_user"],
        db_host: str = STATES["db_host"],
        db_port: int = STATES["db_port"],
        db_password: str = STATES["db_password"],
    ):
        # 初始化数据库连接，使用pymysql模块
        db_info = {
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
            "database": db_name,
        }
        self.db_name = db_name
        self.db_info = db_info
        self.engine = create_engine(
            "mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8"
            % db_info,
            encoding="utf-8",
        )

    def connect(self, db_name: str = None):
        """以pymysql的方式登录数据库，进行更灵活的操作"""
        if db_name is None:
            mydb = pymysql.connect(
                host=self.db_info["host"],
                user=self.db_info["user"],
                password=self.db_info["password"],
            )
        else:
            mydb = pymysql.connect(
                host=self.db_info["host"],
                user=self.db_info["user"],
                password=self.db_info["password"],
                db=db_name,
            )
        return mydb

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

    def show_tables_old(self, db_name: str = None, full=True):
        """显示数据库下的所有表"""
        if db_name is None:
            db_name = self.db_name
        mydb = self.connect()
        mycursor = mydb.cursor()
        if full:
            return mycursor.execute(
                f"select * from information_schema.tables where TABLE_SCHEMA={f'{db_name}'}"
            )
        else:
            return mycursor.execute(
                f"select TABLE_NAME from information_schema.tables where TABLE_SCHEMA={f'{db_name}'}"
            )

    def show_tables(self, db_name: str = None, full: bool = True):
        """显示数据库下的所有表"""
        db_info = self.db_info
        db_info["database"] = "information_schema"
        engine = create_engine(
            "mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8"
            % db_info,
            encoding="utf-8",
        )
        if db_name is None:
            db_name = self.db_name
        if full:
            res = self.get_data_sql_order(
                f"select * from information_schema.tables where TABLE_SCHEMA='{db_name}'"
            )
        else:
            res = self.get_data_sql_order(
                f"select TABLE_NAME from information_schema.tables where TABLE_SCHEMA='{db_name}'"
            )
        res.columns = res.columns.str.lower()
        if full:
            return res
        else:
            return list(sorted(res.table_name))

    def show_databases(self, user_only: bool = True, show_number: bool = True) -> list:
        """显示数据库信息"""
        mydb = self.connect()
        mycursor = mydb.cursor()
        res = self.get_data_sql_order(
            "select SCHEMA_NAME from information_schema.schemata"
        )
        res = list(res.SCHEMA_NAME)
        di = {}
        if user_only:
            res = res[4:]
        if show_number:
            for i in res:
                di[i] = mycursor.execute(
                    f"select * from information_schema.tables where TABLE_SCHEMA='{i}'"
                )
            return di
        else:
            return res

    def get_data_sql_order(self, sql_order: str) -> pd.DataFrame:
        conn = self.engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql_order)
        columns = [i[0] for i in cursor.description]
        df_data = cursor.fetchall()
        df = pd.DataFrame(df_data, columns=columns)
        return df

    def get_data_old(
        self,
        table_name: str,
        fields: str = None,
        startdate: int = None,
        enddate: int = None,
        show_time=False,
    ) -> pd.DataFrame:
        """
        从数据库中读取数据，
        table_name为表名，数字开头的加键盘左上角的`符号，形如`000001.SZ`或`20220717`
        fields形如'date,close,open.amount'，不指定则默认读入所有列
        startdate形如20130326，不指定则默认从头读
        enddate形如20220721，不指定则默认读到尾
        """
        if show_time:
            a = datetime.datetime.now()
        if table_name[0].isdigit():
            table_name = f"`{table_name}`"
        if fields is None:
            fields = "*"
        if startdate is None and enddate is None:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name}"
        elif startdate is None and enddate is not None:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name} where date<={enddate}"
        elif startdate is not None and enddate is None:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name} where date>={startdate}"
        else:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name} where date>={startdate} and date<={enddate}"
        self.sql_order = sql_order
        res = pd.read_sql(sql_order, self.engine)
        res.columns = res.columns.str.lower()
        if show_time:
            b = datetime.datetime.now()
            c = b - a
            l = c.seconds + c.microseconds / 1e6
            l = round(l, 2)
            print(f"共用时{l}秒")
        return res

    def get_data(
        self,
        table_name: str,
        fields: str = None,
        startdate: int = None,
        enddate: int = None,
        show_time=False,
    ) -> pd.DataFrame:
        """
        从数据库中读取数据，
        `table_name`为表名，数字开头的加键盘左上角的
        ```sql
        `
        ```
        符号
        形如
        ```sql
        `000001.SZ`
        ```
        或
        ```sql
        `20220717`
        ```
        `fields`形如
        ```sql
        'date,close,open.amount'
        ```
        不指定则默认读入所有列
        `startdate`形如
        ```sql
        `20130326`
        ```
        不指定则默认从头读
        `enddate`形如
        ```sql
        `20220721`
        ```
        不指定则默认读到尾
        """
        if show_time:
            a = datetime.datetime.now()
        if table_name[0].isdigit():
            table_name = f"`{table_name}`"
        if fields is None:
            fields = "*"
        if startdate is None and enddate is None:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name}"
        elif startdate is None and enddate is not None:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name} where date<={enddate}"
        elif startdate is not None and enddate is None:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name} where date>={startdate}"
        else:
            sql_order = f"SELECT {fields} FROM {self.db_name}.{table_name} where date>={startdate} and date<={enddate}"
        self.sql_order = sql_order
        res = self.get_data_sql_order(sql_order)
        res.columns = res.columns.str.lower()
        if show_time:
            b = datetime.datetime.now()
            c = b - a
            l = c.seconds + c.microseconds / 1e6
            l = round(l, 2)
            print(f"共用时{l}秒")
        return res


class ClickHouseClient(object):
    """clickhouse的一些功能，clickhouse写入数据前，需要先创建表格，表格如果不存在则不能写入
    clickhouse创建表格使用语句如下
    ```sql
    CREATE TABLE minute_data.minute_data
    (   `date` int,
        `num` int,
        `code` VARCHAR(9),
        `open` int,
        `high` int,
        `low` int,
        `close` int,
        `amount` bigint,
        `money` bigint
    ) ENGINE = ReplacingMergeTree()
        PRIMARY KEY(date,num)
        ORDER BY (date, num);
    ```
        其中如果主键不制定，则会默认为第一个，主键不能重复，因此会自动保留最后一个。
        创建表格后，需插入一行数，才算创建成功，否则依然不能写入，插入语句如下
    ```sql
    INSERT INTO minute_data.minute_data (date, code, open, high, low, close, amount, money, num) VALUES
                                            (0,0,0,0,0,0,0,0,0);
    ```
    """

    def __init__(
        self,
        database_name: str,
        database_host: str = "127.0.0.1",
        database_user: str = "default",
        database_password="",
    ):
        self.database_name = database_name
        self.database_host = database_host
        self.database_user = database_user
        self.database_password = database_password
        self.uri = f"clickhouse+native://{database_host}/{database_name}"
        self.engine = create_engine(self.uri)
        # engine = create_engine(self.uri)
        # session = make_session(self.engine)
        # metadata = MetaData(bind=engine)
        #
        # Base = get_declarative_base(metadata=metadata)

    def set_new_engine(self, engine_uri: str) -> None:
        """设置新的地址

        Parameters
        ----------
        engine_uri : str
            新的数据库地址
        """
        self.uri = engine_uri
        self.engine = create_engine(engine_uri)
        logger.success("engine已更改")

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
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        return cur.execute(sql_order)

    def get_data_old(self, sql_order: str) -> pd.DataFrame:
        """以pandas.read_sql的方式读取数据

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
        conn = self.engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql_order)
        df_data = cursor.fetchall()
        if not only_array:
            columns = [i[0] for i in cursor.description]
            df = pd.DataFrame(df_data, columns=columns)
            return df
        else:
            return np.array(df_data)

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

    def get_data_show_time(self, sql_order: str) -> pd.DataFrame:
        """以cursor和sql命令的方式，从数据库中读取数据，并告知所用时间

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

    def save_data(self, df, sql_order: str, if_exists="append", index=False):
        """存储数据，if_exists可以为append或replace或fail，默认append，index为是否保存df的index"""
        raise IOError(
            """
            请使用pandas自带的df.to_sql()来存储，存储时请注意把小数都转化为整数，例如*100（分钟数据都做了这个处理）
            请勿携带空值，提前做好fillna处理。大于2147000000左右的值，请指定类型为bigint，否则为int即可
            句式如：
            (np.around(min1,2)*100).ffill().astype(int).assign(code='000001.SZ').to_sql('minute_data',engine,if_exists='append',index=False)
            """
        )

    def show_all_xxx_in_tableX(self, key: str, table: str) -> list:
        """查询table这个表中，所有不同的key有哪些

        Parameters
        ----------
        key : str
            键的名字
        table : str
            表的名字

        Returns
        -------
        list
            表中全部的键
        """
        df = self.get_data(f"select distinct({key}) from {self.database_name}.{table}")
        return list(df[key])

    # TODO: 将以下两个函数改为，不需要输入表名，也可以返回日期（以时间更长的股票数据表为准）
    def show_all_codes(self, table_name: str) -> list:
        """返回表中所有股票的代码（常用于分钟数据）

        Parameters
        ----------
        table_name : str
            表名

        Returns
        -------
        list
            表中所有的股票代码
        """
        df = self.get_data(
            f"select distinct(code) from {self.database_name}.{table_name}"
        ).sort_values("code")
        return [i for i in list(df.code) if i != "0"]

    def show_all_dates(self, table_name: str, mul_100=False) -> list:
        """返回分钟数据中所有日期（常用于分钟数据）

        Parameters
        ----------
        table_name : str
            表名
        mul_100 : bool, optional
            返回的日期是否成以100, by default False

        Returns
        -------
        list
            表中所有的日期
        """
        df = self.get_data(
            f"select distinct(date) from {self.database_name}.{table_name}"
        ).sort_values("date")
        if mul_100:
            return [i for i in list(df.date) if i != 0]
        else:
            return [int(i / 100) for i in list(df.date) if i != 0]


class PostgreSQL(DriverOfPostgre):
    def __init__(
        self,
        database: str = None,
        user: str = "postgres",
        password: str = "Kingwila98",
        host: str = "127.0.0.1",
        port: int = 5433,
    ) -> None:
        """连接postgresql数据库

        Parameters
        ----------
        database : str, optional
            _description_, by default None
        user : str, optional
            _description_, by default 'postgres'
        password : str, optional
            _description_, by default 'Kingwila98'
        host : str, optional
            _description_, by default '127.0.0.1'
        port : int, optional
            _description_, by default 5433
        """
        super().__init__(user, password, host, port, database)
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )


class Questdb(DriverOfPostgre):
    """Questdb的写入方式都为追加，因此如果想replace之前的数据，请手动删除表格
    Questdb的web console为127.0.0.1:9000，作者已经修改为127.0.0.1:9001"""

    def __init__(
        self,
        user="admin",
        password="quest",
        host="127.0.0.1",
        port="8812",
        database="qdb",
        tmp_csv_path="tmp_dataframe_for_questdb.csv",
    ) -> None:
        """通过postgre的psycopg2驱动连接questdb数据库

        Parameters
        ----------
        user : str, optional
            用户名, by default "admin"
        password : str, optional
            密码, by default "quest"
        host : str, optional
            地址, by default "127.0.0.1"
        port : str, optional
            端口, by default "8812"
        database : str, optional
            数据库, by default "qdb"
        tmp_csv_path : str, optional
            通过csv导入数据时，csv文件的暂存位置, by default "/opt/homebrew/var/questdb/copy_path/tmp_dataframe.csv"
        """
        super().__init__(user, password, host, port, database)
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.tmp_csv_path = tmp_csv_path

    def __addapt_numpy_float64(self, numpy_float64):
        return AsIs(numpy_float64)

    def __addapt_numpy_int64(self, numpy_int64):
        return AsIs(numpy_int64)

    def write_via_df(
        self,
        df: pd.DataFrame,
        table: str,
        str_col: list[str] = None,
        date_col: list[str] = None,
        time_col: list[str] = None,
        data_dict: dict = None,
    ) -> None:
        """通过postgre的方式，直接将pd.Dataframe写入Questdb数据库，此函数不必提前单独创建table，在本函数中会自动检测表是否存在并创建

        Parameters
        ----------
        df : pd.DataFrame
            要写入的数据
        table : str
            要写入的表名
        str_col : list[str], optional
            类型为str的列的列名, by default None
        date_col : list[str], optional
            类型为date的列的列名, by default None
        time_col : list[str], optional
            类型为timestamp的列的列名, by default None
        data_dict : dict, optional
            如果不指定上述参数，也可以通过这一参数，指定所有参数类型，传入字典形式，且value应为字符串，如`'FLOAT'`, by default None
        """
        register_adapter(np.float64, self.__addapt_numpy_float64)
        register_adapter(np.int64, self.__addapt_numpy_int64)
        conn = self.connect()
        # Create a list of tupples from the dataframe values
        tuples = [tuple(x) for x in df.to_numpy()]
        # Comma-separated dataframe columns
        cols = ",".join(list(df.columns))
        # SQL quert to execute
        query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
        cursor = conn.cursor()
        try:
            extras.execute_values(cursor, query, tuples)
            conn.commit()
        except Exception:
            try:
                startstr = "("
                if data_dict is None:
                    data_dict = {k: "FLOAT" for k in list(df.columns)}
                    if str_col is not None:
                        data_dict_str = {k: "symbol" for k in str_col}
                        data_dict.update(data_dict_str)
                    if date_col is not None:
                        data_dict_date = {k: "date" for k in date_col}
                        data_dict.update(data_dict_date)
                    if time_col is not None:
                        data_dict_time = {k: "timestamp" for k in time_col}
                        data_dict.update(data_dict_time)
                        tail = f"timestamp({time_col[0]})"
                for k, v in data_dict.items():
                    startstr = startstr + k + " " + v + ", "
                startstr = startstr[:-2] + ")"
                if time_col is not None:
                    startstr = startstr + " " + tail
                cursor.execute(f"create table {table} {startstr}")
                extras.execute_values(cursor, query, tuples)
                conn.commit()
            except (Exception, pg.DatabaseError) as error:
                print("Error: %s" % error)
                conn.rollback()
                cursor.close()
        cursor.close()

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
            server = "http://localhost:9001/imp"
            response = requests.post(server, files=csv)
        except (Exception, pg.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
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
