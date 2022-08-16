__updated__ = "2022-08-16 15:50:43"

import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sqlalchemy import FLOAT, INT, VARCHAR, BIGINT
from loguru import logger
import datetime
from pure_ocean_breeze.state.state import STATES


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

    def login(self, db_name: str = None):
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
        mycursor = mydb.cursor()
        self.mycursor = mycursor
        return mycursor

    def add_new_database(self, db_name: str = None):
        """添加一个新数据库"""
        mycursor = self.login()
        try:
            mycursor.execute(f"CREATE DATABASE {db_name}")
            logger.success(f"已添加名为{db_name}的数据库")
        except Exception:
            logger.warning(f"已经存在名为{db_name}的数据库，请检查")

    def show_tables_old(self, db_name: str = None, full=True):
        """显示数据库下的所有表"""
        if db_name is None:
            db_name = self.db_name
        mycursor = self.login()
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
        mycursor = self.login()
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
        其中如果主键不制定，则会默认为第一个，主键不能重复，因此会自动保留最后一个。
        创建表格后，需插入一行数，才算创建成功，否则依然不能写入，插入语句如下
    INSERT INTO minute_data.minute_data (date, code, open, high, low, close, amount, money, num) VALUES
                                             (0,0,0,0,0,0,0,0,0);
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

    def set_new_engine(self, engine_uri: str):
        """设置新的网址"""
        self.uri = engine_uri
        self.engine = create_engine(engine_uri)
        logger.success("engine已更改")

    def get_data_old(self, sql_order: str):
        """获取数据"""
        a = pd.read_sql(sql_order, con=self.engine)
        return a

    def get_data(self, sql_order: str) -> pd.DataFrame:
        conn = self.engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql_order)
        columns = [i[0] for i in cursor.description]
        df_data = cursor.fetchall()
        df = pd.DataFrame(df_data, columns=columns)
        return df

    def get_data_old_show_time(self, sql_order: str) -> pd.DataFrame:
        """获取数据，并告知用时"""
        a = datetime.datetime.now()
        df = self.get_data_old(sql_order)
        b = datetime.datetime.now()
        c = b - a
        l = c.seconds + c.microseconds / 1e6
        l = round(l, 2)
        print(f"共用时{l}秒")
        return df

    def get_data_show_time(self, sql_order: str) -> pd.DataFrame:
        """获取数据，并告知用时"""
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
        """查询table这个表中，所有不同的key有哪些，key为某个键的键名，table为表名"""
        df = self.get_data(f"select distinct({key}) from {self.database_name}.{table}")
        return list(df[key])

    # TODO: 将以下两个函数改为，不需要输入表名，也可以返回日期（以时间更长的股票数据表为准）
    def show_all_codes(self, table_name: str) -> list:
        """返回分钟数据中所有股票的代码"""
        df = self.get_data(
            f"select distinct(code) from {self.database_name}.{table_name}"
        ).sort_values("code")
        return [i for i in list(df.code) if i != "0"]

    def show_all_dates(self, table_name: str, mul_100=False) -> list:
        """返回分钟数据中所有日期"""
        df = self.get_data(
            f"select distinct(date) from {self.database_name}.{table_name}"
        ).sort_values("date")
        return [int(i / 100) for i in list(df.date) if i != 0]
