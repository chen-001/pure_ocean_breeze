__updated__ = "2023-02-27 15:22:19"

import numpy as np
import pandas as pd
import time
from sqlalchemy import FLOAT, INT, VARCHAR, BIGINT
import knockknock as kk
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False
import plotly.express as pe
import plotly.io as pio
import tqdm
import os
from loguru import logger
import datetime
from typing import Callable
from functools import reduce,partial
from pure_ocean_breeze.data.database import ClickHouseClient,Questdb
from pure_ocean_breeze.data.tools import drop_duplicates_index
from pure_ocean_breeze.labor.process import pure_moon, decap_industry
from pure_ocean_breeze.data.read_data import read_daily
from pure_ocean_breeze.state.homeplace import HomePlace
from pure_ocean_breeze.data.database import PostgreSQL

try:
    import rqdatac

    rqdatac.init()
except Exception as e:
    print(e)
    desicion = input("米筐连接暂时有错误，是否等待并继续连接，等待请输入y，不等待则输入n")
    if desicion == "y":
        print("连接米筐暂时有错误，将等待30秒后重试")
        time.sleep(30)
        try:
            import rqdatac

            rqdatac.init()
        except Exception as e:
            print(e)
            print("连接米筐暂时有错误，将等待60秒后重试")
            time.sleep(60)
            try:
                import rqdatac

                rqdatac.init()
            except Exception as e:
                print(e)
                print("连接米筐暂时有错误，将等待60秒后重试")
                time.sleep(60)
                try:
                    import rqdatac

                    rqdatac.init()
                except Exception as e:
                    print(e)
                    print("暂时未连接米筐")
    else:
        print("暂时未连接米筐")


class pure_cloud(object):
    """
    为了测试其他不同的频率而设计的类，仅考虑了上市满60天这一要素
    这一回测采取的方案是，对于回测频率n天，将初始资金等分成n笔，每天以1/n的资金调仓
    每笔资金之间相互独立，最终汇聚成一个收益率序列
    """

    def __init__(
        self,
        fac,
        freq,
        group=10,
        boxcox=1,
        trade_cost=0,
        print_comments=1,
        plt_plot=1,
        plotly_plot=0,
        filename="净值走势图",
        comments_writer=None,
        nets_writer=None,
        sheet_name=None,
    ):
        """n是回测的频率，等分成n份，group是回测的组数，boxcox是是否做行业市值中性化"""
        self.fac = fac
        self.freq = freq
        self.group = group
        self.boxcox = boxcox
        self.trade_cost = trade_cost
        ages = read_daily(age=1)
        sts = read_daily(st=1)
        states = read_daily(state=1)
        opens = read_daily(open=1)
        closes = read_daily(close=1)
        capitals = read_daily(flow_cap=1).resample("M").last()
        moon = pure_moon()
        moon.set_basic_data(
            age=ages,
            st=sts,
            state=states,
            open=opens,
            close=closes,
            capital=capitals,
        )
        moon.prerpare()
        ages = moon.ages.copy()
        ages = (ages >= 60) + 0
        self.ages = ages.replace(0, np.nan)
        self.closes = read_daily(close=1)
        self.rets = (
            (self.closes.shift(-self.freq) / self.closes - 1) * self.ages
        ) / self.freq
        self.run(
            print_comments=print_comments,
            plt_plot=plt_plot,
            plotly_plot=plotly_plot,
            filename=filename,
        )
        if comments_writer:
            if sheet_name:
                self.long_short_comments.to_excel(
                    comments_writer, sheet_name=sheet_name
                )
            else:
                raise AttributeError("必须制定sheet_name参数🤒")
        if nets_writer:
            if sheet_name:
                self.group_nets.to_excel(nets_writer, sheet_name=sheet_name)
            else:
                raise AttributeError("必须制定sheet_name参数🤒")

    def comments(self, series, series1):
        """对twins中的结果给出评价
        评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率"""
        ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        duration = (series.index[-1] - series.index[0]).days
        year = duration / 365
        ret_yearly = (series.iloc[-1] / series.iloc[0]) ** (1 / year) - 1
        max_draw = -(series / series.expanding(1).max() - 1).min()
        vol = np.std(series1) * (250**0.5)
        sharpe = ret_yearly / vol
        wins = series1[series1 > 0]
        win_rate = len(wins) / len(series1)
        return pd.Series(
            [ret, ret_yearly, vol, sharpe, max_draw, win_rate],
            index=["总收益率", "年化收益率", "年化波动率", "信息比率", "最大回撤率", "胜率"],
        )

    @kk.desktop_sender(title="嘿，变频回测结束啦～🗓")
    def run(self, print_comments, plt_plot, plotly_plot, filename):
        """对因子值分组并匹配"""
        if self.boxcox:
            self.fac = decap_industry(self.fac,daily=1)
        self.fac = self.fac.T.apply(
            lambda x: pd.qcut(x, self.group, labels=False, duplicates="drop")
        ).T
        self.fac = self.fac.shift(1)
        self.vs = [
            (((self.fac == i) + 0).replace(0, np.nan) * self.rets).mean(axis=1)
            for i in range(self.group)
        ]
        self.group_rets = pd.DataFrame(
            {f"group{k}": list(v) for k, v in zip(range(1, self.group + 1), self.vs)},
            index=self.vs[0].index,
        )
        self.group_rets = self.group_rets.dropna(how="all")
        self.group_rets = self.group_rets
        self.group_nets = (self.group_rets + 1).cumprod()
        self.group_nets = self.group_nets.apply(lambda x: x / x.iloc[0])
        self.one = self.group_nets["group1"]
        self.end = self.group_nets[f"group{self.group}"]
        if self.one.iloc[-1] > self.end.iloc[-1]:
            self.long_name = "group1"
            self.short_name = f"group{self.group}"
        else:
            self.long_name = f"group{self.group}"
            self.short_name = "group1"
        self.long_short_ret = (
            self.group_rets[self.long_name] - self.group_rets[self.short_name]
        )
        self.long_short_net = (self.long_short_ret + 1).cumprod()
        self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
        if self.long_short_net.iloc[-1] < 1:
            self.long_short_ret = (
                self.group_rets[self.short_name] - self.group_rets[self.long_name]
            )
            self.long_short_net = (self.long_short_ret + 1).cumprod()
            self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
            self.long_short_ret = (
                self.group_rets[self.short_name]
                - self.group_rets[self.long_name]
                - 2 * self.trade_cost / self.freq
            )
            self.long_short_net = (self.long_short_ret + 1).cumprod()
            self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
        else:
            self.long_short_ret = (
                self.group_rets[self.long_name]
                - self.group_rets[self.short_name]
                - 2 * self.trade_cost / self.freq
            )
            self.long_short_net = (self.long_short_ret + 1).cumprod()
            self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
        self.group_rets = pd.concat(
            [self.group_rets, self.long_short_ret.to_frame("long_short")], axis=1
        )
        self.group_nets = pd.concat(
            [self.group_nets, self.long_short_net.to_frame("long_short")], axis=1
        )
        self.long_short_comments = self.comments(
            self.long_short_net, self.long_short_ret
        )
        if print_comments:
            print(self.long_short_comments)
        if plt_plot:
            self.group_nets.plot(rot=60)
            plt.savefig(filename + ".png")
            plt.show()
        if plotly_plot:
            fig = pe.line(self.group_nets)
            filename_path = filename + ".html"
            pio.write_html(fig, filename_path, auto_open=True)


class pure_moonson(object):
    """行业轮动回测框架"""

    def __init__(self, fac, group_num=5):
        homeplace = HomePlace()
        pindu = (
            pd.read_parquet(homeplace.daily_data_file + "各行业行情数据.parquet")
            .resample("M")
            .last()
        )
        rindu = pindu / pindu.shift(1) - 1
        self.rindu = rindu
        self.fac = fac
        self.group = self.get_groups(fac, group_num)
        print("未完工，待完善，暂时请勿使用⚠️")

    def get_groups(self, df, groups_num):
        """依据因子值，判断是在第几组"""
        if "group" in list(df.columns):
            df = df.drop(columns=["group"])
        df = df.sort_values(["fac"], ascending=True)
        each_group = round(df.shape[0] / groups_num)
        l = list(
            map(
                lambda x, y: [x] * y,
                list(range(1, groups_num + 1)),
                [each_group] * groups_num,
            )
        )
        l = reduce(lambda x, y: x + y, l)
        if len(l) < df.shape[0]:
            l = l + [groups_num] * (df.shape[0] - len(l))
        l = l[: df.shape[0]]
        df.insert(0, "group", l)
        return df


class pure_fall_flexible(object):
    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
        clickhouse: bool = 0,
        questdb: bool = 0,
    ) -> None:
        """基于clickhouse的分钟数据，计算因子值，每天的因子值用到多日的数据，或者用到截面的数据
        对一段时间的截面数据进行操作，在get_daily_factors的func函数中
        请写入df=df.groupby([xxx]).apply(fff)之类的语句
        然后单独定义一个函数，作为要apply的fff，可以在apply上加进度条

        Parameters
        ----------
        factor_file : str
            用于存储因子的文件名称，请以'.parquet'结尾
        startdate : int, optional
            计算因子的起始日期，形如20220816, by default None
        enddate : int, optional
            计算因子的终止日期，形如20220816, by default None
        kind : str, optional
            指定计算股票还是指数，指数则为'index', by default "stock"
        clickhouse : bool, optional
            使用clickhouse作为数据源，如果postgresql与本参数都为0，将依然从clickhouse中读取, by default 0
        questdb : bool, optional
            使用questdb作为数据源, by default 0
        """
        homeplace = HomePlace()
        self.kind = kind
        if clickhouse == 0 and questdb == 0:
            clickhouse = 1
        self.clickhouse = clickhouse
        self.questdb = questdb
        if clickhouse == 1:
            # 连接clickhouse
            self.chc = ClickHouseClient("minute_data")
        elif questdb:
            self.chc = Questdb()
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = drop_duplicates_index(pd.read_parquet(self.factor_file))
            self.factor_old = factor_old
            # 已经算好的日子
            dates_old = sorted(list(factor_old.index.strftime("%Y%m%d").astype(int)))
            self.dates_old = dates_old
        else:
            self.factor_old = None
            self.dates_old = []
            logger.info("这个因子以前没有，正在重新计算")
        # 读取当前所有的日子
        dates_all = self.chc.show_all_dates(f"minute_data_{kind}")
        if startdate is None:
            ...
        else:
            dates_all = [i for i in dates_all if i >= startdate]
        if enddate is None:
            ...
        else:
            dates_all = [i for i in dates_all if i <= enddate]
        self.dates_all = dates_all
        # 需要新补充的日子
        self.dates_new = sorted([i for i in dates_all if i not in dates_old])

    def __call__(self) -> pd.DataFrame:
        """直接返回因子值的pd.DataFrame

        Returns
        -------
        `pd.DataFrame`
            计算出的因子值
        """
        return self.factor.copy()

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func: Callable,
        fields: str = "*",
        chunksize: int = 250,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        """每次抽取chunksize天的截面上全部股票的分钟数据
        依照定义的函数计算因子值

        Parameters
        ----------
        func : Callable
            用于计算因子值的函数
        fields : str, optional
            股票数据涉及到哪些字段，排除不必要的字段，可以节约读取数据的时间，形如'date,code,num,close,amount,open'
            提取出的数据，自动按照code,date,num排序，因此code,date,num是必不可少的字段, by default "*"
        chunksize : int, optional
            每次读取的截面上的天数, by default 10
        show_time : bool, optional
            展示每次读取数据所需要的时间, by default 0
        tqdm_inside : bool, optional
            将进度条加在内部，而非外部，建议仅chunksize较大时使用, by default 0
        """
        the_func = partial(func)
        # 将需要更新的日子分块，每200天一组，一起运算
        dates_new_len = len(self.dates_new)
        if dates_new_len > 0:
            cut_points = list(range(0, dates_new_len, chunksize)) + [dates_new_len - 1]
            if cut_points[-1] == cut_points[-2]:
                cut_points = cut_points[:-1]
            self.cut_points = cut_points
            self.factor_new = []
            # 开始计算因子值
            if tqdm_inside:
                # 开始计算因子值
                for date1, date2 in cut_points:
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]}"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
                        df = ((df.set_index("code")) / 100).reset_index()
                    tqdm.auto.tqdm.pandas()
                    df = the_func(df)
                    if isinstance(df, pd.Series):
                        df = df.reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            else:
                # 开始计算因子值
                for date1, date2 in tqdm.auto.tqdm(cut_points, desc="不知乘月几人归，落月摇情满江树。"):
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
                        df = ((df.set_index("code")) / 100).reset_index()
                    df = the_func(df)
                    if isinstance(df, pd.Series):
                        df = df.reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            self.factor_new = pd.concat(self.factor_new)
            # 拼接新的和旧的
            self.factor = pd.concat([self.factor_old, self.factor_new]).sort_index()
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.to_parquet(self.factor_file)
            logger.info(f"截止到{new_end_date}的因子值计算完了")
        elif dates_new_len == 1:
            print("共1天")
            if tqdm_inside:
                # 开始计算因子值
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                tqdm.auto.tqdm.pandas()
                df = the_func(df)
                if isinstance(df, pd.Series):
                    df = df.reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            else:
                # 开始计算因子值
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                df = the_func(df)
                if isinstance(df, pd.Series):
                    df = df.reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                self.factor_new.append(df)
            self.factor_new = df
            # 拼接新的和旧的
            self.factor = (
                pd.concat([self.factor_old, self.factor_new])
                .sort_index()
                .drop_duplicates()
            )
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.to_parquet(self.factor_file)
            logger.info(f"补充{self.dates_new[0]}截止到{new_end_date}的因子值计算完了")
        else:
            self.factor = self.factor_old
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")
            
            
def database_update_minute_data_to_postgresql(kind: str) -> None:
    """使用米筐更新分钟数据至postgresql中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'

    Raises
    ------
    `IOError`
        如果未指定股票还是指数，将报错
    """
    if kind == "stock":
        code_type = "CS"
    elif kind == "index":
        code_type = "INDX"
    else:
        raise IOError("总得指定一种类型吧？请从stock和index中选一个")
    # 获取剩余使用额
    user1 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    logger.info(f"今日已使用rqsdk流量{user1}MB")
    # 获取全部股票/指数代码
    cs = rqdatac.all_instruments(type=code_type, market="cn", date=None)
    codes = list(cs.order_book_id)
    # 获取上次更新截止时间
    try:
        qdb = Questdb()
        last_date = max(qdb.show_all_dates(f"minute_data_{kind}"))
    except Exception:
        qdb = Questdb(web_port="9000")
        last_date = max(qdb.show_all_dates(f"minute_data_{kind}"))
    # 本次更新起始日期
    start_date = pd.Timestamp(str(last_date)) + pd.Timedelta(days=1)
    start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
    # 本次更新终止日期
    end_date = datetime.datetime.now()
    if end_date.hour < 17:
        end_date = end_date - pd.Timedelta(days=1)
    end_date = datetime.datetime.strftime(end_date, "%Y-%m-%d")
    logger.info(f"本次将下载从{start_date}到{end_date}的数据")
    # 下载数据
    ts = rqdatac.get_price(
        codes,
        start_date=start_date,
        end_date=end_date,
        frequency="1m",
        fields=["volume", "total_turnover", "high", "low", "close", "open"],
        adjust_type="none",
        skip_suspended=False,
        market="cn",
        expect_df=True,
        time_slice=None,
    )
    # 调整数据格式
    ts = ts.reset_index()
    ts = ts.rename(
        columns={
            "order_book_id": "code",
            "datetime": "date",
            "volume": "amount",
            "total_turnover": "money",
        }
    )
    ts = ts.sort_values(["code", "date"])
    ts.date = ts.date.dt.strftime("%Y%m%d").astype(int)
    ts = ts.groupby(["code", "date"]).apply(
        lambda x: x.assign(num=list(range(1, x.shape[0] + 1)))
    )
    ts.code = ts.code.str.replace(".XSHE", ".SZ")
    ts.code = ts.code.str.replace(".XSHG", ".SH")
    # 数据写入数据库
    pgdb = PostgreSQL("minute_data")
    ts.to_sql(
        f"minute_data_{kind}",
        pgdb.engine,
        if_exists="append",
        index=False,
        dtype={
            "code": VARCHAR(9),
            "date": INT,
            "open": FLOAT,
            "high": FLOAT,
            "low": FLOAT,
            "close": FLOAT,
            "amount": FLOAT,
            "money": FLOAT,
            "num": INT,
        },
    )
    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")
