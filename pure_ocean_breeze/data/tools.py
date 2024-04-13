"""
针对一些不常见的文件格式，读取数据文件的一些工具函数，以及其他数据工具
"""

__updated__ = "2023-07-10 12:46:10"

import os
import pandas as pd
import tqdm.auto
import datetime
import scipy.io as scio
import numpy as np
import numpy_ext as npext
import knockknock as kk
import scipy.stats as ss
from functools import reduce, partial
from loguru import logger
from typing import Callable, Union, Dict, List, Tuple

try:
    import rqdatac

    rqdatac.init()
except Exception:
    print("暂时未连接米筐")
from pure_ocean_breeze.state.homeplace import HomePlace
import deprecation
from pure_ocean_breeze import __version__
from pure_ocean_breeze.state.decorators import do_on_dfs

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


@do_on_dfs
def read_mat(path: str) -> pd.DataFrame:
    """读取mat文件

    Parameters
    ----------
    path : str
        mat文件路径

    Returns
    -------
    `pd.DataFrame`
        字典的第4个value
    """
    return list(scio.loadmat(path).values())[3]


@do_on_dfs
def convert_code(x: str) -> Tuple[str, str]:
    """将米筐代码转换为wind代码，并识别其是股票还是指数

    Parameters
    ----------
    x : str
        米筐的股票/指数代码，以 XSHE 或 XSHG 结尾

    Returns
    -------
    `Tuple[str,str]`
        转换后的股票/指数代码，以及该代码属于股票还是指数
    """
    x1 = x.split("/")[-1].split(".")[0]
    x2 = x.split("/")[-1].split(".")[1]
    if x2 == "XSHE":
        x2 = ".SZ"
    elif x2 == "XSHG":
        x2 = ".SH"
    elif x2 == "SZ":
        x2 = ".XSHE"
    elif x2 == "SH":
        x2 = ".XSHG"
    x = x1 + x2
    if (x1[0] == "0" or x1[:2] == "30") and x2 in [".SZ", ".XSHE"]:
        kind = "stock"
    elif x1[0] == "6" and x2 in [".SH", ".XSHG"]:
        kind = "stock"
    else:
        kind = "index"
    return x, kind


@do_on_dfs
def add_suffix(code: str) -> str:
    """给股票代码加上后缀

    Parameters
    ----------
    code : str
        纯数字组成的字符串类型的股票代码，如000001

    Returns
    -------
    str
        添加完后缀后的股票代码，如000001.SZ
    """
    if not isinstance(code, str):
        code = str(code)
    if len(code) < 6:
        code = "0" * (6 - len(code)) + code
    if code.startswith("0") or code.startswith("3"):
        code = ".".join([code, "SZ"])
    elif code.startswith("6"):
        code = ".".join([code, "SH"])
    elif code.startswith("8"):
        code = ".".join([code, "BJ"])
    return code


@do_on_dfs
def get_value(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """很多因子计算时，会一次性生成很多值，使用时只取出一个值

    Parameters
    ----------
    df : pd.DataFrame
        每个value是一个列表或元组的pd.DataFrame
    n : int
        取第n个值

    Returns
    -------
    `pd.DataFrame`
        仅有第n个值构成的pd.DataFrame
    """

    def get_value_single(x, n):
        try:
            return x[n]
        except Exception:
            return np.nan

    df = df.applymap(lambda x: get_value_single(x, n))
    return df


@do_on_dfs
def indus_name(df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
    """将2021版申万行业的代码，转化为对应行业的名字

    Parameters
    ----------
    df : pd.DataFrame
        一个包含申万一级行业代码的pd.DataFrame，其中某一列或index为行业代码
    col_name : str, optional
        仅某列为行业代码时指定该参数，该列的名字，否则默认转化index, by default None

    Returns
    -------
    `pd.DataFrame`
        转化后的pd.DataFrame
    """
    names = pd.DataFrame(
        {
            "indus_we_cant_same": [
                "801170.SI",
                "801010.SI",
                "801140.SI",
                "801080.SI",
                "801780.SI",
                "801110.SI",
                "801230.SI",
                "801950.SI",
                "801180.SI",
                "801040.SI",
                "801740.SI",
                "801890.SI",
                "801770.SI",
                "801960.SI",
                "801200.SI",
                "801120.SI",
                "801710.SI",
                "801720.SI",
                "801880.SI",
                "801750.SI",
                "801050.SI",
                "801790.SI",
                "801150.SI",
                "801980.SI",
                "801030.SI",
                "801730.SI",
                "801160.SI",
                "801130.SI",
                "801210.SI",
                "801970.SI",
                "801760.SI",
            ],
            "行业名称": [
                "交通运输",
                "农林牧渔",
                "轻工制造",
                "电子",
                "银行",
                "家用电器",
                "综合",
                "煤炭",
                "房地产",
                "钢铁",
                "国防军工",
                "机械设备",
                "通信",
                "石油石化",
                "商贸零售",
                "食品饮料",
                "建筑材料",
                "建筑装饰",
                "汽车",
                "计算机",
                "有色金属",
                "非银金融",
                "医药生物",
                "美容护理",
                "基础化工",
                "电力设备",
                "公用事业",
                "纺织服饰",
                "社会服务",
                "环保",
                "传媒",
            ],
        }
    ).sort_values(["indus_we_cant_same"])
    if col_name:
        names = names.rename(columns={"indus_we_cant_same": col_name})
        df = pd.merge(df, names, on=[col_name])
    else:
        df = df.reset_index()
        df = df.rename(columns={list(df.columns)[0]: "indus_we_cant_same"})
        df = (
            pd.merge(df, names, on=["indus_we_cant_same"])
            .set_index("行业名称")
            .drop(columns=["indus_we_cant_same"])
        )
    return df


def rqdatac_show_used() -> float:
    """查询流量使用情况

    Returns
    -------
    `float`
        当日已经使用的流量MB数
    """
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    print(f"今日已使用rqsdk流量{user2}MB")
    return user2


@do_on_dfs
def add_suffix(code: str) -> str:
    """给没有后缀的股票代码加上wind后缀

    Parameters
    ----------
    code : str
        没有后缀的股票代码

    Returns
    -------
    `str`
        加完wind后缀的股票代码
    """
    if code.startswith("0") or code.startswith("3"):
        code = code + ".SZ"
    elif code.startswith("6"):
        code = code + ".SH"
    elif code.startswith("8"):
        code = code + ".BJ"
    else:
        code = code + ".UN"
    return code


@do_on_dfs
def 生成每日分类表(
    df: pd.DataFrame, code: str, entry: str, exit: str, kind: str
) -> pd.DataFrame:
    """
    ```
    df是要包含任意多列的表格，为dataframe格式，主要内容为，每一行是
    一只股票或一只基金的代码、分类、进入该分类的时间、移除该分类的时间，
    除此之外，还可以包含很多其他内容
    code是股票代码列的列名，为字符串格式；
    entry是股票进入该分类的日期的列名，为字符串格式
    exit是股票退出该分类的日期的列名，为字符串格式
    kind是分类列的列名，为字符串格式
    ```
    """
    df = df[[code, entry, exit, kind]]
    df = df.fillna(int(datetime.datetime.now().strftime("%Y%m%d")))
    try:
        if type(df[entry].iloc[0]) == str:
            df[entry] = df[entry].astype(str)
            df[exit] = df[exit].astype(str)
        else:
            df[entry] = df[entry].astype(int).astype(str)
            df[exit] = df[exit].astype(int).astype(str)
    except Exception:
        print("您的进入日期和推出日期，既不是字符串，又不是数字格式，好好检查一下吧")
    df = df.set_index([code, kind])
    df = df.stack().to_frame(name="date")

    def fill_middle(df1):
        min_time = df1.date.min()
        max_time = df1.date.max()
        df2 = pd.DataFrame({"date": pd.date_range(min_time, max_time)})
        return df2

    ff = df.reset_index().groupby([code, kind]).apply(fill_middle)
    ff = ff.reset_index()
    ff = ff[[code, kind, "date"]]
    ff = ff[ff.date >= pd.Timestamp("2004-01-01")]
    return ff


@do_on_dfs
def set_index_first(df: pd.DataFrame) -> pd.DataFrame:
    """将dataframe的第一列，无论其是什么名字，都设置为index

    Parameters
    ----------
    df : pd.DataFrame
        要修改的dataframe
    Returns
    -------
    pd.DataFrame
        修改后的dataframe
    """
    df = df.set_index(list(df.columns)[0])
    return df


def merge_many(
    dfs: List[pd.DataFrame], names: list = None, how: str = "outer"
) -> pd.DataFrame:
    """将多个宽dataframe依据columns和index，拼接在一起，拼成一个长dataframe

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        将所有要拼接的宽表放在一个列表里
    names : list, optional
        拼接后，每一列宽表对应的名字, by default None
    how : str, optional
        拼接的方式, by default 'outer'

    Returns
    -------
    pd.DataFrame
        拼接后的dataframe
    """
    num = len(dfs)
    if names is None:
        names = [f"fac{i+1}" for i in range(num)]
    dfs = [i.stack().reset_index() for i in dfs]
    dfs = [i.rename(columns={list(i.columns)[-1]: j}) for i, j in zip(dfs, names)]
    dfs = [
        i.rename(columns={list(i.columns)[-2]: "code", list(i.columns)[0]: "date"})
        for i in dfs
    ]
    df = reduce(lambda x, y: pd.merge(x, y, on=["date", "code"], how=how), dfs)
    return df


class pure_dawn(object):
    """
    因子切割论的母框架，可以对两个因子进行类似于因子切割的操作
    可用于派生任何"以两个因子生成一个因子"的子类
    使用举例
    cut函数里，必须带有输入变量df,df有两个columns，一个名为'fac1'，一个名为'fac2'，df是最近一个回看期内的数据
    ```python
    class Cut(pure_dawn):

    def cut(self,df):
        df=df.sort_values('fac1')
        df=df.assign(fac3=df.fac1*df.fac2)
        ret0=df.fac2.iloc[:4].mean()
        ret1=df.fac2.iloc[4:8].mean()
        ret2=df.fac2.iloc[8:12].mean()
        ret3=df.fac2.iloc[12:16].mean()
        ret4=df.fac2.iloc[16:].mean()
        aret0=df.fac3.iloc[:4].mean()
        aret1=df.fac3.iloc[4:8].mean()
        aret2=df.fac3.iloc[8:12].mean()
        aret3=df.fac3.iloc[12:16].mean()
        aret4=df.fac3.iloc[16:].mean()
        return ret0,ret1,ret2,ret3,ret4,aret0,aret1,aret2,aret3,aret4

    cut=Cut(ct,ret_inday)
    cut.run(cut.cut)

    cut0=get_value(cut(),0)
    cut1=get_value(cut(),1)
    cut2=get_value(cut(),2)
    cut3=get_value(cut(),3)
    cut4=get_value(cut(),4)
    ```
    """

    def __init__(self, fac1: pd.DataFrame, fac2: pd.DataFrame, *args: list) -> None:
        """几个因子的操作，每个月操作一次

        Parameters
        ----------
        fac1 : pd.DataFrame
            因子值1，index为时间，columns为股票代码，values为因子值
        fac2 : pd.DataFrame
            因子2，index为时间，columns为股票代码，values为因子值
        """
        self.fac1 = fac1
        self.fac1 = self.fac1.stack().reset_index()
        self.fac1.columns = ["date", "code", "fac1"]
        self.fac2 = fac2
        self.fac2 = self.fac2.stack().reset_index()
        self.fac2.columns = ["date", "code", "fac2"]
        fac_all = pd.merge(self.fac1, self.fac2, on=["date", "code"])
        for i, fac in enumerate(args):
            fac = fac.stack().reset_index()
            fac.columns = ["date", "code", f"fac{i+3}"]
            fac_all = pd.merge(fac_all, fac, on=["date", "code"])
        fac_all = fac_all.sort_values(["date", "code"])
        self.fac = fac_all.copy()

    def __call__(self) -> pd.DataFrame:
        """返回最终月度因子值

        Returns
        -------
        `pd.DataFrame`
            最终因子值
        """
        return self.fac.copy()

    def get_fac_long_and_tradedays(self):
        """将两个因子的矩阵转化为长列表"""
        self.tradedays = sorted(list(set(self.fac.date)))

    def get_month_starts_and_ends(self, backsee=20):
        """计算出每个月回看期间的起点日和终点日"""
        self.month_ends = [
            i
            for i, j in zip(self.tradedays[:-1], self.tradedays[1:])
            if i.month != j.month
        ]
        self.month_ends.append(self.tradedays[-1])
        self.month_starts = [
            self.find_begin(self.tradedays, i, backsee=backsee) for i in self.month_ends
        ]
        self.month_starts[0] = self.tradedays[0]

    def find_begin(self, tradedays, end_day, backsee=20):
        """找出回看若干天的开始日，默认为20"""
        end_day_index = tradedays.index(end_day)
        start_day_index = end_day_index - backsee + 1
        start_day = tradedays[start_day_index]
        return start_day

    def make_monthly_factors_single_code(self, df, func, daily):
        """
        对单一股票来计算月度因子
        func为单月执行的函数，返回值应为月度因子，如一个float或一个list
        df为一个股票的四列表，包含时间、代码、因子1和因子2
        """
        res = {}
        if daily:
            ones = [self.find_begin(i) for i in self.tradedays[self.backsee - 1 :]]
            twos = self.tradedays[self.backsee - 1 :]
        else:
            ones = self.month_starts
            twos = self.month_ends
        for start, end in zip(ones, twos):
            this_month = df[(df.date >= start) & (df.date <= end)]
            res[end] = func(this_month)
        dates = list(res.keys())
        corrs = list(res.values())
        part = pd.DataFrame({"date": dates, "corr": corrs})
        return part

    @staticmethod
    def for_cross_via_zip(func):
        """返回值为多个pd.Series，每个pd.Series的index为股票代码，values为单个因子值
        例如
        ```python
        return (
                    pd.Series([1.54,8.77,9.99……],index=['000001.SZ','000002.SZ','000004.SZ'……]),
                    pd.Series([3.54,6.98,9.01……],index=['000001.SZ','000002.SZ','000004.SZ'……]),
                )
        ```
        上例中，每个股票一天返回两个因子值，每个pd.Series对应一个因子值
        """

        def full_run(df, *args, **kwargs):
            res = func(df, *args, **kwargs)
            if isinstance(res, pd.Series):
                return res
            else:
                res = pd.concat(res, axis=1)
                res.columns = [f"fac{i}" for i in range(len(res.columns))]
                res = res.assign(fac=list(zip(*[res[i] for i in list(res.columns)])))
                return res.fac

        return full_run

    def get_monthly_factor(
        self, func, whole_cross: bool = 0, daily: bool = 0, history_file: str = None
    ):
        """运行自己写的函数，获得月度因子"""
        if daily:
            iter_item = self.tradedays[self.backsee - 1 :]
        else:
            iter_item = self.month_ends
        res = []
        if history_file is not None:
            if os.path.exists(homeplace.update_data_file + history_file):
                old = pd.read_parquet(homeplace.update_data_file + history_file)
                old_date = old.index.max()
                if old_date == self.fac.date.max():
                    logger.info(f"本地文件已经是最新的了，无需计算")
                    self.fac = old
                else:
                    try:
                        new_date = self.find_begin(
                            self.tradedays, old_date, self.backsee
                        )
                        fac = self.fac[self.fac.date > new_date]
                        iter_item = [i for i in iter_item if i > new_date]
                        if whole_cross:
                            for end_date in tqdm.auto.tqdm(iter_item):
                                start_date = self.find_begin(
                                    self.tradedays, end_date, self.backsee
                                )
                                if start_date < end_date:
                                    df = self.fac[
                                        (self.fac.date >= start_date)
                                        & (self.fac.date <= end_date)
                                    ]
                                else:
                                    df = self.fac[self.fac.date <= end_date]
                                df = func(df)
                                df = df.to_frame().T
                                df.index = [end_date]
                                res.append(df)
                            fac = pd.concat(res).resample("M").last()
                            self.fac = pd.concat([old, fac])
                        else:
                            tqdm.auto.tqdm.pandas(
                                desc="when the dawn comes, tonight will be a memory too."
                            )
                            fac = fac.groupby(["code"]).progress_apply(
                                lambda x: self.make_monthly_factors_single_code(
                                    x, func, daily=daily
                                )
                            )
                            fac = (
                                fac.reset_index(level=1, drop=True)
                                .reset_index()
                                .set_index(["date", "code"])
                                .unstack()
                            )
                            fac.columns = [i[1] for i in list(fac.columns)]
                            fac = fac.resample("M").last()
                            self.fac = pd.concat([old, fac])
                        self.fac.to_parquet(homeplace.update_data_file + history_file)
                        logger.success(f"本地文件已经更新完成")
                    except Exception:
                        logger.info(f"本地文件已经是最新的了，无需计算")
            else:
                logger.info("第一次计算，请耐心等待……")
                if whole_cross:
                    for end_date in tqdm.auto.tqdm(iter_item):
                        start_date = self.find_begin(
                            self.tradedays, end_date, self.backsee
                        )
                        if start_date < end_date:
                            df = self.fac[
                                (self.fac.date >= start_date)
                                & (self.fac.date <= end_date)
                            ]
                        else:
                            df = self.fac[self.fac.date <= end_date]
                        df = func(df)
                        df = df.to_frame().T
                        df.index = [end_date]
                        res.append(df)
                    self.fac = pd.concat(res).resample("M").last()
                else:
                    tqdm.auto.tqdm.pandas(
                        desc="when the dawn comes, tonight will be a memory too."
                    )
                    self.fac = self.fac.groupby(["code"]).progress_apply(
                        lambda x: self.make_monthly_factors_single_code(
                            x, func, daily=daily
                        )
                    )
                    self.fac = (
                        self.fac.reset_index(level=1, drop=True)
                        .reset_index()
                        .set_index(["date", "code"])
                        .unstack()
                    )
                    self.fac.columns = [i[1] for i in list(self.fac.columns)]
                    self.fac = self.fac.resample("M").last()
                self.fac.to_parquet(homeplace.update_data_file + history_file)
                logger.success(f"本地文件已经写入完成")
        else:
            logger.warning(
                "您本次计算没有指定任何本地文件路径，这很可能会导致大量的重复计算和不必要的时间浪费，请注意！"
            )
            if daily:
                logger.warning(
                    "您指定的是日频计算，非月频计算，因此强烈建议您指定history_file参数！！"
                )
            if whole_cross:
                for end_date in tqdm.auto.tqdm(iter_item):
                    start_date = self.find_begin(self.tradedays, end_date, self.backsee)
                    if start_date < end_date:
                        df = self.fac[
                            (self.fac.date >= start_date) & (self.fac.date <= end_date)
                        ]
                    else:
                        df = self.fac[self.fac.date <= end_date]
                    df = func(df)
                    df = df.to_frame().T
                    df.index = [end_date]
                    res.append(df)
                self.fac = pd.concat(res).resample("M").last()
            else:
                tqdm.auto.tqdm.pandas(
                    desc="when the dawn comes, tonight will be a memory too."
                )
                self.fac = self.fac.groupby(["code"]).progress_apply(
                    lambda x: self.make_monthly_factors_single_code(
                        x, func, daily=daily
                    )
                )
                self.fac = (
                    self.fac.reset_index(level=1, drop=True)
                    .reset_index()
                    .set_index(["date", "code"])
                    .unstack()
                )
                self.fac.columns = [i[1] for i in list(self.fac.columns)]
                self.fac = self.fac.resample("M").last()

    @kk.desktop_sender(title="嘿，切割完成啦🛁")
    def run(
        self,
        func: Callable,
        backsee: int = 20,
        whole_cross: bool = 0,
        daily: bool = 0,
        history_file: str = None,
    ) -> None:
        """执行计算的框架，产生因子值

        Parameters
        ----------
        func : Callable
            每个月要进行的操作
        backsee : int, optional
            回看期，即每个月月底对过去多少天进行计算, by default 20
        whole_cross : bool, optional
            是否同时取横截面上所有股票进行计算, by default 20
        daily : bool, optional
            是否每日计算, by default 20
        history_file : str, optional
            存储历史数据的文件名, by default None
        """
        self.backsee = backsee
        self.get_fac_long_and_tradedays()
        self.get_month_starts_and_ends(backsee=backsee)
        self.get_monthly_factor(
            func, whole_cross=whole_cross, daily=daily, history_file=history_file
        )


def corr_two_daily(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    history: str = None,
    rolling_window: int = 20,
    n_jobs: int = 1,
    daily: bool = 1,
    method: str = "pearson",
) -> pd.DataFrame:
    """求两个因子，在相同股票上，时序上滚动窗口下的相关系数

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个因子，index为时间，columns为股票代码
    df2 : pd.DataFrame
        第二个因子，index为时间，columns为股票代码
    history : str, optional
        从某处读取计算好的历史文件
    rolling_window : int, optional
        滚动窗口, by default 20
    n_jobs : int, optional
        并行数量, by default 1
    daily : bool, optional
        是否每天计算, by default 1
    method : str, optional
        使用哪种方法计算相关系数, by default 'pearson'

    Returns
    -------
    pd.DataFrame
        相关系数后的结果，index为时间，columns为股票代码
    """
    if daily:
        if method == "pearson":

            def corr_in(a, b, c):
                return c.iloc[-1], np.corrcoef(a, b)[0, 1]

            return func_two_daily(
                df1=df1,
                df2=df2,
                func=corr_in,
                history=history,
                rolling_window=rolling_window,
                n_jobs=n_jobs,
            )
        elif method == "spearman":

            def corr_in(a, b, c):
                return c.iloc[-1], np.corrcoef(np.argsort(a), np.argsort(b))[0, 1]

            return func_two_daily(
                df1=df1,
                df2=df2,
                func=corr_in,
                history=history,
                rolling_window=rolling_window,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError("您输入的方法暂不支持")
    else:
        if method == "pearson":

            class Cut(pure_dawn):
                def cut(self, df: pd.DataFrame):
                    return df[["fac1", "fac2"]].corr().iloc[0, 1]

            cut = Cut(df1, df2)
            cut.run(cut.cut, backsee=rolling_window, history_file=history)
            return cut()
        elif method == "spearman":

            class Cut(pure_dawn):
                def cut(self, df: pd.DataFrame):
                    return df[["fac1", "fac2"]].rank().corr().iloc[0, 1]

            cut = Cut(df1, df2)
            cut.run(cut.cut, backsee=rolling_window, history_file=history)
            return cut()
        else:
            raise ValueError("您输入的方法暂不支持")


def cov_two_daily(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    history: str = None,
    rolling_window: int = 20,
    n_jobs: int = 1,
    daily: bool = 1,
) -> pd.DataFrame:
    """求两个因子，在相同股票上，时序上滚动窗口下的协方差

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个因子，index为时间，columns为股票代码
    df2 : pd.DataFrame
        第二个因子，index为时间，columns为股票代码
    history : str, optional
        从某处读取计算好的历史文件
    rolling_window : int, optional
        滚动窗口, by default 20
    n_jobs : int, optional
        并行数量, by default 1
    daily : bool, optional
        是否每天计算, by default 1

    Returns
    -------
    pd.DataFrame
        求协方差后的结果，index为时间，columns为股票代码
    """
    if daily:

        def cov_in(a, b, c):
            return c.iloc[-1], np.cov(a, b)[0, 1]

        return func_two_daily(
            df1=df1,
            df2=df2,
            func=cov_in,
            history=history,
            rolling_window=rolling_window,
            n_jobs=n_jobs,
        )
    else:

        class Cut(pure_dawn):
            def cut(self, df: pd.DataFrame):
                return df[["fac1", "fac2"]].cov().iloc[0, 1]

        cut = Cut(df1, df2)
        cut.run(cut.cut, backsee=rolling_window, history_file=history)
        return cut()


def func_two_daily(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    func: Callable,
    history: str = None,
    rolling_window: int = 20,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """求两个因子，在相同股票上，时序上滚动窗口下的相关系数

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个因子，index为时间，columns为股票代码
    df2 : pd.DataFrame
        第二个因子，index为时间，columns为股票代码
    func : Callable
        要对两列数进行操作的函数
    history : str, optional
        从某处读取计算好的历史文件
    rolling_window : int, optional
        滚动窗口, by default 20
    n_jobs : int, optional
        并行数量, by default 1

    Returns
    -------
    pd.DataFrame
        计算后的结果，index为时间，columns为股票代码
    """

    the_func = partial(func)

    def func_rolling(df):
        df = df.sort_values(["date"])
        if df.shape[0] > rolling_window:
            df = npext.rolling_apply(
                the_func, rolling_window, df.fac1, df.fac2, df.date, n_jobs=n_jobs
            )
            return df

    homeplace = HomePlace()
    if history is not None:
        if os.path.exists(homeplace.update_data_file + history):
            old = pd.read_parquet(homeplace.update_data_file + history)
            new_end = min(df1.index.max(), df2.index.max())
            if new_end > old.index.max():
                old_end = datetime.datetime.strftime(old.index.max(), "%Y%m%d")
                logger.info(f"上次更新到了{old_end}")
                df1a = df1[df1.index <= old.index.max()].tail(rolling_window - 1)
                df1b = df1[df1.index > old.index.max()]
                df1 = pd.concat([df1a, df1b])
                df2a = df2[df2.index <= old.index.max()].tail(rolling_window - 1)
                df2b = df2[df2.index > old.index.max()]
                df2 = pd.concat([df2a, df2b])
                twins = merge_many([df1, df2])

                tqdm.auto.tqdm.pandas()
                corrs = twins.groupby(["code"]).progress_apply(func_rolling)
                cor = []
                for i in range(len(corrs)):
                    df = (
                        pd.DataFrame(corrs.iloc[i]).dropna().assign(code=corrs.index[i])
                    )
                    cor.append(df)
                cors = pd.concat(cor)
                cors.columns = ["date", "corr", "code"]
                cors = cors.pivot(index="date", columns="code", values="corr")
                if history is not None:
                    if os.path.exists(homeplace.update_data_file + history):
                        cors = pd.concat([old, cors])
                    cors = drop_duplicates_index(cors)
                    cors.to_parquet(homeplace.update_data_file + history)
                    new_end = datetime.datetime.strftime(cors.index.max(), "%Y%m%d")
                    logger.info(f"已经更新至{new_end}")
                return cors
            else:
                logger.info(f"已经是最新的了")
                return old
        else:
            logger.info("第一次计算，请耐心等待，计算完成后将存储")
            twins = merge_many([df1, df2])
            tqdm.auto.tqdm.pandas()
            corrs = twins.groupby(["code"]).progress_apply(func_rolling)
            cor = []
            for i in range(len(corrs)):
                df = pd.DataFrame(corrs.iloc[i]).dropna().assign(code=corrs.index[i])
                cor.append(df)
            cors = pd.concat(cor)
            cors.columns = ["date", "corr", "code"]
            cors = cors.pivot(index="date", columns="code", values="corr")
            if history is not None:
                if os.path.exists(homeplace.update_data_file + history):
                    cors = pd.concat([old, cors])
                cors = drop_duplicates_index(cors)
                cors.to_parquet(homeplace.update_data_file + history)
                new_end = datetime.datetime.strftime(cors.index.max(), "%Y%m%d")
                logger.info(f"已经更新至{new_end}")
            return cors
    else:
        logger.warning(
            "您本次计算没有指定任何本地文件路径，这很可能会导致大量的重复计算和不必要的时间浪费，请注意！"
        )
        twins = merge_many([df1, df2])
        tqdm.auto.tqdm.pandas()
        corrs = twins.groupby(["code"]).progress_apply(func_rolling)
        cor = []
        for i in range(len(corrs)):
            df = pd.DataFrame(corrs.iloc[i]).dropna().assign(code=corrs.index[i])
            cor.append(df)
        cors = pd.concat(cor)
        cors.columns = ["date", "corr", "code"]
        cors = cors.pivot(index="date", columns="code", values="corr")
        return cors


@do_on_dfs
def drop_duplicates_index(new: pd.DataFrame) -> pd.DataFrame:
    """对dataframe依照其index进行去重，并保留最上面的行

    Parameters
    ----------
    new : pd.DataFrame
        要去重的dataframe

    Returns
    -------
    pd.DataFrame
        去重后的dataframe
    """
    pri_name = new.index.name
    new = new.reset_index()
    new = new.rename(
        columns={
            list(new.columns)[0]: "tmp_name_for_this_function_never_same_to_others"
        }
    )
    new = new.drop_duplicates(
        subset=["tmp_name_for_this_function_never_same_to_others"], keep="first"
    )
    new = new.set_index("tmp_name_for_this_function_never_same_to_others")
    if pri_name == "tmp_name_for_this_function_never_same_to_others":
        new.index.name = "date"
    else:
        new.index.name = pri_name
    return new


def select_max(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """两个columns与index完全相同的df，每个值都挑出较大值

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个df
    df2 : pd.DataFrame
        第二个df

    Returns
    -------
    `pd.DataFrame`
        两个df每个value中的较大者
    """
    return (df1 + df2 + np.abs(df1 - df2)) / 2


def select_min(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """两个columns与index完全相同的df，每个值都挑出较小值

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个df
    df2 : pd.DataFrame
        第二个df

    Returns
    -------
    `pd.DataFrame`
        两个df每个value中的较小者
    """
    return (df1 + df2 - np.abs(df1 - df2)) / 2


@do_on_dfs
def debj(df: pd.DataFrame) -> pd.DataFrame:
    """去除因子中的北交所数据

    Parameters
    ----------
    df : pd.DataFrame
        包含北交所的因子dataframe，index是时间，columns是股票代码

    Returns
    -------
    pd.DataFrame
        去除北交所股票的因子dataframe
    """
    df = df[[i for i in list(df.columns) if i[0] in ["0", "3", "6"]]]
    return df


@do_on_dfs
def standardlize(df: pd.DataFrame, all_pos: bool = 0) -> pd.DataFrame:
    """对因子dataframe做横截面z-score标准化

    Parameters
    ----------
    df : pd.DataFrame
        要做中性化的因子值，index是时间，columns是股票代码
    all_pos : bool, optional
        是否要将值都变成正数，通过减去截面的最小值实现, by default 0

    Returns
    -------
    pd.DataFrame
        标准化之后的因子
    """
    df = ((df.T - df.T.mean()) / df.T.std()).T
    if all_pos:
        df = (df.T - df.T.min()).T
    return df


@do_on_dfs
def count_value(df: pd.DataFrame, with_zero: bool = 0) -> int:
    """计算dataframe中总共有多少（非0）非空的值

    Parameters
    ----------
    df : pd.DataFrame
        要检测的dataframe
    with_zero : bool, optional
        统计数量时，是否也把值为0的数据统计进去, by default 0

    Returns
    -------
    int
        （非0）非空的数据的个数
    """
    y = np.sign(np.abs(df))
    if with_zero:
        y = np.sign(y + 1)
    return y.sum().sum()


@do_on_dfs
def detect_nan(df: pd.DataFrame) -> bool:
    """检查一个pd.DataFrame中是否存在空值

    Parameters
    ----------
    df : pd.DataFrame
        待检查的pd.DataFrame

    Returns
    -------
    `bool`
        检查结果，有空值为True，否则为False
    """
    x = df.isna() + 0
    if x.sum().sum():
        print("存在空值")
        return True
    else:
        print("不存在空值")
        return False


@do_on_dfs
def get_abs(df: pd.DataFrame, quantile: float = None, square: bool = 0) -> pd.DataFrame:
    """均值距离化：计算因子与截面均值的距离

    Parameters
    ----------
    df : pd.DataFrame
        未均值距离化的因子，index为时间，columns为股票代码
    quantile : bool, optional
        为1则计算到某个分位点的距离, by default None
    square : bool, optional
        为1则计算距离的平方, by default 0

    Returns
    -------
    `pd.DataFrame`
        均值距离化之后的因子值
    """
    if not square:
        if quantile is not None:
            return np.abs((df.T - df.T.quantile(quantile)).T)
        else:
            return np.abs((df.T - df.T.mean()).T)
    else:
        if quantile is not None:
            return ((df.T - df.T.quantile(quantile)).T) ** 2
        else:
            return ((df.T - df.T.mean()).T) ** 2


@do_on_dfs
def get_normal(df: pd.DataFrame) -> pd.DataFrame:
    """将因子横截面正态化

    Parameters
    ----------
    df : pd.DataFrame
        原始因子，index是时间，columns是股票代码

    Returns
    -------
    `pd.DataFrame`
        每个横截面都呈现正态分布的因子
    """
    df = df.replace(0, np.nan)
    df = df.T.apply(lambda x: ss.boxcox(x)[0]).T
    return df


@do_on_dfs
def coin_reverse(
    ret20: pd.DataFrame, vol20: pd.DataFrame, mean: bool = 1, positive_negtive: bool = 0
) -> pd.DataFrame:
    """球队硬币法：根据vol20的大小，翻转一半ret20，把vol20较大的部分，给ret20添加负号

    Parameters
    ----------
    ret20 : pd.DataFrame
        要被翻转的因子，index是时间，columns是股票代码
    vol20 : pd.DataFrame
        翻转的依据，index是时间，columns是股票代码
    mean : bool, optional
        为1则以是否大于截面均值为标准翻转，否则以是否大于截面中位数为标准, by default 1
    positive_negtive : bool, optional
        是否截面上正负值的两部分，各翻转一半, by default 0

    Returns
    -------
    `pd.DataFrame`
        翻转后的因子值
    """
    if positive_negtive:
        if not mean:
            down20 = np.sign(ret20)
            down20 = down20.replace(1, np.nan)
            down20 = down20.replace(-1, 1)

            vol20_down = down20 * vol20
            vol20_down = (vol20_down.T - vol20_down.T.median()).T
            vol20_down = np.sign(vol20_down)
            ret20_down = ret20[ret20 < 0]
            ret20_down = vol20_down * ret20_down

            up20 = np.sign(ret20)
            up20 = up20.replace(-1, np.nan)

            vol20_up = up20 * vol20
            vol20_up = (vol20_up.T - vol20_up.T.median()).T
            vol20_up = np.sign(vol20_up)
            ret20_up = ret20[ret20 > 0]
            ret20_up = vol20_up * ret20_up

            ret20_up = ret20_up.replace(np.nan, 0)
            ret20_down = ret20_down.replace(np.nan, 0)
            new_ret20 = ret20_up + ret20_down
            new_ret20_tr = new_ret20.replace(0, np.nan)
            return new_ret20_tr
        else:
            down20 = np.sign(ret20)
            down20 = down20.replace(1, np.nan)
            down20 = down20.replace(-1, 1)

            vol20_down = down20 * vol20
            vol20_down = (vol20_down.T - vol20_down.T.mean()).T
            vol20_down = np.sign(vol20_down)
            ret20_down = ret20[ret20 < 0]
            ret20_down = vol20_down * ret20_down

            up20 = np.sign(ret20)
            up20 = up20.replace(-1, np.nan)

            vol20_up = up20 * vol20
            vol20_up = (vol20_up.T - vol20_up.T.mean()).T
            vol20_up = np.sign(vol20_up)
            ret20_up = ret20[ret20 > 0]
            ret20_up = vol20_up * ret20_up

            ret20_up = ret20_up.replace(np.nan, 0)
            ret20_down = ret20_down.replace(np.nan, 0)
            new_ret20 = ret20_up + ret20_down
            new_ret20_tr = new_ret20.replace(0, np.nan)
            return new_ret20_tr
    else:
        if not mean:
            vol20_dummy = np.sign((vol20.T - vol20.T.median()).T)
            ret20 = ret20 * vol20_dummy
            return ret20
        else:
            vol20_dummy = np.sign((vol20.T - vol20.T.mean()).T)
            ret20 = ret20 * vol20_dummy
            return ret20


@do_on_dfs
def multidfs_to_one(*args: list) -> pd.DataFrame:
    """很多个df，各有一部分，其余位置都是空，
    想把各自df有值的部分保留，都没有值的部分继续设为空

    Returns
    -------
    `pd.DataFrame`
        合并后的df
    """
    dfs = [i.fillna(0) for i in args]
    background = np.sign(np.abs(np.sign(sum(dfs))) + 1).replace(1, 0)
    dfs = [(i + background).fillna(0) for i in dfs]
    df_nans = [i.isna() for i in dfs]
    nan = reduce(lambda x, y: x * y, df_nans)
    nan = nan.replace(1, np.nan)
    nan = nan.replace(0, 1)
    df_final = sum(dfs) * nan
    return df_final


@do_on_dfs
def to_percent(x: float) -> Union[float, str]:
    """把小数转化为2位小数的百分数

    Parameters
    ----------
    x : float
        要转换的小数

    Returns
    -------
    Union[float,str]
        空值则依然为空，否则返回带%的字符串
    """
    if np.isnan(x):
        return x
    else:
        x = str(round(x * 100, 2)) + "%"
        return x


@do_on_dfs
def calc_exp_list(window: int, half_life: int) -> np.ndarray:
    """生成半衰序列

    Parameters
    ----------
    window : int
        窗口期
    half_life : int
        半衰期

    Returns
    -------
    `np.ndarray`
        半衰序列
    """
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)


@do_on_dfs
def calcWeightedStd(series: pd.Series, weights: Union[pd.Series, np.ndarray]) -> float:
    """计算半衰加权标准差

    Parameters
    ----------
    series : pd.Series
        目标序列
    weights : Union[pd.Series,np.ndarray]
        权重序列

    Returns
    -------
    `float`
        半衰加权标准差
    """
    weights /= np.sum(weights)
    return np.sqrt(np.sum((series - np.mean(series)) ** 2 * weights))


def get_list_std(delta_sts: List[pd.DataFrame]) -> pd.DataFrame:
    """同一天多个因子，计算这些因子在当天的标准差

    Parameters
    ----------
    delta_sts : List[pd.DataFrame]
        多个因子构成的list，每个因子index为时间，columns为股票代码

    Returns
    -------
    `pd.DataFrame`
        每天每只股票多个因子的标准差
    """
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum(delta_sts_std)
    delta_sts_std = delta_sts_std**0.5 / len(delta_sts) ** 0.5
    return delta_sts_std


def get_list_std_weighted(delta_sts: List[pd.DataFrame], weights: list) -> pd.DataFrame:
    """对多个df对应位置上的值求加权标准差

    Parameters
    ----------
    delta_sts : List[pd.DataFrame]
        多个dataframe
    weights : list
        权重序列

    Returns
    -------
    pd.DataFrame
        标准差序列
    """
    weights = [i / sum(weights) for i in weights]
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum([i * j for i, j in zip(delta_sts_std, weights)])
    return delta_sts_std**0.5


@do_on_dfs
def to_group(df: pd.DataFrame, group: int = 10) -> pd.DataFrame:
    """把一个index为时间，code为时间的df，每个截面上的值，按照排序分为group组，将值改为组号，从0开始

    Parameters
    ----------
    df : pd.DataFrame
        要改为组号的df
    group : int, optional
        分为多少组, by default 10

    Returns
    -------
    pd.DataFrame
        组号组成的dataframe
    """
    df = df.T.apply(lambda x: pd.qcut(x, group, labels=False, duplicates="drop")).T
    return df


def same_columns(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """保留多个dataframe共同columns的部分

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        多个dataframe

    Returns
    -------
    List[pd.DataFrame]
        保留共同部分后的结果
    """
    dfs = [i.T for i in dfs]
    res = []
    for i, df in enumerate(dfs):
        others = dfs[:i] + dfs[i + 1 :]

        for other in others:
            df = df[df.index.isin(other.index)]
        res.append(df.T)
    return res


def same_index(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """保留多个dataframe共同index的部分

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        多个dataframe

    Returns
    -------
    List[pd.DataFrame]
        保留共同部分后的结果
    """
    res = []
    for i, df in enumerate(dfs):
        others = dfs[:i] + dfs[i + 1 :]

        for other in others:
            df = df[df.index.isin(other.index)]
        res.append(df)
    return res


@do_on_dfs
def feather_to_parquet(folder: str):
    """将某个路径下的所有feather文件都转化为parquet文件

    Parameters
    ----------
    folder : str
        要转化的文件夹路径
    """
    files = os.listdir(folder)
    files = [folder + i for i in files]
    for file in tqdm.auto.tqdm(files):
        try:
            df = pd.read_feather(file)
            if (("date" in list(df.columns)) and ("code" not in list(df.columns))) or (
                "index" in list(df.columns)
            ):
                df = df.set_index(list(df.columns)[0])
            df.to_parquet(file.split(".")[0] + ".parquet")
        except Exception:
            logger.warning(f"{file}不是parquet文件")


def feather_to_parquet_all():
    """将数据库中所有的feather文件都转化为parquet文件"""
    homeplace = HomePlace()
    feather_to_parquet(homeplace.daily_data_file)
    feather_to_parquet(homeplace.barra_data_file)
    feather_to_parquet(homeplace.final_factor_file)
    feather_to_parquet(homeplace.update_data_file)
    feather_to_parquet(homeplace.factor_data_file)
    logger.success(
        "数据库中的feather文件全部被转化为了parquet文件，您可以手动删除所有的feather文件了"
    )


def zip_many_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """将多个dataframe，拼在一起，相同index和columns指向的那个values，变为多个dataframe的值的列表
    通常用于存储整合分钟数据计算的因子值

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        多个dataframe，每一个的values都是float形式

    Returns
    -------
    pd.DataFrame
        整合后的dataframe，每一个values都是list的形式
    """
    df = merge_many(dfs)
    cols = [df[f"fac{i}"] for i in range(1, len(dfs) + 1)]
    df = df.assign(fac=pd.Series(zip(*cols)))
    df = df.pivot(index="date", columns="code", values="fac")
    return df


def get_values(df: pd.DataFrame) -> List[pd.DataFrame]:
    """从一个values为列表的dataframe中，一次性取出所有值，分别设置为一个dataframe，并依照顺序存储在列表中

    Parameters
    ----------
    df : pd.DataFrame
        一个values为list的dataframe

    Returns
    -------
    List[pd.DataFrame]
        多个dataframe，每一个的values都是float形式
    """
    d = df.dropna(how="all", axis=1)
    d = d.iloc[:, 0].dropna()
    num = len(d.iloc[0])
    facs = list(map(lambda x: get_value(df, x), range(num)))
    return facs


@do_on_dfs
def get_fac_via_corr(
    df: pd.DataFrame,
    history_file: str = None,
    backsee: int = 20,
    fillna_method: Union[float, str] = "ffill",
    corr_method: str = "pearson",
    daily: bool = 0,
    abs: bool = 0,
    riskmetrics: bool = 0,
    riskmetrics_lambda: float = 0.94,
) -> pd.DataFrame:
    """对一个日频因子，对其滚动时间窗口进行因子月度化计算。
    具体操作为每天（或每月月底）计算过去20天因子值的相关性矩阵，
    然后对每个股票的所有相关系数求均值

    Parameters
    ----------
    df : pd.DataFrame
        日频因子，index为时间，columns为股票代码，values为时间
    history_file : str, optional
        用于存储历史数据的本地文件, by default None
    backsee : int, optional
        滚动窗口长度, by default 20
    fillna_method : Union[float, str], optional
        由于存在缺失值时，相关性矩阵的计算存在问题，因此这里对其进行补全，可选择补全方式，输入`‘ffill'`或`'bfill'`即为取前取后填充,输入数字则为用固定数字填充 by default "ffill"
    corr_method : str, optional
        求相关性的方法，可以指定`'pearson'`、`'spearman'`、`'kendall'`, default 'pearson'
    daily : bool, optional
        是否每天滚动, by default 0
    abs : bool, optional
        是否要对相关系数矩阵取绝对值, by default 0
    riskmetrics : bool, optional
        使用RiskMetrics方法，对相关性进行调整，增加临近交易日的权重, by default 0
    riskmetrics_lambda : float, optional
        使用RiskMetrics方法时的lambda参数, by default 0.94

    Returns
    -------
    pd.DataFrame
        月度化后的因子值
    """
    homeplace = HomePlace()
    if history_file is not None:
        if os.path.exists(homeplace.update_data_file + history_file):
            old = pd.read_parquet(homeplace.update_data_file + history_file)
        else:
            old = None
            logger.info("这一结果是新的，将从头计算")
    else:
        old = None
    if old is not None:
        old_end = old.index.max()
        pastpart = df[df.index <= old_end]
        old_tail = pastpart.tail(backsee - 1)
        old_end_str = datetime.datetime.strftime(old_end, "%Y%m%d")
        logger.info(f"上次计算到了{old_end_str}")
        df = df[df.index > old_end]
        if df.shape[0] > 0:
            df = pd.concat([old_tail, df])
            ends = list(df.index)
            ends = pd.Series(ends, index=ends)
            ends = ends.resample("M").last()
            ends = list(ends)
            if daily:
                iters = list(df.index)
            else:
                iters = ends
            dfs = []
            for end in tqdm.auto.tqdm(iters):
                if isinstance(fillna_method, float):
                    df0 = (
                        df.loc[:end]
                        .tail(backsee)
                        .dropna(how="all", axis=1)
                        .fillna(fillna_method)
                        .dropna(axis=1)
                    )
                else:
                    df0 = (
                        df.loc[:end]
                        .tail(backsee)
                        .dropna(how="all", axis=1)
                        .fillna(method=fillna_method)
                        .dropna(axis=1)
                    )
                if riskmetrics:
                    df0 = (
                        (df0 - df0.mean()).T
                        * (
                            pd.Series(
                                [
                                    riskmetrics_lambda ** (backsee - i)
                                    for i in range(df0.shape[0])
                                ],
                                index=df0.index,
                            )
                            ** 0.5
                        )
                    ).T
                if corr_method == "spearman":
                    corr = df0.rank().corr()
                else:
                    corr = df0.corr(method=corr_method)
                if abs:
                    corr = corr.abs()
                df0 = corr.mean().to_frame(end)
                dfs.append(df0)
            dfs = pd.concat(dfs, axis=1).T
            dfs = drop_duplicates_index(pd.concat([old, dfs]))
            dfs.to_parquet(homeplace.update_data_file + history_file)
            if daily:
                return dfs
            else:
                return dfs.resample("M").last()
        else:
            logger.info("已经是最新的了")
            return old
    else:
        ends = list(df.index)
        ends = pd.Series(ends, index=ends)
        ends = ends.resample("M").last()
        ends = list(ends)
        if daily:
            iters = list(df.index)
        else:
            iters = ends
        dfs = []
        for end in tqdm.auto.tqdm(iters):
            if isinstance(fillna_method, float):
                df0 = (
                    df.loc[:end]
                    .tail(backsee)
                    .dropna(how="all", axis=1)
                    .fillna(fillna_method)
                    .dropna(axis=1)
                )
            else:
                df0 = (
                    df.loc[:end]
                    .tail(backsee)
                    .dropna(how="all", axis=1)
                    .fillna(method=fillna_method)
                    .dropna(axis=1)
                )
            if riskmetrics:
                df0 = (
                    (df0 - df0.mean()).T
                    * (
                        pd.Series(
                            [
                                riskmetrics_lambda ** (backsee - i)
                                for i in range(df0.shape[0])
                            ],
                            index=df0.index,
                        )
                        ** 0.5
                    )
                ).T
            if corr_method == "spearman":
                corr = df0.rank().corr()
            else:
                corr = df0.corr(method=corr_method)
            if abs:
                corr = corr.abs()
            df0 = corr.mean().to_frame(end)
            dfs.append(df0)
        dfs = pd.concat(dfs, axis=1).T
        if history_file is not None:
            dfs.to_parquet(homeplace.update_data_file + history_file)
        if daily:
            return dfs
        else:
            return dfs.resample("M").last()


@do_on_dfs
def get_fac_cross_via_func(
    df: pd.DataFrame,
    func: Callable,
    history_file: str = None,
    backsee: int = 20,
    fillna_method: Union[float, str, None] = "ffill",
    daily: bool = 0,
) -> pd.DataFrame:
    """对一个日频因子，对其滚动时间窗口进行因子月度化计算。
    具体操作为每天（或每月月底）截取过去一段窗口，并进行某个自定义的操作，

    Parameters
    ----------
    df : pd.DataFrame
        日频因子，index为时间，columns为股票代码，values为时间
    func : Callable
        自定义的操作函数，需要对一个窗口时间内的面板数据进行处理，最终要返回一个series，index为股票代码，values为月度化的因子值，name无所谓
    history_file : str, optional
        用于存储历史数据的本地文件, by default None
    backsee : int, optional
        滚动窗口长度, by default 20
    fillna_method : Union[float, str], optional
        对缺失值进行补全，可选择补全方式，输入`'ffill'`或`'bfill'`即为取前取后填充；输入数字则为用固定数字填充；输入None则不填充缺失值 by default "ffill"
    daily : bool, optional
        是否每天滚动, by default 0

    Returns
    -------
    pd.DataFrame
        月度化后的因子值
    """
    homeplace = HomePlace()
    if history_file is not None:
        if os.path.exists(homeplace.update_data_file + history_file):
            old = pd.read_parquet(homeplace.update_data_file + history_file)
        else:
            old = None
            logger.info("这一结果是新的，将从头计算")
    else:
        old = None
    if old is not None:
        old_end = old.index.max()
        pastpart = df[df.index <= old_end]
        old_tail = pastpart.tail(backsee - 1)
        old_end_str = datetime.datetime.strftime(old_end, "%Y%m%d")
        logger.info(f"上次计算到了{old_end_str}")
        df = df[df.index > old_end]
        if df.shape[0] > 0:
            df = pd.concat([old_tail, df])
            ends = list(df.index)
            ends = pd.Series(ends, index=ends)
            ends = ends.resample("M").last()
            ends = list(ends)
            if daily:
                iters = list(df.index)
            else:
                iters = ends
            dfs = []
            for end in tqdm.auto.tqdm(iters):
                if isinstance(fillna_method, float):
                    df0 = (
                        df.loc[:end]
                        .tail(backsee)
                        .dropna(how="all", axis=1)
                        .fillna(fillna_method)
                        .dropna(axis=1)
                    )
                elif isinstance(fillna_method, str):
                    df0 = (
                        df.loc[:end]
                        .tail(backsee)
                        .dropna(how="all", axis=1)
                        .fillna(method=fillna_method)
                        .dropna(axis=1)
                    )
                else:
                    df0 = df.loc[:end].tail(backsee).dropna(how="all", axis=1)
                corr = func(df0).to_frame(end)
                dfs.append(corr)
            dfs = pd.concat(dfs, axis=1).T
            dfs = drop_duplicates_index(pd.concat([old, dfs]))
            dfs.to_parquet(homeplace.update_data_file + history_file)
            return dfs
        else:
            logger.info("已经是最新的了")
            return old
    else:
        ends = list(df.index)
        ends = pd.Series(ends, index=ends)
        ends = ends.resample("M").last()
        ends = list(ends)
        if daily:
            iters = list(df.index)
        else:
            iters = ends
        dfs = []
        for end in tqdm.auto.tqdm(iters):
            if isinstance(fillna_method, float):
                df0 = (
                    df.loc[:end]
                    .tail(backsee)
                    .dropna(how="all", axis=1)
                    .fillna(fillna_method)
                    .dropna(axis=1)
                )
            elif isinstance(fillna_method, str):
                df0 = (
                    df.loc[:end]
                    .tail(backsee)
                    .dropna(how="all", axis=1)
                    .fillna(method=fillna_method)
                    .dropna(axis=1)
                )
            else:
                df0 = df.loc[:end].tail(backsee).dropna(how="all", axis=1)
            corr = func(df0).to_frame(end)
            dfs.append(corr)
        dfs = pd.concat(dfs, axis=1).T
        if history_file is not None:
            dfs.to_parquet(homeplace.update_data_file + history_file)
        return dfs


@do_on_dfs
def 计算连续期数(ret0: pd.Series, point: float = 0) -> pd.Series:
    """计算一列数，持续大于或持续小于某个临界点的期数

    Parameters
    ----------
    ret0 : pd.Series
        收益率序列、或者某个指标的序列
    point : float, optional
        临界值, by default 0

    Returns
    -------
    pd.Series
        持续大于或小于的期数
    """
    ret = ret0.copy()
    ret = ((ret >= point) + 0).replace(0, -1)
    ret = ret.to_frame("signal").assign(num=range(ret.shape[0]))
    ret.signal = ret.signal.diff().shift(-1)
    ret1 = ret[ret.signal != 0]
    ret1 = ret1.assign(duration=ret1.num.diff())
    ret = pd.concat([ret, ret1[["duration"]]], axis=1)
    ret.signal = ret.signal.diff()
    ret2 = ret[ret.signal.abs() == 2]
    ret2 = ret2.assign(add_duration=1)
    ret = pd.concat([ret, ret2[["add_duration"]]], axis=1)
    ret.duration = ret.duration.fillna(0)
    ret.add_duration = ret.add_duration.fillna(0)
    ret.duration = select_max(ret.duration, ret.add_duration)
    ret.duration = ret.duration.replace(0, np.nan).interpolate()
    return ret.duration


@do_on_dfs
def all_pos(df: pd.DataFrame) -> pd.DataFrame:
    """将因子值每个截面上减去最小值，从而都变成非负数

    Parameters
    ----------
    df : pd.DataFrame
        因子值，index为时间，columns为股票代码，values为因子值

    Returns
    -------
    pd.DataFrame
        变化后非负的因子值
    """
    return (df.T - df.T.min()).T


@do_on_dfs
def clip_mad(
    df: pd.DataFrame, n: float = 5, replace: bool = 1, keep_trend: bool = 1
) -> pd.DataFrame:
    if keep_trend:
        df = df.stack().reset_index()
        df.columns = ["date", "code", "fac"]

        def clip_sing(x: pd.DataFrame, n: float = 3):
            median = x.fac.quantile(0.5)
            diff_median = ((x.fac - median).abs()).quantile(0.5)
            max_range1 = median + n * diff_median
            min_range1 = median - n * diff_median
            max_range2 = median + (n + 0.5) * diff_median
            min_range2 = median - (n + 0.5) * diff_median
            x = x.sort_values(["fac"])
            x_min = x[x.fac <= min_range1]
            x_max = x[x.fac >= max_range1]
            x_middle = x[(x.fac > min_range1) & (x.fac < max_range1)]
            x_min.fac = np.nan
            x_max.fac = np.nan
            if x_min.shape[0] >= 1:
                x_min.fac.iloc[-1] = min_range1
                if x_min.shape[0] >= 2:
                    x_min.fac.iloc[0] = min_range2
                    x_min.fac = x_min.fac.interpolate()
            if x_max.shape[0] >= 1:
                x_max.fac.iloc[-1] = max_range2
                if x_max.shape[0] >= 2:
                    x_max.fac.iloc[0] = max_range1
                    x_max.fac = x_max.fac.interpolate()
            x = pd.concat([x_min, x_middle, x_max]).sort_values(["code"])
            return x

        df = df.groupby(["date"]).apply(lambda x: clip_sing(x, n))
        try:
            df = df.reset_index()
        except Exception:
            ...
        df = df.drop_duplicates(subset=["date", "code"]).pivot(
            index="date", columns="code", values="fac"
        )
        return df
    elif replace:

        def clip_sing(x: pd.Series, n: float = 3):
            median = x.quantile(0.5)
            diff_median = ((x - median).abs()).quantile(0.5)
            max_range = median + n * diff_median
            min_range = median - n * diff_median
            x = x.where(x < max_range, max_range)
            x = x.where(x > min_range, min_range)
            return x

        df1 = df.T.apply(lambda x: clip_sing(x, n)).T
        df = np.abs(np.sign(df)) * df1
        return df
    else:
        df0 = df.T
        median = df0.quantile(0.5)
        diff_median = ((df0 - median).abs()).quantile(0.5)
        max_range = median + n * diff_median
        min_range = median - n * diff_median
        mid1 = (((df0 - min_range) >= 0) + 0).replace(0, np.nan)
        mid2 = (((df0 - max_range) <= 0) + 0).replace(0, np.nan)
        return (df0 * mid1 * mid2).T


@do_on_dfs
def clip_three_sigma(df: pd.DataFrame, n: float = 3) -> pd.DataFrame:
    df0 = df.T
    mean = df0.mean()
    std = df0.std()
    max_range = mean + n * std
    min_range = mean - n * std
    mid1 = (((df0 - min_range) >= 0) + 0).replace(0, np.nan)
    mid2 = (((df0 - max_range) <= 0) + 0).replace(0, np.nan)
    return (df0 * mid1 * mid2).T


@do_on_dfs
def clip_percentile(
    df: pd.DataFrame, min_percent: float = 0.025, max_percent: float = 0.975
) -> pd.DataFrame:
    df0 = df.T
    max_range = df0.quantile(max_percent)
    min_range = df0.quantile(min_percent)
    mid1 = (((df0 - min_range) >= 0) + 0).replace(0, np.nan)
    mid2 = (((df0 - max_range) <= 0) + 0).replace(0, np.nan)
    return (df0 * mid1 * mid2).T


@do_on_dfs
def clip(
    df: pd.DataFrame,
    mad: bool = 0,
    three_sigma: bool = 0,
    percentile: bool = 0,
    parameter: Union[float, tuple] = None,
) -> pd.DataFrame:
    """对因子值进行截面去极值的操作

    Parameters
    ----------
    df : pd.DataFrame
        要处理的因子表，columns为股票代码，index为时间
    mad : bool, optional
        使用mad法去极值，先计算所有因子与平均值之间的距离总和来检测离群值, by default 0
    three_sigma : bool, optional
        根据均值和几倍标准差做调整, by default 0
    percentile : bool, optional
        根据上下限的分位数去极值, by default 0
    parameter : Union[float,tuple], optional
        参数，mad和three_sigma默认参数为3，输入float形式；而percentile默认参数为(0.025,0.975)，输入tuple形式, by default None
    [参考资料](https://blog.csdn.net/The_Time_Runner/article/details/100118505)

    Returns
    -------
    pd.DataFrame
        去极值后的参数

    Raises
    ------
    ValueError
        不指定方法或参数类型错误，将报错
    """
    if mad and ((isinstance(parameter, float)) or (parameter is None)):
        return clip_mad(df, parameter)
    elif three_sigma and ((isinstance(parameter, float)) or (parameter is None)):
        return clip_three_sigma(df, parameter)
    elif percentile and ((isinstance(parameter, tuple)) or (parameter is None)):
        return clip_percentile(df, parameter[0], parameter[1])
    else:
        raise ValueError("参数输入错误")


def judge_factor_by_third(
    fac1: pd.DataFrame, fac2: pd.DataFrame, judge: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """对于fac1和fac2两个因子，依据judge这个series或dataframe进行判断，
    judge可能为全市场的某个时序指标，也可能是每个股票各一个的指标，
    如果judge这一期的值大于0，则取fac1的值，小于0则取fac2的值

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1，index为时间，columns为股票代码，values为因子值
    fac2 : pd.DataFrame
        因子2，index为时间，columns为股票代码，values为因子值
    judge : Union[pd.DataFrame,pd.Series]
        市场指标或个股指标，为市场指标时，则输入series形式，index为时间，values为指标值
        为个股指标时，则输入dataframe形式，index为时间，columns为股票代码，values为因子值

    Returns
    -------
    pd.DataFrame
        合成后的因子值，index为时间，columns为股票代码，values为因子值
    """
    if isinstance(judge, pd.Series):
        judge = pd.DataFrame(
            {k: list(judge) for k in list(fac1.columns)}, index=judge.index
        )
    s1 = (judge > 0) + 0
    s2 = (judge < 0) + 0
    fac1 = fac1 * s1
    fac2 = fac2 * s2
    fac = fac1 + fac2
    have = np.sign(fac1.abs() + 1)
    return fac * have
