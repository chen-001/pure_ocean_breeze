import copy
import warnings

warnings.filterwarnings("ignore")
import os
import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
import statsmodels.formula.api as smf
from functools import partial, reduce
from collections import Iterable
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import plotly.express as pe
import plotly.io as pio
import scipy.stats as ss
from loguru import logger
import time
from functools import lru_cache, wraps

# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
import h5py
from cachier import cachier
import pickle
import knockknock as kk
import alphalens as al
import dcube as dc
from tenacity import retry
import pickledb
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.application import MIMEApplication
import pymysql
from sqlalchemy import create_engine
from sqlalchemy import FLOAT, INT, VARCHAR, BIGINT
import varname
from typing import Union
import pretty_errors
import rqdatac

rqdatac.init()

# DONE: 补充每个部分的数据类型和文档，并尝试用AI Python Docstring生成文档
# TODO: 补充分行业的行业内测试，封装为类（包括各行业Rank IC和行业多头超额，以及多头超额名单（可选是否中性化））
# DONE: 彻底替换其中的minute_data_file，重写pure_fall类，变为只更新用的类
# DONE: 对函数和类重新分类排序，按照功能划分
# DONE: 重写函数说明和注释
# DONE: 写文档
# TODO: 增加读取初级因子值（生成多个因子值时，拆分出不同的初级因子，放在最终因子数据的文件夹下，每个因子一个文件夹）
# DONE: 拆分不同模块
# DONE: 拆分更新日志
# TODO: 一键读取指数成分股
# TODO: 一键读入上市天数、st状态、交易状态


STATES = {
    "NO_LOG": False,
    "NO_COMMENT": False,
    "NO_SAVE": False,
    "NO_PLOT": False,
    "START": 20130101,
    "db_host": "127.0.0.1",
    "db_port": 3306,
    "db_user": "root",
    "db_password": "Kingwila98",
}


class HomePlace(object):
    __slots__ = [
        "daily_data_file",
        "minute_data_file",
        "factor_data_file",
        "barra_data_file",
        "update_data_file",
        "api_token",
        "final_factor_file",
        "daily_enddate",
        "minute_enddate",
    ]

    def __init__(self):
        user_file = os.path.expanduser("~") + "/"
        path_file = open(user_file + "paths.settings", "rb")
        paths = pickle.load(path_file)
        for k in self.__slots__:
            setattr(self, k, paths[k])


homeplace = HomePlace()
pro = dc.pro_api(homeplace.api_token)


class params_setter(object):
    """用于标注设置参数部分的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是设置参数类型的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func
        self.func = func

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            # if not STATES['NO_LOG:
            #     logger.info(f'{func.__name__} has been called $ kind of params_setter')

        return wrapper


class main_process(object):
    """用于标记主逻辑过程的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是主逻辑过程的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            # if STATES['NO_LOG:
            #     logger.success(f'{func.__name__} has been called $ kind of main_process')

        return wrapper


class tool_box(object):
    """用于标注工具箱部分的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是工具箱的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func

        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            # logger.success(f'{func.__name__} has been called $ kind of tool_box')
            return res

        return wrapper


class history_remain(object):
    """用于历史遗留部分的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是历史遗留的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            # logger.success(f'{func.__name__} has been called $ kind of history_remain')

        return wrapper


@cachier()
def read_daily(
    path=None,
    open=0,
    close=0,
    high=0,
    low=0,
    tr=0,
    sharenum=0,
    volume=0,
    unadjust=0,
    start=STATES["START"],
):
    """读取日频数据,使用read_daily.clear_cache()来清空缓存"""

    def read_mat(path):
        homeplace = HomePlace()
        col = list(
            scio.loadmat(homeplace.daily_data_file + "AllStockCode.mat").values()
        )[3]
        index = list(
            scio.loadmat(homeplace.daily_data_file + "TradingDate_Daily.mat").values()
        )[3]
        col = [i[0] for i in col[0]]
        index = index[0].tolist()
        path = homeplace.daily_data_file + path
        data = list(scio.loadmat(path).values())[3]
        data = pd.DataFrame(data, index=index, columns=col)
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
        data = data.replace(0, np.nan)
        data = data[data.index >= pd.Timestamp(str(start))]
        return data

    if not unadjust:
        if path:
            return read_mat(path)
        elif open:
            trs = read_mat("AllStock_DailyTR.mat")
            opens = read_mat("AllStock_DailyOpen_dividend.mat")
            return np.sign(trs) * opens
        elif close:
            trs = read_mat("AllStock_DailyTR.mat")
            closes = read_mat("AllStock_DailyClose_dividend.mat")
            return np.sign(trs) * closes
        elif high:
            trs = read_mat("AllStock_DailyTR.mat")
            highs = read_mat("AllStock_DailyHigh_dividend.mat")
            return np.sign(trs) * highs
        elif low:
            trs = read_mat("AllStock_DailyTR.mat")
            lows = read_mat("AllStock_DailyLow_dividend.mat")
            return np.sign(trs) * lows
        elif tr:
            trs = read_mat("AllStock_DailyTR.mat")
            return trs
        elif sharenum:
            sharenums = read_mat("AllStock_DailyAShareNum.mat")
            return sharenums
        elif volume:
            volumes = read_mat("AllStock_DailyVolume.mat")
            return volumes
        else:
            raise IOError("阁下总得读点什么吧？🤒")
    else:
        if path:
            return read_mat(path)
        elif open:
            trs = read_mat("AllStock_DailyTR.mat")
            opens = read_mat("AllStock_DailyOpen.mat")
            return np.sign(trs) * opens
        elif close:
            trs = read_mat("AllStock_DailyTR.mat")
            closes = read_mat("AllStock_DailyClose.mat")
            return np.sign(trs) * closes
        elif high:
            trs = read_mat("AllStock_DailyTR.mat")
            highs = read_mat("AllStock_DailyHigh.mat")
            return np.sign(trs) * highs
        elif low:
            trs = read_mat("AllStock_DailyTR.mat")
            lows = read_mat("AllStock_DailyLow.mat")
            return np.sign(trs) * lows
        elif tr:
            trs = read_mat("AllStock_DailyTR.mat")
            return trs
        elif sharenum:
            sharenums = read_mat("AllStock_DailyAShareNum.mat")
            return sharenums
        elif volume:
            volumes = read_mat("AllStock_DailyVolume.mat")
            return volumes
        else:
            raise IOError("阁下总得读点什么吧？🤒")


def read_market(
    full=False, wide=True, open=0, high=0, low=0, close=0, amount=0, money=0
):
    """读取wind全A日行情，如果为full，则直接返回原始表格，如果full为False，则返回部分数据
    如果wide为True，则返回方阵形式，index是时间，columns是股票代码，每一列的数据都一样，这么做是指为了便于与个股运算"""
    market = pd.read_excel(homeplace.daily_data_file + "wind全A日行情.xlsx")
    market = market.drop(columns=["代码", "名称"])
    market.columns = ["date", "open", "high", "low", "close", "amount", "money"]
    market.money = market.money * 1000000
    if full:
        return market
    else:
        if wide:
            tr = read_daily(tr=1)
            tr = np.abs(np.sign(tr)).replace(1, 0).fillna(0)
            if open:
                market = market[["date", "open"]]
                market.date = pd.to_datetime(market.date)
                market = market[market.date.isin(list(tr.index))]
                market = market.set_index("date")
                market = market["open"]
                tr_one = tr.iloc[:, 0]
                market = market + tr_one
                market = market.fillna(method="ffill")
                market = pd.DataFrame(
                    {k: list(market) for k in list(tr.columns)}, index=tr.index
                )
            elif high:
                market = market[["date", "high"]]
                market.date = pd.to_datetime(market.date)
                market = market[market.date.isin(list(tr.index))]
                market = market.set_index("date")
                market = market["high"]
                tr_one = tr.iloc[:, 0]
                market = market + tr_one
                market = market.fillna(method="ffill")
                market = pd.DataFrame(
                    {k: list(market) for k in list(tr.columns)}, index=tr.index
                )
            elif low:
                market = market[["date", "low"]]
                market.date = pd.to_datetime(market.date)
                market = market[market.date.isin(list(tr.index))]
                market = market.set_index("date")
                market = market["date", "low"]
                tr_one = tr.iloc[:, 0]
                market = market + tr_one
                market = market.fillna(method="ffill")
                market = pd.DataFrame(
                    {k: list(market) for k in list(tr.columns)}, index=tr.index
                )
            elif close:
                market = market[["date", "close"]]
                market.date = pd.to_datetime(market.date)
                market = market[market.date.isin(list(tr.index))]
                market = market.set_index("date")
                market = market["close"]
                tr_one = tr.iloc[:, 0]
                market = market + tr_one
                market = market.fillna(method="ffill")
                market = pd.DataFrame(
                    {k: list(market) for k in list(tr.columns)}, index=tr.index
                )
            elif amount:
                market = market[["date", "amount"]]
                market.date = pd.to_datetime(market.date)
                market = market[market.date.isin(list(tr.index))]
                market = market.set_index("date")
                market = market["amount"]
                tr_one = tr.iloc[:, 0]
                market = market + tr_one
                market = market.fillna(method="ffill")
                market = pd.DataFrame(
                    {k: list(market) for k in list(tr.columns)}, index=tr.index
                )
            elif money:
                market = market[["date", "money"]]
                market.date = pd.to_datetime(market.date)
                market = market[market.date.isin(list(tr.index))]
                market = market.set_index("date")
                market = market["money"]
                tr_one = tr.iloc[:, 0]
                market = market + tr_one
                market = market.fillna(method="ffill")
                market = pd.DataFrame(
                    {k: list(market) for k in list(tr.columns)}, index=tr.index
                )
            else:
                raise IOError("您总得读点什么吧？🤒")
            return market
        else:
            cols = [
                varname.nameof(i)
                for i in [open, high, low, close, amount, money]
                if i == 1
            ]
            market = market[["date"] + cols]
            return market


def read_h5(path: str) -> dict:
    """
    Reads a HDF5 file into a dictionary of pandas DataFrames.

    Parameters
    ----------
    path : str
        The path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary of pandas DataFrames.
    """
    res = {}
    a = h5py.File(path)
    for k, v in tqdm.tqdm(list(a.items()), desc="数据加载中……"):
        value = list(v.values())[-1]
        col = [i.decode("utf-8") for i in list(list(v.values())[0])]
        ind = [i.decode("utf-8") for i in list(list(v.values())[1])]
        res[k] = pd.DataFrame(value, columns=col, index=ind)
    return res


def read_h5_new(path: str) -> pd.DataFrame:
    """
    读取h5文件
    :param path: 输入h5文件的路径
    :return: 返回一个pd.DataFrame
    """
    a = h5py.File(path)
    v = list(a.values())[0]
    v = a[v.name][:]
    return pd.DataFrame(v)


def convert_code(x):
    """将米筐代码转换为wind代码，并识别其是股票还是指数"""
    x1 = x.split("/")[-1].split(".")[0]
    x2 = x.split("/")[-1].split(".")[1]
    if x2 == "XSHE":
        x2 = ".SZ"
    elif x2 == "XSHG":
        x2 = ".SH"
    x = x1 + x2
    if x1[0] in ["0", "3"] and x2 == ".SZ":
        kind = "stock"
    elif x1[0] == "6" and x2 == ".SH":
        kind = "stock"
    else:
        kind = "index"
    return x, kind


def get_value(df, n):
    """很多因子计算时，会一次性生成很多值，使用时只取出一个值"""

    def get_value_single(x, n):
        try:
            return x[n]
        except Exception:
            return np.nan

    df = df.applymap(lambda x: get_value_single(x, n))
    return df


def comment_on_rets_and_nets(rets, nets, name):
    """
    输入收益率序列和净值序列，输出年化收益、年化波动、信息比率、月度胜率和最大回撤率
    输入2个pd.Series，时间是索引
    """
    duration_nets = (nets.index[-1] - nets.index[0]).days
    year_nets = duration_nets / 365
    ret_yearly = (nets.iloc[-1] / nets.iloc[0]) ** (1 / year_nets) - 1
    max_draw = ((nets.cummax() - nets) / nets.cummax()).max()
    vol = np.std(rets) * (12**0.5)
    info_rate = ret_yearly / vol
    win_rate = len(rets[rets > 0]) / len(rets)
    comments = pd.DataFrame(
        {
            "年化收益率": ret_yearly,
            "年化波动率": vol,
            "信息比率": info_rate,
            "月度胜率": win_rate,
            "最大回撤率": max_draw,
        },
        index=[name],
    ).T
    return comments


def comments_on_twins(series, series1):
    """对twins中的结果给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率"""
    ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    duration = (series.index[-1] - series.index[0]).days
    year = duration / 365
    ret_yearly = (series.iloc[-1] / series.iloc[0]) ** (1 / year) - 1
    max_draw = -(series / series.expanding(1).max() - 1).min()
    vol = np.std(series1) * (12**0.5)
    sharpe = ret_yearly / vol
    wins = series1[series1 > 0]
    win_rate = len(wins) / len(series1)
    return pd.Series(
        [ret, ret_yearly, vol, sharpe, win_rate, max_draw],
        index=["总收益率", "年化收益率", "年化波动率", "信息比率", "胜率", "最大回撤率"],
    )


def comments_on_twins_periods(series, series1, periods=None):
    """对twins中的结果给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率"""
    ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    duration = (series.index[-1] - series.index[0]).days
    year = duration / 365
    ret_yearly = (series.iloc[-1] / series.iloc[0]) ** (1 / year) - 1
    max_draw = -(series / series.expanding(1).max() - 1).min()
    vol = np.std(series1) * (252**0.5) * (periods**0.5)
    sharpe = ret_yearly / vol
    wins = series1[series1 > 0]
    win_rate = len(wins) / len(series1)
    return pd.Series(
        [ret, ret_yearly, vol, sharpe, win_rate, max_draw],
        index=["总收益率", "年化收益率", "年化波动率", "信息比率", "胜率", "最大回撤率"],
    )


def daily_factor_on300500_old(
    fac, hs300=False, zz500=False, zz800=False, zz1000=False, gz2000=False, other=False
):
    """输入日频因子，把日频因子变为仅在300或者500上的股票池"""
    last = fac.resample("M").last()
    homeplace = HomePlace()
    if fac.shape[0] / last.shape[0] > 2:
        if hs300:
            df = (
                pd.read_feather(homeplace.daily_data_file + "沪深300成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz500:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证500成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz800:
            df1 = pd.read_feather(
                homeplace.daily_data_file + "沪深300成分股.feather"
            ).set_index("index")
            df2 = pd.read_feather(
                homeplace.daily_data_file + "中证500成分股.feather"
            ).set_index("index")
            df = df1 + df2
            df = df.replace(0, np.nan)
            df = df * fac
            df = df.dropna(how="all")
        elif zz1000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证1000成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif gz2000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "国证2000成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif other:
            tr = read_daily(tr=1).fillna(0).replace(0, 1)
            tr = np.sign(tr)
            df1 = (
                tr
                * pd.read_feather(
                    homeplace.daily_data_file + "沪深300成分股.feather"
                ).set_index("index")
            ).fillna(0)
            df2 = (
                tr
                * pd.read_feather(
                    homeplace.daily_data_file + "中证500成分股.feather"
                ).set_index("index")
            ).fillna(0)
            df3 = (
                tr
                * pd.read_feather(
                    homeplace.daily_data_file + "中证1000成分股.feather"
                ).set_index("index")
            ).fillna(0)
            df = (1 - df1) * (1 - df2) * (1 - df3) * tr
            df = df.replace(0, np.nan) * fac
            df = df.dropna(how="all")
        else:
            raise ValueError("总得指定一下是哪个成分股吧🤒")
    else:
        if hs300:
            df = (
                pd.read_feather(homeplace.daily_data_file + "沪深300成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz500:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证500成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz800:
            df1 = (
                pd.read_feather(homeplace.daily_data_file + "沪深300成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            )
            df2 = (
                pd.read_feather(homeplace.daily_data_file + "中证500成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            )
            df = df1 + df2
            df = df.replace(0, np.nan)
            df = df * fac
            df = df.dropna(how="all")
        elif zz1000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证1000成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif gz2000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "国证2000成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif other:
            tr = read_daily(tr=1).fillna(0).replace(0, 1).resample("M").last()
            tr = np.sign(tr)
            df1 = (
                tr
                * pd.read_feather(homeplace.daily_data_file + "沪深300成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            ).fillna(0)
            df2 = (
                tr
                * pd.read_feather(homeplace.daily_data_file + "中证500成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            ).fillna(0)
            df3 = (
                tr
                * pd.read_feather(homeplace.daily_data_file + "中证1000成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            ).fillna(0)
            df = (1 - df1) * (1 - df2) * (1 - df3)
            df = df.replace(0, np.nan) * fac
            df = df.dropna(how="all")
        else:
            raise ValueError("总得指定一下是哪个成分股吧🤒")
    return df


def daily_factor_on300500(
    fac, hs300=False, zz500=False, zz800=False, zz1000=False, gz2000=False, other=False
):
    """输入日频因子，把日频因子变为仅在300或者500上的股票池"""
    last = fac.resample("M").last()
    homeplace = HomePlace()
    if fac.shape[0] / last.shape[0] > 2:
        if hs300:
            df = (
                pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz500:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz800:
            df1 = pd.read_feather(
                homeplace.daily_data_file + "沪深300日成分股.feather"
            ).set_index("index")
            df2 = pd.read_feather(
                homeplace.daily_data_file + "中证500日成分股.feather"
            ).set_index("index")
            df = df1 + df2
            df = df.replace(0, np.nan)
            df = df * fac
            df = df.dropna(how="all")
        elif zz1000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证1000日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif gz2000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "国证2000日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
            )
            df = df * fac
            df = df.dropna(how="all")
        elif other:
            tr = read_daily(tr=1).fillna(0).replace(0, 1)
            tr = np.sign(tr)
            df1 = (
                tr
                * pd.read_feather(
                    homeplace.daily_data_file + "沪深300日成分股.feather"
                ).set_index("index")
            ).fillna(0)
            df2 = (
                tr
                * pd.read_feather(
                    homeplace.daily_data_file + "中证500日成分股.feather"
                ).set_index("index")
            ).fillna(0)
            df3 = (
                tr
                * pd.read_feather(
                    homeplace.daily_data_file + "中证1000日成分股.feather"
                ).set_index("index")
            ).fillna(0)
            df = (1 - df1) * (1 - df2) * (1 - df3) * tr
            df = df.replace(0, np.nan) * fac
            df = df.dropna(how="all")
        else:
            raise ValueError("总得指定一下是哪个成分股吧🤒")
    else:
        if hs300:
            df = (
                pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz500:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif zz800:
            df1 = (
                pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            )
            df2 = (
                pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            )
            df = df1 + df2
            df = df.replace(0, np.nan)
            df = df * fac
            df = df.dropna(how="all")
        elif zz1000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "中证1000日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif gz2000:
            df = (
                pd.read_feather(homeplace.daily_data_file + "国证2000日成分股.feather")
                .set_index("index")
                .replace(0, np.nan)
                .resample("M")
                .last()
            )
            df = df * fac
            df = df.dropna(how="all")
        elif other:
            tr = read_daily(tr=1).fillna(0).replace(0, 1).resample("M").last()
            tr = np.sign(tr)
            df1 = (
                tr
                * pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            ).fillna(0)
            df2 = (
                tr
                * pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            ).fillna(0)
            df3 = (
                tr
                * pd.read_feather(homeplace.daily_data_file + "中证1000日成分股.feather")
                .set_index("index")
                .resample("M")
                .last()
            ).fillna(0)
            df = (1 - df1) * (1 - df2) * (1 - df3)
            df = df.replace(0, np.nan) * fac
            df = df.dropna(how="all")
        else:
            raise ValueError("总得指定一下是哪个成分股吧🤒")
    return df


def select_max(df1, df2):
    """两个columns与index完全相同的df，每个值都挑出较大值"""
    return (df1 + df2 + np.abs(df1 - df2)) / 2


def select_min(df1, df2):
    """两个columns与index完全相同的df，每个值都挑出较小值"""
    return (df1 + df2 - np.abs(df1 - df2)) / 2


@kk.desktop_sender(title="嘿，行业中性化做完啦～🛁")
def decap(df, daily=False, monthly=False):
    """做市值中性化"""
    tqdm.tqdm.pandas()
    share = read_daily("AllStock_DailyAShareNum.mat")
    undi_close = read_daily("AllStock_DailyClose.mat")
    cap = (share * undi_close).stack().reset_index()
    cap.columns = ["date", "code", "cap"]
    cap.cap = ss.boxcox(cap.cap)[0]

    def single(x):
        x.cap = ss.boxcox(x.cap)[0]
        return x

    cap = cap.groupby(["date"]).apply(single)
    cap = cap.set_index(["date", "code"]).unstack()
    cap.columns = [i[1] for i in list(cap.columns)]
    cap_monthly = cap.resample("M").last()
    last = df.resample("M").last()
    if df.shape[0] / last.shape[0] < 2:
        monthly = True
    else:
        daily = True
    if daily:
        df = (pure_fallmount(df) - (pure_fallmount(cap),))()
    elif monthly:
        df = (pure_fallmount(df) - (pure_fallmount(cap_monthly),))()
    else:
        raise NotImplementedError("必须指定频率")
    return df


@kk.desktop_sender(title="嘿，行业市值中性化做完啦～🛁")
def decap_industry(df, daily=False, monthly=False):
    last = df.resample("M").last()
    homeplace = HomePlace()
    share = read_daily("AllStock_DailyAShareNum.mat")
    undi_close = read_daily("AllStock_DailyClose.mat")
    cap = (share * undi_close).stack().reset_index()
    cap.columns = ["date", "code", "cap"]
    cap.cap = ss.boxcox(cap.cap)[0]

    def single(x):
        x.cap = ss.boxcox(x.cap)[0]
        return x

    cap = cap.groupby(["date"]).apply(single)
    df = df.stack().reset_index()
    df.columns = ["date", "code", "fac"]
    df = pd.merge(df, cap, on=["date", "code"])
    if df.shape[0] / last.shape[0] < 2:
        monthly = True
    else:
        daily = True

    def neutralize_factors(df):
        """组内对因子进行市值中性化"""
        industry_codes = list(df.columns)
        industry_codes = [i for i in industry_codes if i.startswith("w")]
        industry_codes_str = "+".join(industry_codes)
        ols_result = smf.ols("fac~cap+" + industry_codes_str, data=df).fit()
        ols_w = ols_result.params["cap"]
        ols_b = ols_result.params["Intercept"]
        ols_bs = {}
        for ind in industry_codes:
            ols_bs[ind] = ols_result.params[ind]
        df.fac = df.fac - ols_w * df.cap - ols_b
        for k, v in ols_bs.items():
            df.fac = df.fac - v * df[k]
        df = df[["fac"]]
        return df

    if monthly:
        industry_dummy = (
            pd.read_feather(homeplace.daily_data_file + "申万行业2021版哑变量.feather")
            .set_index("date")
            .groupby("code")
            .resample("M")
            .last()
        )
        industry_dummy = industry_dummy.fillna(0).drop(columns=["code"]).reset_index()
        industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
        col = ["code", "date"] + industry_ws
    elif daily:
        industry_dummy = pd.read_feather(
            homeplace.daily_data_file + "申万行业2021版哑变量.feather"
        ).fillna(0)
        industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
        col = ["date", "code"] + industry_ws
    industry_dummy.columns = col
    df = pd.merge(df, industry_dummy, on=["date", "code"])
    df = df.set_index(["date", "code"])
    tqdm.tqdm.pandas()
    df = df.groupby(["date"]).progress_apply(neutralize_factors)
    df = df.unstack()
    df.columns = [i[1] for i in list(df.columns)]
    return df


def detect_nan(df):
    x = np.sum(df.to_numpy().flatten())
    if np.isnan(x):
        print(df)
        return np.nan
    else:
        return x


def deboth(df):
    shen = pure_moonnight(df, boxcox=1)
    return shen()


def boom_four(df, minus=None, backsee=20, daily=False, min_periods=None):
    """使用20天均值和标准差，生成4个因子"""
    if min_periods is None:
        min_periods = int(backsee * 0.5)
    if not daily:
        df_mean = (
            df.rolling(backsee, min_periods=min_periods).mean().resample("M").last()
        )
        df_std = df.rolling(backsee, min_periods=min_periods).std().resample("M").last()
        twins_add = (pure_fallmount(df_mean) + (pure_fallmount(df_std),))()
        rtwins_add = df_mean.rank(axis=1) + df_std.rank(axis=1)
        twins_minus = (pure_fallmount(df_mean) + (pure_fallmount(-df_std),))()
        rtwins_minus = df_mean.rank(axis=1) - df_std.rank(axis=1)
    else:
        df_mean = df.rolling(backsee, min_periods=min_periods).mean()
        df_std = df.rolling(backsee, min_periods=min_periods).std()
        twins_add = (pure_fallmount(df_mean) + (pure_fallmount(df_std),))()
        rtwins_add = df_mean.rank(axis=1) + df_std.rank(axis=1)
        twins_minus = (pure_fallmount(df_mean) + (pure_fallmount(-df_std),))()
        rtwins_minus = df_mean.rank(axis=1) - df_std.rank(axis=1)
    return df_mean, df_std, twins_add, rtwins_add, twins_minus, rtwins_minus


def get_abs(df, median=False, square=False):
    """生产因子截面上距离均值的距离"""
    if not square:
        if median:
            return np.abs((df.T - df.T.median()).T)
        else:
            return np.abs((df.T - df.T.mean()).T)
    else:
        if median:
            return ((df.T - df.T.median()).T) ** 2
        else:
            return ((df.T - df.T.mean()).T) ** 2


def add_cross_standardlize(*args):
    """将众多因子横截面标准化之后相加"""
    fms = [pure_fallmount(i) for i in args]
    one = fms[0]
    others = fms[1:]
    final = one + others
    return final()


def get_normal(df):
    """将因子横截面正态化"""
    df = df.replace(0, np.nan)
    df = df.T.apply(lambda x: ss.boxcox(x)[0]).T
    return df


def read_index_three_old(day=False):
    """读取三大指数的原始行情数据，返回并保存在本地"""

    def read(file):
        homeplace = HomePlace()
        file1 = homeplace.daily_data_file + f"{file}日行情.xlsx"
        df = pd.read_excel(file1)
        df.columns = [
            "code",
            "name",
            "date",
            "open",
            "high",
            "low",
            "close",
            "amount",
            "money",
        ]
        df = df[["date", "close"]]
        df.date = pd.to_datetime(df.date)
        df = df.set_index("date")
        df = df.resample("M").last()
        df.columns = [file]
        if not day:
            df = df[df.index >= pd.Timestamp(str(STATES["START"]))]
        else:
            df = df[df.index >= pd.Timestamp(day)]
        df = df / df[file].iloc[0]
        return df

    hs300 = read("沪深300")
    zz500 = read("中证500")
    zz1000 = read("中证1000")
    w = pd.ExcelWriter("3510原始行情.xlsx")
    hs300.to_excel(w, sheet_name="300")
    zz500.to_excel(w, sheet_name="500")
    zz1000.to_excel(w, sheet_name="1000")
    w.save()
    w.close()
    return hs300, zz500, zz1000


def read_index_three(day=None):
    """读取三大指数的原始行情数据，返回并保存在本地"""
    if day is None:
        day = STATES["START"]
    res = pd.read_feather(homeplace.daily_data_file + "3510行情.feather").set_index(
        "date"
    )
    hs300, zz500, zz1000 = res.沪深300, res.中证500, res.中证1000
    hs300 = hs300[hs300.index >= pd.Timestamp(str(day))]
    zz500 = zz500[zz500.index >= pd.Timestamp(str(day))]
    zz1000 = zz1000[zz1000.index >= pd.Timestamp(str(day))]

    def to_one(df):
        df = df / df.iloc[0]
        return df

    w = pd.ExcelWriter("3510原始行情.xlsx")
    hs300.to_excel(w, sheet_name="300")
    zz500.to_excel(w, sheet_name="500")
    zz1000.to_excel(w, sheet_name="1000")
    w.save()
    w.close()
    return hs300, zz500, zz1000


def read_industry_prices(day=None, monthly=True):
    """读取申万一级行业指数"""
    if day is None:
        day = STATES["START"]
    df = pd.read_feather(homeplace.daily_data_file + "各行业行情数据.feather").set_index(
        "date"
    )
    if monthly:
        df = df.resample("M").last()
    return df


def make_relative_comments(ret_fac, hs300=0, zz500=0, zz1000=0, day=None):
    if hs300:
        net_index = read_index_three(day=day)[0]
    elif zz500:
        net_index = read_index_three(day=day)[1]
    elif zz1000:
        net_index = read_index_three(day=day)[2]
    else:
        raise IOError("你总得指定一个股票池吧？")
    ret_index = net_index.pct_change()
    ret = ret_fac - ret_index
    ret = ret.dropna()
    net = (1 + ret).cumprod()
    ntop = pd.Series(1, index=[net.index.min() - pd.DateOffset(months=1)])
    rtop = pd.Series(0, index=[net.index.min() - pd.DateOffset(months=1)])
    net = pd.concat([ntop, net]).resample("M").last()
    ret = pd.concat([rtop, ret]).resample("M").last()
    com = comments_on_twins(net, ret)
    return com


def make_relative_comments_plot(ret_fac, hs300=0, zz500=0, zz1000=0, day=None):
    if hs300:
        net_index = read_index_three(day=day)[0]
    elif zz500:
        net_index = read_index_three(day=day)[1]
    elif zz1000:
        net_index = read_index_three(day=day)[2]
    else:
        raise IOError("你总得指定一个股票池吧？")
    ret_index = net_index.pct_change()
    ret = ret_fac - ret_index
    ret = ret.dropna()
    net = (1 + ret).cumprod()
    ntop = pd.Series(1, index=[net.index.min() - pd.DateOffset(months=1)])
    rtop = pd.Series(0, index=[net.index.min() - pd.DateOffset(months=1)])
    net = pd.concat([ntop, net]).resample("M").last()
    ret = pd.concat([rtop, ret]).resample("M").last()
    com = comments_on_twins(net, ret)
    net.plot(rot=60)
    plt.show()
    return net


def comments_ten(shen):
    rets_cols = list(shen.shen.group_rets.columns)
    rets_cols = rets_cols[:-1]
    coms = []
    for i in rets_cols:
        ret = shen.shen.group_rets[i]
        net = shen.shen.group_net_values[i]
        com = comments_on_twins(net, ret)
        com = com.to_frame(i)
        coms.append(com)
    df = pd.concat(coms, axis=1)
    return df.T


def coin_reverse(ret20, vol20, mean=1, positive_negtive=0):
    """根据vol20的大小，翻转一半ret20，把vol20较大的部分，给ret20添加负号"""
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


def indus_name(df, col_name=None):
    """将2021版申万行业的代码，转化为对应行业的名字"""
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


INDUS_DICT = {
    k: v
    for k, v in zip(
        [
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
        [
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
    )
}

INDEX_DICT = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "399303.SZ": "国证2000",
}


def multidfs_to_one(*args):
    """很多个df，各有一部分，其余位置都是空，
    想把各自df有值的部分保留，都没有值的部分继续设为空"""
    dfs = [i.fillna(0) for i in args]
    background = np.sign(np.abs(np.sign(sum(dfs))) + 1).replace(1, 0)
    dfs = [(i + background).fillna(0) for i in dfs]
    df_nans = [i.isna() for i in dfs]
    nan = reduce(lambda x, y: x * y, df_nans)
    nan = nan.replace(1, np.nan)
    nan = nan.replace(0, 1)
    df_final = sum(dfs) * nan
    return df_final


def to_tradeends(df):
    """将最后一个自然日改变为最后一个交易日"""
    trs = read_daily(tr=1)
    trs = trs.assign(tradeends=list(trs.index))
    trs = trs[["tradeends"]]
    trs = trs.resample("M").last()
    df = pd.concat([trs, df], axis=1)
    df = df.set_index(["tradeends"])
    return df


def market_kind(df, zhuban=False, chuangye=False, kechuang=False, beijing=False):
    """与宽基指数成分股的函数类似，限定股票在某个具体板块上"""
    trs = read_daily(tr=1)
    codes = list(trs.columns)
    dates = list(trs.index)
    if chuangye and kechuang:
        dummys = [1 if code[:2] in ["30", "68"] else np.nan for code in codes]
    else:
        if zhuban:
            dummys = [1 if code[:2] in ["00", "60"] else np.nan for code in codes]
        elif chuangye:
            dummys = [1 if code.startswith("3") else np.nan for code in codes]
        elif kechuang:
            dummys = [1 if code.startswith("68") else np.nan for code in codes]
        elif beijing:
            dummys = [1 if code.startswith("8") else np.nan for code in codes]
        else:
            raise ValueError("你总得选一个股票池吧？🤒")
    dummy_dict = {k: v for k, v in zip(codes, dummys)}
    dummy_df = pd.DataFrame(dummy_dict, index=dates)
    df = df * dummy_df
    return df


def to_percent(x):
    """把小数转化为2位小数的百分数"""
    if np.isnan(x):
        return x
    else:
        x = str(round(x * 100, 2)) + "%"
        return x


def show_corr(fac1, fac2, method="spearman", plt_plot=True):
    """展示两个因子的截面相关性"""
    both1 = fac1.stack().reset_index()
    befo1 = fac2.stack().reset_index()
    both1.columns = ["date", "code", "both"]
    befo1.columns = ["date", "code", "befo"]
    twins = pd.merge(both1, befo1, on=["date", "code"]).set_index(["date", "code"])
    corr = twins.groupby("date").apply(lambda x: x.corr(method=method).iloc[0, 1])
    if plt_plot:
        corr.plot(rot=60)
        plt.show()
    return corr.mean()


def show_corrs(
    factors: list[pd.DataFrame] = None,
    factor_names: list[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
) -> pd.DataFrame:
    """展示很多因子的截面相关性"""
    corrs = []
    for i in range(len(factors)):
        main_i = factors[i]
        follows = factors[i + 1 :]
        corr = [show_corr(main_i, i, plt_plot=False) for i in follows]
        corr = [np.nan] * (i + 1) + corr
        corrs.append(corr)
    if factor_names is None:
        factor_names = [f"fac{i}" for i in list(range(1, len(factors) + 1))]
    corrs = pd.DataFrame(corrs, columns=factor_names, index=factor_names)
    np.fill_diagonal(corrs.to_numpy(), 1)
    if show_percent:
        pcorrs = corrs.applymap(to_percent)
    else:
        pcorrs = corrs.copy()
    if print_bool:
        print(pcorrs)
    return corrs


# 半衰期序列
def calc_exp_list(window, half_life):
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)


# weighted_std
def calcWeightedStd(series, weights):
    """
    加权平均std
    """
    weights /= np.sum(weights)
    return np.sqrt(np.sum((series - np.mean(series)) ** 2 * weights))


def other_periods_comments_nets(
    fac,
    period=None,
    way=None,
    comments_writer=None,
    nets_writer=None,
    sheetname=None,
    group_num=10,
):
    """不同频率下的评价指标，请输入行业市值中性化后的因子值"""
    closes = read_daily(open=1).shift(-1)
    fac1 = fac.stack()
    df = al.utils.get_clean_factor_and_forward_returns(
        fac1,
        closes[closes.index.isin(fac.index)],
        quantiles=group_num,
        periods=(period,),
    )
    df = df.reset_index()
    ics = df.groupby(["date"])[[f"{period}D", "factor"]].apply(
        lambda x: x.corr(method="spearman").iloc[0, 1]
    )
    ic = ics.mean()
    ir = ics.std()
    icir = ic / ir * (252**0.5) / (period**0.5)
    df = df.groupby(["date", "factor_quantile"])[f"{period}D"].mean() / period
    df = df.unstack()
    df.columns = [f"分组{i}" for i in list(df.columns)]
    if way == "pos":
        df = df.assign(多空对冲=df[f"分组{group_num}"] - df.分组1)
    elif way == "neg":
        df = df.assign(多空对冲=df.分组1 - df[f"分组{group_num}"])
    nets = (df + 1).cumprod()
    nets = nets.apply(lambda x: x / x.iloc[0])
    nets.plot(rot=60)
    plt.show()
    comments = comments_on_twins_periods(nets.多空对冲, df.多空对冲, period)
    comments = pd.concat(
        [pd.Series([ic, icir], index=["Rank IC", "Rank ICIR"]), comments]
    )
    print(comments)
    if sheetname is None:
        ...
    else:
        if comments_writer is None:
            ...
        else:
            comments.to_excel(comments_writer, sheetname)
        if nets_writer is None:
            ...
        else:
            nets.to_excel(nets_writer, sheetname)
    return comments, nets


def get_list_std(delta_sts):
    """同一天多个因子，计算这些因子在当天的标准差"""
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum(delta_sts_std)
    delta_sts_std = delta_sts_std**0.5 / len(delta_sts)
    return delta_sts_std


def get_industry_dummies(daily=False, monthly=False):
    """生成31个行业的哑变量矩阵，返回一个字典"""
    homeplace = HomePlace()
    if monthly:
        industry_dummy = pd.read_feather(
            homeplace.daily_data_file + "申万行业2021版哑变量.feather"
        )
        industry_dummy = (
            industry_dummy.set_index("date")
            .groupby("code")
            .resample("M")
            .last()
            .fillna(0)
            .drop(columns=["code"])
            .reset_index()
        )
    elif daily:
        industry_dummy = pd.read_feather(
            homeplace.daily_data_file + "申万行业2021版哑变量.feather"
        ).fillna(0)
    else:
        raise ValueError("您总得指定一个频率吧？🤒")
    ws = list(industry_dummy.columns)[2:]
    ress = {}
    for w in ws:
        df = industry_dummy[["date", "code", w]]
        df = df.pivot(index="date", columns="code", values=w)
        df = df.replace(0, np.nan)
        ress[w] = df
    return ress


class pure_moon:
    __slots__ = [
        "homeplace" "path_prefix",
        "codes_path",
        "tradedays_path",
        "ages_path",
        "sts_path",
        "states_path",
        "opens_path",
        "closes_path",
        "highs_path",
        "lows_path",
        "pricloses_path",
        "flowshares_path",
        "amounts_path",
        "turnovers_path",
        "factors_file",
        "sts_monthly_file",
        "states_monthly_file",
        "sts_monthly_by10_file",
        "states_monthly_by10_file",
        "factors",
        "codes",
        "tradedays",
        "ages",
        "amounts",
        "closes",
        "flowshares",
        "highs",
        "lows",
        "opens",
        "pricloses",
        "states",
        "sts",
        "turnovers",
        "sts_monthly",
        "states_monthly",
        "ages_monthly",
        "tris_monthly",
        "opens_monthly",
        "closes_monthly",
        "rets_monthly",
        "opens_monthly_shift",
        "rets_monthly_begin",
        "limit_ups",
        "limit_downs",
        "data",
        "ic_icir_and_rank",
        "rets_monthly_limit_downs",
        "group_rets",
        "long_short_rets",
        "long_short_net_values",
        "group_net_values",
        "long_short_ret_yearly",
        "long_short_vol_yearly",
        "long_short_info_ratio",
        "long_short_win_times",
        "long_short_win_ratio",
        "retreats",
        "max_retreat",
        "long_short_comments",
        "total_comments",
        "square_rets",
        "cap",
        "cap_value",
        "industry_dummy",
        "industry_codes",
        "industry_codes_str",
        "industry_ws",
        "factors_out",
        "pricloses_copy",
        "flowshares_copy",
    ]

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(cls):
        now = datetime.datetime.now()
        now = datetime.datetime.strftime(now, format="%Y-%m-%d %H:%M:%S")
        cls.homeplace = HomePlace()
        # logger.add('pure_moon'+now+'.log')
        # 绝对路径前缀
        cls.path_prefix = cls.homeplace.daily_data_file
        # 股票代码文件
        cls.codes_path = "AllStockCode.mat"
        # 交易日期文件
        cls.tradedays_path = "TradingDate_Daily.mat"
        # 上市天数文件
        cls.ages_path = "AllStock_DailyListedDate.mat"
        # st日子标志文件
        cls.sts_path = "AllStock_DailyST.mat"
        # 交易状态文件
        cls.states_path = "AllStock_DailyStatus.mat"
        # 复权开盘价数据文件
        cls.opens_path = "AllStock_DailyOpen_dividend.mat"
        # 复权收盘价数据文件
        cls.closes_path = "AllStock_DailyClose_dividend.mat"
        # 复权最高价数据文件
        cls.highs_path = "Allstock_DailyHigh_dividend.mat"
        # 复权最低价数据文件
        cls.lows_path = "Allstock_DailyLow_dividend.mat"
        # 不复权收盘价数据文件
        cls.pricloses_path = "AllStock_DailyClose.mat"
        # 流通股本数据文件
        cls.flowshares_path = "AllStock_DailyAShareNum.mat"
        # 成交量数据文件
        cls.amounts_path = "AllStock_DailyVolume.mat"
        # 换手率数据文件
        cls.turnovers_path = "AllStock_DailyTR.mat"
        # 因子数据文件
        cls.factors_file = ""
        # 已经算好的月度st状态文件
        cls.sts_monthly_file = "sts_monthly.feather"
        # 已经算好的月度交易状态文件
        cls.states_monthly_file = "states_monthly.feather"
        # 已经算好的月度st_by10状态文件
        cls.sts_monthly_by10_file = "sts_monthly_by10.feather"
        # 已经算好的月度交易状态文件
        cls.states_monthly_by10_file = "states_monthly_by10.feather"
        # 拼接绝对路径前缀和相对路径
        dirs = dir(cls)
        dirs.remove("new_path")
        dirs.remove("set_factor_file")
        dirs = [i for i in dirs if i.endswith("path")] + [
            i for i in dirs if i.endswith("file")
        ]
        dirs_values = list(map(lambda x, y: getattr(x, y), [cls] * len(dirs), dirs))
        dirs_values = list(
            map(lambda x, y: x + y, [cls.path_prefix] * len(dirs), dirs_values)
        )
        for attr, value in zip(dirs, dirs_values):
            setattr(cls, attr, value)

    def __call__(self, fallmount=0):
        """调用对象则返回因子值"""
        df = self.factors_out.copy()
        # df=df.set_index(['date', 'code']).unstack()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        if fallmount == 0:
            return df
        else:
            return pure_fallmount(df)

    @params_setter(slogan=None)
    # @lru_cache(maxsize=None)
    def set_factor_file(self, factors_file):
        """设置因子文件的路径，因子文件列名应为股票代码，索引为时间"""
        self.factors_file = factors_file
        self.factors = pd.read_feather(self.factors_file)
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

    @params_setter(slogan=None)
    # @lru_cache(maxsize=None)
    def set_factor_df_date_as_index(self, df):
        """设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间"""
        df = df.reset_index()
        df.columns = ["date"] + list(df.columns)[1:]
        # df.date=df.date.apply(self.next_month_end)
        # df=df.set_index('date')
        self.factors = df
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

    @params_setter(slogan=None)
    # @lru_cache(maxsize=None)
    def set_factor_df_wide(self, df):
        """从dataframe读入因子宽数据"""
        if isinstance(df, pure_fallmount):
            df = df()
        self.factors = df.copy()
        # self.factors.date=self.factors.date.apply(self.next_month_end)
        # self.factors=self.factors.set_index('date')
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

    # def set_factor_df_long(self,df):
    #     '''从dataframe读入因子长数据'''
    #     self.factors=df
    #     self.factors.columns=['date','code','fac']

    @classmethod
    @lru_cache(maxsize=None)
    @history_remain(slogan=None)
    def new_path(cls, **kwargs):
        """修改日频数据文件的路径，便于更新数据
        要修改的路径以字典形式传入，键为属性名，值为要设置的新路径"""
        for key, value in kwargs.items():
            setattr(cls, key, value)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def col_and_index(cls):
        """读取股票代码，作为未来表格的行名
        读取交易日历，作为未来表格的索引"""
        cls.codes = list(scio.loadmat(cls.codes_path).values())[3]
        cls.tradedays = list(scio.loadmat(cls.tradedays_path).values())[3].astype(str)
        cls.codes = cls.codes.flatten().tolist()
        # cls.tradedays = cls.tradedays.flatten().tolist()
        cls.codes = list(map(lambda x: x[0], cls.codes))
        cls.tradedays = cls.tradedays[0].tolist()

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def loadmat(cls, path):
        """重写一个加载mat文件的函数，以使代码更简洁"""
        return list(scio.loadmat(path).values())[3]

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def make_df(cls, data):
        """将读入的数据，和股票代码与时间拼接，做成dataframe"""
        data = pd.DataFrame(data, columns=cls.codes, index=cls.tradedays)
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
        return data

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def load_all_files(cls):
        """加全部的mat文件"""
        attrs = dir(cls)
        attrs = [i for i in attrs if i.endswith("path")]
        attrs.remove("codes_path")
        attrs.remove("tradedays_path")
        attrs.remove("new_path")
        for attr in attrs:
            new_attr = attr[:-5]
            setattr(cls, new_attr, cls.make_df(cls.loadmat(getattr(cls, attr))))
        cls.opens = cls.opens.replace(0, np.nan)
        cls.closes = cls.closes.replace(0, np.nan)
        cls.pricloses_copy = cls.pricloses.copy()
        cls.flowshares_copy = cls.flowshares.copy()

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_st(cls, df):
        """比较一个月内st的天数，如果st天数多，就删除本月，如果正常多，就保留本月"""
        st_count = len(df[df == 1])
        normal_count = len(df[df != 1])
        if st_count >= normal_count:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_st_by10(cls, df):
        """比较一个月内正常交易的天数，如果少于10天，就删除本月"""
        normal_count = len(df[df != 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_state(cls, df):
        """比较一个月内非正常交易的天数，如果非正常交易天数多，就删除本月，否则保留本月"""
        abnormal_count = len(df[df == 0])
        normal_count = len(df[df == 1])
        if abnormal_count >= normal_count:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_state_by10(cls, df):
        """比较一个月内正常交易天数，如果少于10天，就删除本月"""
        normal_count = len(df[df == 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def read_add(cls, pridf, df, func):
        """由于数据更新，过去计算的月度状态可能需要追加"""
        # if not STATES['NO_LOG']:
        #     logger.info(f'this is max_index of pridf{pridf.index.max()}')
        #     logger.info(f'this is max_index of df{df.index.max()}')
        if pridf.index.max() > df.index.max():
            df_add = pridf[pridf.index > df.index.max()]
            df_add = df_add.resample("M").apply(func)
            df = pd.concat([df, df_add])
            return df
        else:
            return df

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def write_feather(cls, df, path):
        """将算出来的数据存入本地，以免造成重复运算"""
        df1 = df.copy()
        df1 = df1.reset_index()
        df1.to_feather(path)

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def daily_to_monthly(cls, pridf, path, func):
        """把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算"""
        try:
            # if not STATES['NO_LOG']:
            #     logger.info('try to read the prepared state file')
            month_df = pd.read_feather(path).set_index("index")
            # if not STATES['NO_LOG']:
            #     logger.info('state file load success')
            month_df = cls.read_add(pridf, month_df, func)
            # if not STATES['NO_LOG']:
            #     logger.info('adding after state file has finish')
            cls.write_feather(month_df, path)
            # if not STATES['NO_LOG']:
            #     logger.info('the feather is new now')
        except Exception as e:
            if not STATES["NO_LOG"]:
                logger.error("error occurs when read state files")
                logger.error(e)
            print("state file rewriting……")
            month_df = pridf.resample("M").apply(func)
            cls.write_feather(month_df, path)
        return month_df

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def daily_to_monthly_by10(cls, pridf, path, func):
        """把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算"""
        try:
            month_df = pd.read_feather(path).set_index("date")
            month_df = cls.read_add(pridf, month_df, func)
            cls.write_feather(month_df, path)
        except Exception:
            print("rewriting")
            month_df = pridf.resample("M").apply(func)
            cls.write_feather(month_df, path)
        return month_df

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def judge_month(cls):
        """生成一个月综合判断的表格"""
        cls.sts_monthly = cls.daily_to_monthly(
            cls.sts, cls.sts_monthly_file, cls.judge_month_st
        )
        cls.states_monthly = cls.daily_to_monthly(
            cls.states, cls.states_monthly_file, cls.judge_month_state
        )
        cls.ages_monthly = cls.ages.resample("M").last()
        cls.ages_monthly = np.sign(cls.ages_monthly.applymap(lambda x: x - 60)).replace(
            -1, 0
        )
        cls.tris_monthly = cls.sts_monthly * cls.states_monthly * cls.ages_monthly
        cls.tris_monthly = cls.tris_monthly.replace(0, np.nan)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def judge_month_by10(cls):
        """生成一个月综合判断的表格"""
        cls.sts_monthly = cls.daily_to_monthly(
            cls.sts, cls.sts_monthly_by10_file, cls.judge_month_st_by10
        )
        cls.states_monthly = cls.daily_to_monthly(
            cls.states, cls.states_monthly_by10_file, cls.judge_month_state_by10
        )
        cls.ages_monthly = cls.ages.resample("M").last()
        cls.ages_monthly = np.sign(cls.ages_monthly.applymap(lambda x: x - 60)).replace(
            -1, 0
        )
        cls.tris_monthly = cls.sts_monthly * cls.states_monthly * cls.ages_monthly
        cls.tris_monthly = cls.tris_monthly.replace(0, np.nan)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def get_rets_month(cls):
        """计算每月的收益率，并根据每月做出交易状态，做出删减"""
        cls.opens_monthly = cls.opens.resample("M").first()
        cls.closes_monthly = cls.closes.resample("M").last()
        cls.rets_monthly = (cls.closes_monthly - cls.opens_monthly) / cls.opens_monthly
        cls.rets_monthly = cls.rets_monthly * cls.tris_monthly
        cls.rets_monthly = cls.rets_monthly.stack().reset_index()
        cls.rets_monthly.columns = ["date", "code", "ret"]

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def neutralize_factors(cls, df):
        """组内对因子进行市值中性化"""
        industry_codes = list(df.columns)
        industry_codes = [i for i in industry_codes if i.startswith("w")]
        industry_codes_str = "+".join(industry_codes)
        ols_result = smf.ols("fac~cap_size+" + industry_codes_str, data=df).fit()
        ols_w = ols_result.params["cap_size"]
        ols_b = ols_result.params["Intercept"]
        ols_bs = {}
        for ind in industry_codes:
            ols_bs[ind] = ols_result.params[ind]
        df.fac = df.fac - ols_w * df.cap_size - ols_b
        for k, v in ols_bs.items():
            df.fac = df.fac - v * df[k]
        df = df[["fac"]]
        return df

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def get_log_cap(cls, boxcox=False):
        """获得对数市值"""
        try:
            cls.pricloses = cls.pricloses.replace(0, np.nan)
            cls.flowshares = cls.flowshares.replace(0, np.nan)
            cls.pricloses = cls.pricloses.resample("M").last()
        except Exception:
            cls.pricloses = cls.pricloses_copy.copy()
            cls.pricloses = cls.pricloses.replace(0, np.nan)
            cls.flowshares = cls.flowshares.replace(0, np.nan)
            cls.pricloses = cls.pricloses.resample("M").last()
        cls.pricloses = cls.pricloses.stack().reset_index()
        cls.pricloses.columns = ["date", "code", "priclose"]
        try:
            cls.flowshares = cls.flowshares.resample("M").last()
        except Exception:
            cls.flowshares = cls.flowshares_copy.copy()
            cls.flowshares = cls.flowshares.resample("M").last()
        cls.flowshares = cls.flowshares.stack().reset_index()
        cls.flowshares.columns = ["date", "code", "flowshare"]
        cls.flowshares = pd.merge(cls.flowshares, cls.pricloses, on=["date", "code"])
        cls.cap = cls.flowshares.assign(
            cap_size=cls.flowshares.flowshare * cls.flowshares.priclose
        )
        if boxcox:

            def single(x):
                x.cap_size = ss.boxcox(x.cap_size)[0]
                return x

            cls.cap = cls.cap.groupby(["date"]).apply(single)
        else:
            cls.cap["cap_size"] = np.log(cls.cap["cap_size"])

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_neutral_factors(self):
        """对因子进行市值中性化"""
        self.factors = self.factors.set_index("date")
        self.factors.index = self.factors.index + pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        last_date = self.tris_monthly.index.max() + pd.DateOffset(months=1)
        last_date = last_date + pd.tseries.offsets.MonthEnd()
        add_tail = pd.DataFrame(1, index=[last_date], columns=self.tris_monthly.columns)
        tris_monthly = pd.concat([self.tris_monthly, add_tail])
        self.factors = self.factors * tris_monthly
        self.factors.index = self.factors.index - pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.factors = pd.merge(
            self.factors, self.cap, how="inner", on=["date", "code"]
        )
        self.industry_dummy = (
            pd.read_feather(self.homeplace.daily_data_file + "申万行业2021版哑变量.feather")
            .set_index("date")
            .groupby("code")
            .resample("M")
            .last()
        )
        self.industry_dummy = self.industry_dummy.drop(columns=["code"]).reset_index()
        self.industry_ws = [f"w{i}" for i in range(1, self.industry_dummy.shape[1] - 1)]
        col = ["code", "date"] + self.industry_ws
        self.industry_dummy.columns = col
        self.factors = pd.merge(self.factors, self.industry_dummy, on=["date", "code"])
        self.factors = self.factors.set_index(["date", "code"])
        self.factors = self.factors.groupby(["date"]).apply(self.neutralize_factors)
        self.factors = self.factors.reset_index()

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def deal_with_factors(self):
        """删除不符合交易条件的因子数据"""
        self.factors = self.factors.set_index("date")
        self.factors_out = self.factors.copy()
        self.factors.index = self.factors.index + pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors * self.tris_monthly
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def deal_with_factors_after_neutralize(self):
        """中性化之后的因子处理方法"""
        self.factors = self.factors.set_index(["date", "code"])
        self.factors = self.factors.unstack()
        self.factors_out = self.factors.copy()
        self.factors.index = self.factors.index + pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        self.factors.columns = list(map(lambda x: x[1], list(self.factors.columns)))
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def find_limit(cls, df, up=1):
        """计算涨跌幅超过9.8%的股票，并将其存储进一个长列表里
        其中时间列，为某月的最后一天；涨停日虽然为下月初第一天，但这里标注的时间统一为上月最后一天"""
        limit_df = np.sign(df.applymap(lambda x: x - up * 0.098)).replace(
            -1 * up, np.nan
        )
        limit_df = limit_df.stack().reset_index()
        limit_df.columns = ["date", "code", "limit_up_signal"]
        limit_df = limit_df[["date", "code"]]
        return limit_df

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def get_limit_ups_downs(cls):
        """找月初第一天就涨停"""
        """或者是月末跌停的股票"""
        cls.opens_monthly_shift = cls.opens_monthly.copy()
        cls.opens_monthly_shift = cls.opens_monthly_shift.shift(-1)
        cls.rets_monthly_begin = (
            cls.opens_monthly_shift - cls.closes_monthly
        ) / cls.closes_monthly
        cls.closes2_monthly = cls.closes.shift(1).resample("M").last()
        cls.rets_monthly_last = (
            cls.closes_monthly - cls.closes2_monthly
        ) / cls.closes2_monthly
        cls.limit_ups = cls.find_limit(cls.rets_monthly_begin, up=1)
        cls.limit_downs = cls.find_limit(cls.rets_monthly_last, up=-1)

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_ic_rankic(cls, df):
        """计算IC和RankIC"""
        df1 = df[["ret", "fac"]]
        ic = df1.corr(method="pearson").iloc[0, 1]
        rankic = df1.corr(method="spearman").iloc[0, 1]
        df2 = pd.DataFrame({"ic": [ic], "rankic": [rankic]})
        return df2

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_icir_rankicir(cls, df):
        """计算ICIR和RankICIR"""
        ic = df.ic.mean()
        rankic = df.rankic.mean()
        icir = ic / np.std(df.ic) * (12 ** (0.5))
        rankicir = rankic / np.std(df.rankic) * (12 ** (0.5))
        return pd.DataFrame(
            {"IC": [ic], "ICIR": [icir], "RankIC": [rankic], "RankICIR": [rankicir]},
            index=["评价指标"],
        )

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_ic_icir_and_rank(cls, df):
        """计算IC、ICIR、RankIC、RankICIR"""
        df1 = df.groupby("date").apply(cls.get_ic_rankic)
        df2 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        dura = (df.date.max() - df.date.min()).days / 365
        t_value = df2.iloc[3, 0] * (dura ** (1 / 2))
        df3 = pd.DataFrame({"评价指标": [t_value]}, index=["RankIC均值t值"])
        df4 = pd.concat([df2, df3])
        return df4

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_groups(cls, df, groups_num):
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

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    # @history_remain(slogan='abandoned')
    def next_month_end(cls, x):
        """找到下个月最后一天"""
        x1 = x = x + relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def limit_old_to_new(cls, limit, data):
        """获取跌停股在旧月的组号，然后将日期调整到新月里
        涨停股则获得新月里涨停股的代码和时间，然后直接删去"""
        data1 = data.copy()
        data1 = data1.reset_index()
        data1.columns = ["data_index"] + list(data1.columns)[1:]
        old = pd.merge(limit, data1, how="inner", on=["date", "code"])
        old = old.set_index("data_index")
        old = old[["group", "date", "code"]]
        old.date = list(map(cls.next_month_end, list(old.date)))
        return old

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_data(self, groups_num):
        """拼接因子数据和每月收益率数据，并对涨停和跌停股加以处理"""
        self.data = pd.merge(
            self.rets_monthly, self.factors, how="inner", on=["date", "code"]
        )
        self.ic_icir_and_rank = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby("date").apply(
            lambda x: self.get_groups(x, groups_num)
        )
        self.data = self.data.reset_index(drop=True)
        limit_ups_object = self.limit_old_to_new(self.limit_ups, self.data)
        limit_downs_object = self.limit_old_to_new(self.limit_downs, self.data)
        self.data = self.data.drop(limit_ups_object.index)
        rets_monthly_limit_downs = pd.merge(
            self.rets_monthly, limit_downs_object, how="inner", on=["date", "code"]
        )
        self.data = pd.concat([self.data, rets_monthly_limit_downs])

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def select_data_time(self, time_start, time_end):
        """筛选特定的时间段"""
        if time_start:
            self.data = self.data[self.data.date >= time_start]
        if time_end:
            self.data = self.data[self.data.date <= time_end]

    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def make_start_to_one(self, l):
        """让净值序列的第一个数变成1"""
        min_date = self.factors.date.min()
        add_date = min_date - relativedelta(days=min_date.day)
        add_l = pd.Series([1], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def to_group_ret(self, l):
        """每一组的年化收益率"""
        ret = l[-1] ** (12 / len(l)) - 1
        return ret

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_group_rets_net_values(self, groups_num=10, value_weighted=False):
        """计算组内每一期的平均收益，生成每日收益率序列和净值序列"""
        if value_weighted:
            cap_value = self.pricloses_copy * self.flowshares_copy
            cap_value = cap_value.resample("M").last().shift(1)
            cap_value = cap_value * self.tris_monthly
            # cap_value=np.log(cap_value)
            cap_value = cap_value.stack().reset_index()
            cap_value.columns = ["date", "code", "cap_value"]
            self.data = pd.merge(self.data, cap_value, on=["date", "code"])

            def in_g(df):
                df.cap_value = df.cap_value / df.cap_value.sum()
                df.ret = df.ret * df.cap_value
                return df.ret.sum()

            self.group_rets = self.data.groupby(["date", "group"]).apply(in_g)
        else:
            self.group_rets = self.data.groupby(["date", "group"]).apply(
                lambda x: x.ret.mean()
            )
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[
            self.group_rets.index <= self.factors.date.max()
        ]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix("group")
        self.long_short_rets = (
            self.group_rets["group1"] - self.group_rets["group" + str(groups_num)]
        )
        self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = (
                self.group_rets["group" + str(groups_num)] - self.group_rets["group1"]
            )
            self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        self.long_short_net_values = self.make_start_to_one(self.long_short_net_values)
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.applymap(lambda x: x + 1)
        self.group_net_values = self.group_net_values.cumprod()
        self.group_net_values = self.group_net_values.apply(self.make_start_to_one)
        a = groups_num ** (0.5)
        # 判断是否要两个因子画表格
        if a == int(a):
            self.square_rets = (
                self.group_net_values.iloc[:, :-1].apply(self.to_group_ret).to_numpy()
            )
            self.square_rets = self.square_rets.reshape((int(a), int(a)))
            self.square_rets = pd.DataFrame(
                self.square_rets,
                columns=list(range(1, int(a) + 1)),
                index=list(range(1, int(a) + 1)),
            )
            print("这是self.square_rets", self.square_rets)

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_long_short_comments(self, on_paper=False):
        """计算多空对冲的相关评价指标
        包括年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        self.long_short_ret_yearly = (
            self.long_short_net_values[-1] ** (12 / len(self.long_short_net_values)) - 1
        )
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (12**0.5)
        self.long_short_info_ratio = (
            self.long_short_ret_yearly / self.long_short_vol_yearly
        )
        self.long_short_win_times = len(self.long_short_rets[self.long_short_rets > 0])
        self.long_short_win_ratio = self.long_short_win_times / len(
            self.long_short_rets
        )
        self.max_retreat = -(
            self.long_short_net_values / self.long_short_net_values.expanding(1).max()
            - 1
        ).min()
        if on_paper:
            self.long_short_comments = pd.DataFrame(
                {
                    "评价指标": [
                        self.long_short_ret_yearly,
                        self.long_short_vol_yearly,
                        self.long_short_info_ratio,
                        self.long_short_win_ratio,
                        self.max_retreat,
                    ]
                },
                index=["年化收益率", "年化波动率", "收益波动比", "月度胜率", "最大回撤率"],
            )
        else:
            self.long_short_comments = pd.DataFrame(
                {
                    "评价指标": [
                        self.long_short_ret_yearly,
                        self.long_short_vol_yearly,
                        self.long_short_info_ratio,
                        self.long_short_win_ratio,
                        self.max_retreat,
                    ]
                },
                index=["年化收益率", "年化波动率", "信息比率", "月度胜率", "最大回撤率"],
            )

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_total_comments(self):
        """综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        self.total_comments = pd.concat(
            [self.ic_icir_and_rank, self.long_short_comments]
        )

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def plot_net_values(self, y2, filename):
        """使用matplotlib来画图，y2为是否对多空组合采用双y轴"""
        self.group_net_values.plot(secondary_y=y2, rot=60)
        filename_path = filename + ".png"
        if not STATES["NO_SAVE"]:
            plt.savefig(filename_path)

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def plotly_net_values(self, filename):
        """使用plotly.express画图"""
        fig = pe.line(self.group_net_values)
        filename_path = filename + ".html"
        pio.write_html(fig, filename_path, auto_open=True)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def prerpare(cls):
        """通用数据准备"""
        cls.col_and_index()
        cls.load_all_files()
        cls.judge_month()
        cls.get_rets_month()

    # @lru_cache(maxsize=None)
    @kk.desktop_sender(title="嘿，回测结束啦～🗓")
    def run(
        self,
        groups_num=10,
        neutralize=False,
        boxcox=False,
        value_weighted=False,
        y2=False,
        plt_plot=True,
        plotly_plot=False,
        filename="分组净值图",
        time_start=None,
        time_end=None,
        print_comments=True,
        comments_writer=None,
        net_values_writer=None,
        rets_writer=None,
        comments_sheetname=None,
        net_values_sheetname=None,
        rets_sheetname=None,
        on_paper=False,
        sheetname=None,
    ):
        """运行回测部分"""
        if comments_writer and not (comments_sheetname or sheetname):
            raise IOError("把total_comments输出到excel中时，必须指定sheetname🤒")
        if net_values_writer and not (net_values_sheetname or sheetname):
            raise IOError("把group_net_values输出到excel中时，必须指定sheetname🤒")
        if rets_writer and not (rets_sheetname or sheetname):
            raise IOError("把group_rets输出到excel中时，必须指定sheetname🤒")
        if neutralize:
            self.get_log_cap()
            self.get_neutral_factors()
            self.deal_with_factors_after_neutralize()
        elif boxcox:
            self.get_log_cap(boxcox=True)
            self.get_neutral_factors()
            self.deal_with_factors_after_neutralize()
        else:
            self.deal_with_factors()
        self.get_limit_ups_downs()
        self.get_data(groups_num)
        self.select_data_time(time_start, time_end)
        self.get_group_rets_net_values(
            groups_num=groups_num, value_weighted=value_weighted
        )
        self.get_long_short_comments(on_paper=on_paper)
        self.get_total_comments()
        if plt_plot:
            if not STATES["NO_PLOT"]:
                if filename:
                    self.plot_net_values(y2=y2, filename=filename)
                else:
                    self.plot_net_values(
                        y2=y2,
                        filename=self.factors_file.split(".")[-2].split("/")[-1]
                        + str(groups_num)
                        + "分组",
                    )
                plt.show()
        if plotly_plot:
            if not STATES["NO_PLOT"]:
                if filename:
                    self.plotly_net_values(filename=filename)
                else:
                    self.plotly_net_values(
                        filename=self.factors_file.split(".")[-2].split("/")[-1]
                        + str(groups_num)
                        + "分组"
                    )
        if print_comments:
            if not STATES["NO_COMMENT"]:
                print(self.total_comments)
        if sheetname:
            if comments_writer:
                total_comments = self.total_comments.copy()
                tc = list(total_comments.评价指标)
                tc[0] = str(round(tc[0] * 100, 2)) + "%"
                tc[1] = str(round(tc[1], 2))
                tc[2] = str(round(tc[2] * 100, 2)) + "%"
                tc[3] = str(round(tc[3], 2))
                tc[4] = str(round(tc[4], 2))
                tc[5] = str(round(tc[5] * 100, 2)) + "%"
                tc[6] = str(round(tc[6] * 100, 2)) + "%"
                tc[7] = str(round(tc[7], 2))
                tc[8] = str(round(tc[8] * 100, 2)) + "%"
                tc[9] = str(round(tc[9] * 100, 2)) + "%"
                new_total_comments = pd.DataFrame(
                    {sheetname: tc}, index=total_comments.index
                )
                new_total_comments.T.to_excel(comments_writer, sheet_name=sheetname)
            if net_values_writer:
                groups_net_values = self.group_net_values.copy()
                groups_net_values.index = groups_net_values.index.strftime("%Y/%m/%d")
                groups_net_values.columns = [
                    f"分组{i}" for i in range(1, len(list(groups_net_values.columns)))
                ] + ["多空对冲（右轴）"]
                groups_net_values.to_excel(net_values_writer, sheet_name=sheetname)
            if rets_writer:
                group_rets = self.group_rets.copy()
                group_rets.index = group_rets.index.strftime("%Y/%m/%d")
                group_rets.columns = [
                    f"分组{i}" for i in range(1, len(list(group_rets.columns)))
                ] + ["多空对冲（右轴）"]
                group_rets.to_excel(rets_writer, sheet_name=sheetname)
        else:
            if comments_writer and comments_sheetname:
                total_comments = self.total_comments.copy()
                tc = list(total_comments.评价指标)
                tc[0] = str(round(tc[0] * 100, 2)) + "%"
                tc[1] = str(round(tc[1], 2))
                tc[2] = str(round(tc[2] * 100, 2)) + "%"
                tc[3] = str(round(tc[3], 2))
                tc[4] = str(round(tc[4], 2))
                tc[5] = str(round(tc[5] * 100, 2)) + "%"
                tc[6] = str(round(tc[6] * 100, 2)) + "%"
                tc[7] = str(round(tc[7], 2))
                tc[8] = str(round(tc[8] * 100, 2)) + "%"
                tc[9] = str(round(tc[9] * 100, 2)) + "%"
                new_total_comments = pd.DataFrame(
                    {comments_sheetname: tc}, index=total_comments.index
                )
                new_total_comments.T.to_excel(
                    comments_writer, sheet_name=comments_sheetname
                )
            if net_values_writer and net_values_sheetname:
                groups_net_values = self.group_net_values.copy()
                groups_net_values.index = groups_net_values.index.strftime("%Y/%m/%d")
                groups_net_values.columns = [
                    f"分组{i}" for i in range(1, len(list(groups_net_values.columns)))
                ] + ["多空对冲（右轴）"]
                groups_net_values.to_excel(
                    net_values_writer, sheet_name=net_values_sheetname
                )
            if rets_writer and rets_sheetname:
                group_rets = self.group_rets.copy()
                group_rets.index = group_rets.index.strftime("%Y/%m/%d")
                group_rets.columns = [
                    f"分组{i}" for i in range(1, len(list(group_rets.columns)))
                ] + ["多空对冲（右轴）"]
                group_rets.to_excel(rets_writer, sheet_name=rets_sheetname)


class pure_fall:
    # DONE：修改为因子文件名可以带“日频_“，也可以不带“日频_“
    def __init__(
        self,
        minute_files_path=None,
        minute_columns=["date", "open", "high", "low", "close", "amount", "money"],
        daily_factors_path=None,
        monthly_factors_path=None,
    ):
        self.homeplace = HomePlace()
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path = minute_files_path
        else:
            self.minute_files_path = self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files = os.listdir(self.minute_files_path)
        self.minute_files = [i for i in self.minute_files if i.endswith(".mat")]
        self.minute_files = sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns = minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list = []
        # 更新数据用的列表
        self.daily_factors_list_update = []
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        # 最终月度因子表格
        self.monthly_factors = None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path = daily_factors_path
        else:
            self.daily_factors_path = self.homeplace.factor_data_file + "日频_"
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path = monthly_factors_path
        else:
            self.monthly_factors_path = self.homeplace.factor_data_file + "月频_"

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """将几个因子截面标准化之后，因子值相加"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __truediv__(self, selfa):
        """两个一正一副的因子，可以用此方法相减"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac = fac1 - fac2
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __floordiv__(self, selfa):
        """两个因子一正一负，可以用此方法相除"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2 = fac2 - fac2.min()
        fac = fac1 / fac2
        fac = fac.replace(np.inf, np.nan)
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    @kk.desktop_sender(title="嘿，正交化结束啦～🐬")
    def __sub__(self, selfa):
        """用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可"""
        tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def wide_to_long(self, df, i):
        """将宽数据转化为长数据，用于因子表转化和拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", i]
        df = df.set_index(["date", "code"])
        return df

    def de_in_group(self, df, help_names):
        """对每个时间，分别做回归，剔除相关因子"""
        ols_order = "fac~" + "+".join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params["Intercept"]
        df = df[["fac"]]
        return df

    def mat_to_df(self, mat, use_datetime=True):
        """将mat文件变成"""
        mat_path = "/".join([self.minute_files_path, mat])
        df = list(scio.loadmat(mat_path).values())[3]
        df = pd.DataFrame(df, columns=self.minute_columns)
        if use_datetime:
            df.date = pd.to_datetime(df.date.apply(str), format="%Y%m%d")
            df = df.set_index("date")
        return df

    def add_suffix(self, code):
        """给股票代码加上后缀"""
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

    def minute_to_daily(
        self,
        func,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
        update=0,
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """

        if add_priclose:
            for mat in tqdm.tqdm(
                self.minute_files, desc="来日纵使千千阙歌，飘于远方我路上；来日纵使千千晚星，亮过今晚月亮。都不及今宵这刻美丽🌙"
            ):
                try:
                    code = self.add_suffix(mat[-10:-4])
                    self.code = code
                    df = self.mat_to_df(mat, use_datetime=True)
                    if add_tr:
                        share = read_daily("AllStock_DailyAShareNum.mat")
                        share_this = share[code].to_frame("sharenum").reset_index()
                        share_this.columns = ["date", "sharenum"]
                        df = df.reset_index()
                        df.columns = ["date"] + list(df.columns)[1:]
                        df = pd.merge(df, share_this, on=["date"], how="left")
                        df = df.assign(tr=df.amount / df.sharenum)
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df.date = df.date.dt.strftime("%Y%m%d")
                    df.date = df.date.astype(int)
                    df = df[(df.date >= start_date) & (df.date <= end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    priclose = df.groupby("date").last()
                    priclose = priclose.shift(1).reset_index()
                    df = pd.concat([priclose, df])
                    the_func = partial(func)
                    df = df.groupby("date").apply(the_func)
                    df = df.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(df)
                    else:
                        self.daily_factors_list_update.append(df)
                except Exception as e:
                    if not STATES["NO_LOG"]:
                        logger.warning(f"{code} 缺失")
                        logger.error(e)
        else:
            for mat in tqdm.tqdm(
                self.minute_files, desc="来日纵使千千阙歌，飘于远方我路上；来日纵使千千晚星，亮过今晚月亮。都不及今宵这刻美丽🌙"
            ):
                try:
                    code = self.add_suffix(mat[-10:-4])
                    self.code = code
                    df = self.mat_to_df(mat, use_datetime=True)
                    if add_tr:
                        share = read_daily("AllStock_DailyAShareNum.mat")
                        share_this = share[code].to_frame("sharenum").reset_index()
                        share_this.columns = ["date", "sharenum"]
                        df = df.reset_index()
                        df.columns = ["date"] + list(df.columns)[1:]
                        df = pd.merge(df, share_this, on=["date"], how="left")
                        df = df.assign(tr=df.amount / df.sharenum)
                    the_func = partial(func)
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df.date = df.date.dt.strftime("%Y%m%d")
                    df.date = df.date.astype(int)
                    df = df[(df.date >= start_date) & (df.date <= end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    df = df.groupby("date").apply(the_func)
                    df = df.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(df)
                    else:
                        self.daily_factors_list_update.append(df)
                except Exception as e:
                    if not STATES["NO_LOG"]:
                        logger.warning(f"{code} 缺失")
                        logger.error(e)
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors_update.index = pd.to_datetime(
                self.daily_factors_update.index.astype(int), format="%Y%m%d"
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
            self.daily_factors.index = pd.to_datetime(
                self.daily_factors.index.astype(int), format="%Y%m%d"
            )
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def minute_to_daily_whole(
        self, func, start_date=10000000, end_date=30000000, update=0
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """
        for mat in tqdm.tqdm(self.minute_files):
            self.code = self.add_suffix(mat[-10:-4])
            df = self.mat_to_df(mat)
            df.date = df.date.astype(int)
            df = df[(df.date >= start_date) & (df.date <= end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func = partial(func)
            df = func(df)
            if isinstance(df, pd.DataFrame):
                df.columns = [self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df, pd.Series):
                df = df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not STATES["NO_LOG"]:
                    logger.warning(f"df is {df}")
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func,
        whole=False,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
    ):
        """调用分钟到日度方法，算出日频数据"""
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index("date")
            sql = sqlConfig("minute_data_alter")
            now_minute_data = sql.show_tables(full=False)[-1]
            now_minute_data = pd.Timestamp(now_minute_data)
            if self.daily_factors.index.max() < now_minute_data:
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data}，开始更新……"
                    )
                start_date_update = int(
                    datetime.datetime.strftime(
                        self.daily_factors.index.max() + pd.Timedelta("1 day"), "%Y%m%d"
                    )
                )
                end_date_update = int(
                    datetime.datetime.strftime(now_minute_data, "%Y%m%d")
                )
                if whole:
                    self.minute_to_daily_whole(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
                else:
                    self.minute_to_daily(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
        except Exception:
            if whole:
                self.minute_to_daily_whole(
                    func, start_date=start_date, end_date=end_date
                )
            else:
                self.minute_to_daily(
                    func,
                    add_priclose=add_priclose,
                    add_tr=add_tr,
                    start_date=start_date,
                    end_date=end_date,
                )

    def get_single_day_factor(self, func, day: int) -> pd.DataFrame:
        """计算单日的因子值，通过sql数据库，读取单日的数据，然后计算因子值"""
        sql = sqlConfig("minute_data_alter")
        df = sql.get_data(str(day))
        the_func = partial(func)
        df = df.groupby(["code"]).apply(the_func).to_frame()
        df.columns = [str(day)]
        df = df.T
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        return df

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors_alter(
        self, func, start_date=10000000, end_date=30000000
    ) -> pd.DataFrame:
        """通过minute_data_alter数据库一天一天计算因子值"""
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index("date")
            sql = sqlConfig("minute_data_stock_alter")
            now_minute_datas = sql.show_tables(full=False)
            now_minute_data = now_minute_datas[-1]
            now_minute_data = pd.Timestamp(now_minute_data)
            if self.daily_factors.index.max() < now_minute_data:
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data}，开始更新……"
                    )
                old_end = datetime.datetime.strftime(
                    self.daily_factors.index.max(), "%Y%m%d"
                )
                now_minute_datas = [i for i in now_minute_datas if i > old_end]
                dfs = []
                for c in tqdm.tqdm(now_minute_datas, desc="桂棹兮兰桨，击空明兮遂流光🌊"):
                    df = self.get_single_day_factor(func, c)
                    dfs.append(df)
                dfs = pd.concat(dfs)
                dfs = dfs.sort_index()
                self.daily_factors = pd.concat([self.daily_factors, dfs])
                self.daily_factors = self.daily_factors.dropna(how="all")
                self.daily_factors = self.daily_factors[
                    self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
                ]
                self.daily_factors.reset_index().to_feather(self.daily_factors_path)
                if not STATES["NO_LOG"]:
                    logger.success("更新已完成")

        except Exception:
            self.minute_to_daily(func, start_date=start_date, end_date=end_date)

    def get_neutral_monthly_factors(self, df, boxcox=False):
        """对月度因子做市值中性化处理"""
        shen = pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5, boxcox=True, plt=False, print_comments=False)
        else:
            shen.run(5, neutralize=True, plt=False, print_comments=False)
        new_factors = shen.factors.copy()
        new_factors = new_factors.set_index(["date", "code"]).unstack()
        new_factors.columns = list(map(lambda x: x[1], list(new_factors.columns)))
        new_factors = new_factors.reset_index()
        add_start_point = new_factors.date.min()
        add_start_point = add_start_point - pd.Timedelta(days=add_start_point.day)
        new_factors.date = new_factors.date.shift(1)
        new_factors.date = new_factors.date.fillna(add_start_point)
        new_factors = new_factors.set_index("date")
        return new_factors

    def get_monthly_factors(self, func, neutralize, boxcox):
        """将日频的因子转化为月频因子"""
        two_parts = self.monthly_factors_path.split(".")
        try:
            self.monthly_factors = pd.read_feather(self.monthly_factors_path)
            self.monthly_factors = self.monthly_factors.set_index("date")
        except Exception:
            the_func = partial(func)
            self.monthly_factors = the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors
                )
            elif boxcox:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors, boxcox=True
                )

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(
        self,
        whole=False,
        daily_func=None,
        monthly_func=None,
        neutralize=False,
        boxcox=False,
    ):
        """执必要的函数，将分钟数据变成月度因子"""
        self.get_daily_factors(daily_func, whole)
        self.get_monthly_factors(monthly_func, neutralize, boxcox)


class pure_fall_frequent(object):
    """对单只股票单日进行操作"""

    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
    ):
        self.kind = kind
        # 连接clickhouse
        self.chc = ClickHouseClient(f"minute_data")
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = pd.read_feather(self.factor_file)
            factor_old.columns = ["date"] + list(factor_old.columns)[1:]
            factor_old = factor_old.set_index("date")
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
        self.dates_new = sorted([i for i in dates_all if i not in self.dates_old])
        if len(self.dates_old) == 0:
            ...
        else:
            self.dates_new = self.dates_new[1:]

    def __call__(self):
        return self.factor.copy()

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func,
        fields: str = "*",
        chunksize: int = 250,
        show_time=True,
        tqdm_inside=True,
    ):
        """对单只股票单日进行操作"""
        the_func = partial(func)
        # 将需要更新的日子分块，每200天一组，一起运算
        dates_new_len = len(self.dates_new)
        if dates_new_len > 1:
            cut_points = list(range(0, dates_new_len, chunksize)) + [dates_new_len - 1]
            if cut_points[-1] == cut_points[-2]:
                cut_points = cut_points[:-1]
            cuts = tuple(zip(cut_points[:-1], cut_points[1:]))
            print(f"共{len(cuts)}段")
            self.cut_points = cut_points
            self.factor_new = []
            if tqdm_inside:
                # 开始计算因子值
                for date1, date2 in cuts:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    df = ((df.set_index("code")) / 100).reset_index()
                    tqdm.tqdm.pandas()
                    df = df.groupby(["date", "code"]).progress_apply(the_func)
                    df = df.to_frame("fac").reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            else:
                # 开始计算因子值
                for date1, date2 in tqdm.tqdm(cuts, desc="不知乘月几人归，落月摇情满江树。"):
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    df = ((df.set_index("code")) / 100).reset_index()
                    df = df.groupby(["date", "code"]).apply(the_func)
                    df = df.to_frame("fac").reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            self.factor_new = pd.concat(self.factor_new)
            # 拼接新的和旧的
            self.factor = pd.concat([self.factor_old, self.factor_new]).sort_index()
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"截止到{new_end_date}的因子值计算完了")
        elif dates_new_len == 1:
            print("共1天")
            if tqdm_inside:
                # 开始计算因子值
                sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                df = ((df.set_index("code")) / 100).reset_index()
                tqdm.tqdm.pandas()
                df = df.groupby(["date", "code"]).progress_apply(the_func)
                df = df.to_frame("fac").reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            else:
                # 开始计算因子值
                sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                df = ((df.set_index("code")) / 100).reset_index()
                df = df.groupby(["date", "code"]).apply(the_func)
                df = df.to_frame("fac").reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            self.factor_new = df
            # 拼接新的和旧的
            self.factor = pd.concat([self.factor_old, self.factor_new]).sort_index()
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"补充{self.dates_new[0]}截止到{new_end_date}的因子值计算完了")
        else:
            self.factor = self.factor_old
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")


class pure_fall_flexible(object):
    """对一段时间的截面数据进行操作，在get_daily_factors的func函数中，请写入df=df.groupby([xxx]).apply(fff)之类的语句，
    然后单独定义一个函数，作为要apply的fff，可以在apply上加进度条"""

    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
    ):
        self.kind = kind
        # 连接clickhouse
        self.chc = ClickHouseClient(f"minute_data")
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = pd.read_feather(self.factor_file)
            factor_old.columns = ["date"] + list(factor_old.columns)[1:]
            factor_old = factor_old.set_index("date")
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

    def __call__(self):
        return self.factor.copy()

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func,
        fields: str = "*",
        chunksize: int = 250,
        show_time=True,
        tqdm_inside=True,
    ):
        """对一段时间的截面数据进行操作"""
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
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    df = ((df.set_index("code")) / 100).reset_index()
                    tqdm.tqdm.pandas()
                    df = the_func(df)
                    if isinstance(df, pd.Series):
                        df = df.reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            else:
                # 开始计算因子值
                for date1, date2 in tqdm.tqdm(cut_points, desc="不知乘月几人归，落月摇情满江树。"):
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
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
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"截止到{new_end_date}的因子值计算完了")
        elif dates_new_len == 1:
            print("共1天")
            if tqdm_inside:
                # 开始计算因子值
                sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                df = ((df.set_index("code")) / 100).reset_index()
                tqdm.tqdm.pandas()
                df = the_func(df)
                if isinstance(df, pd.Series):
                    df = df.reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            else:
                # 开始计算因子值
                sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
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
            self.factor = pd.concat([self.factor_old, self.factor_new]).sort_index()
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"补充{self.dates_new[0]}截止到{new_end_date}的因子值计算完了")
        else:
            self.factor = self.factor_old
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")


class pure_sunbath:
    def __init__(
        self,
        minute_files_path=None,
        minute_columns=["date", "open", "high", "low", "close", "amount", "money"],
        daily_factors_path=None,
        monthly_factors_path=None,
    ):
        self.homeplace = HomePlace()
        print("在浴室的温暖里，连错误也不可怕；在阳光的照耀下，黑暗将无所遁形。")
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path = minute_files_path
        else:
            self.minute_files_path = self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files = os.listdir(self.minute_files_path)
        self.minute_files = [i for i in self.minute_files if i.endswith(".mat")]
        self.minute_files = sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns = minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list = []
        # 更新数据用的列表
        self.daily_factors_list_update = []
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        # 最终月度因子表格
        self.monthly_factors = None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path = daily_factors_path
        else:
            self.daily_factors_path = self.homeplace.factor_data_file + "日频_"
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path = monthly_factors_path
        else:
            self.monthly_factors_path = self.homeplace.factor_data_file + "月频_"

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """将几个因子截面标准化之后，因子值相加"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __truediv__(self, selfa):
        """两个一正一副的因子，可以用此方法相减"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac = fac1 - fac2
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __floordiv__(self, selfa):
        """两个因子一正一负，可以用此方法相除"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2 = fac2 - fac2.min()
        fac = fac1 / fac2
        fac = fac.replace(np.inf, np.nan)
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __sub__(self, selfa):
        """用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可"""
        tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def wide_to_long(self, df, i):
        """将宽数据转化为长数据，用于因子表转化和拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", i]
        df = df.set_index(["date", "code"])
        return df

    def de_in_group(self, df, help_names):
        """对每个时间，分别做回归，剔除相关因子"""
        ols_order = "fac~" + "+".join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params["Intercept"]
        df = df[["fac"]]
        return df

    def mat_to_df(self, mat, use_datetime=True):
        """将mat文件变成"""
        mat_path = "/".join([self.minute_files_path, mat])
        df = list(scio.loadmat(mat_path).values())[3]
        df = pd.DataFrame(df, columns=self.minute_columns)
        if use_datetime:
            df.date = pd.to_datetime(df.date.apply(str), format="%Y%m%d")
            df = df.set_index("date")
        return df

    def add_suffix(self, code):
        """给股票代码加上后缀"""
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

    def minute_to_daily(
        self,
        func,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
        update=0,
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """

        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files):
                # try:
                code = self.add_suffix(mat[-10:-4])
                self.code = code
                df = self.mat_to_df(mat, use_datetime=True)
                if add_tr:
                    share = read_daily("AllStock_DailyAShareNum.mat")
                    share_this = share[code].to_frame("sharenum").reset_index()
                    share_this.columns = ["date", "sharenum"]
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df = pd.merge(df, share_this, on=["date"], how="left")
                    df = df.assign(tr=df.amount / df.sharenum)
                df = df.reset_index()
                df.columns = ["date"] + list(df.columns)[1:]
                df.date = df.date.dt.strftime("%Y%m%d")
                df.date = df.date.astype(int)
                df = df[(df.date >= start_date) & (df.date <= end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                priclose = df.groupby("date").last()
                priclose = priclose.shift(1).reset_index()
                df = pd.concat([priclose, df])
                the_func = partial(func)
                df = df.groupby("date").apply(the_func)
                df = df.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
                # except Exception as e:
                #     if not STATES['NO_LOG:
                #         logger.warning(f'{code} 缺失')
                #         logger.error(e)
        else:
            for mat in tqdm.tqdm(self.minute_files):
                # try:
                code = self.add_suffix(mat[-10:-4])
                self.code = code
                df = self.mat_to_df(mat, use_datetime=True)
                if add_tr:
                    share = read_daily("AllStock_DailyAShareNum.mat")
                    share_this = share[code].to_frame("sharenum").reset_index()
                    share_this.columns = ["date", "sharenum"]
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df = pd.merge(df, share_this, on=["date"], how="left")
                    df = df.assign(tr=df.amount / df.sharenum)
                the_func = partial(func)
                df = df.reset_index()
                df.columns = ["date"] + list(df.columns)[1:]
                df.date = df.date.dt.strftime("%Y%m%d")
                df.date = df.date.astype(int)
                df = df[(df.date >= start_date) & (df.date <= end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                df = df.groupby("date").apply(the_func)
                df = df.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
                # except Exception as e:
                #     if not STATES['NO_LOG:
                #         logger.warning(f'{code} 缺失')
                #         logger.error(e)
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors_update.index = pd.to_datetime(
                self.daily_factors_update.index.astype(int), format="%Y%m%d"
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
            self.daily_factors.index = pd.to_datetime(
                self.daily_factors.index.astype(int), format="%Y%m%d"
            )
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def minute_to_daily_whole(
        self, func, start_date=10000000, end_date=30000000, update=0
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """
        for mat in tqdm.tqdm(self.minute_files):
            self.code = self.add_suffix(mat[-10:-4])
            df = self.mat_to_df(mat)
            df.date = df.date.astype(int)
            df = df[(df.date >= start_date) & (df.date <= end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func = partial(func)
            df = func(df)
            if isinstance(df, pd.DataFrame):
                df.columns = [self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df, pd.Series):
                df = df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not STATES["NO_LOG"]:
                    logger.warning(f"df is {df}")
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    def get_daily_factors(
        self,
        func,
        whole=False,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
    ):
        """调用分钟到日度方法，算出日频数据"""
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index("date")
            sql = sqlConfig("minute_data_alter")
            now_minute_data = sql.show_tables(full=False)[-1]
            now_minute_data = pd.Timestamp(now_minute_data)
            if self.daily_factors.index.max() < now_minute_data:
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data}，开始更新……"
                    )
                start_date_update = int(
                    datetime.datetime.strftime(
                        self.daily_factors.index.max() + pd.Timedelta("1 day"), "%Y%m%d"
                    )
                )
                end_date_update = int(
                    datetime.datetime.strftime(now_minute_data, "%Y%m%d")
                )
                if whole:
                    self.minute_to_daily_whole(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
                else:
                    self.minute_to_daily(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
        except Exception:
            if whole:
                self.minute_to_daily_whole(
                    func, start_date=start_date, end_date=end_date
                )
            else:
                self.minute_to_daily(
                    func,
                    add_priclose=add_priclose,
                    add_tr=add_tr,
                    start_date=start_date,
                    end_date=end_date,
                )

    def get_single_day_factor(self, func, day: int) -> pd.DataFrame:
        """计算单日的因子值，通过sql数据库，读取单日的数据，然后计算因子值"""
        sql = sqlConfig("minute_data_alter")
        df = sql.get_data(str(day))
        the_func = partial(func)
        df = df.groupby(["code"]).apply(the_func).to_frame()
        df.columns = [str(day)]
        df = df.T
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        return df

    def get_daily_factors_alter(
        self, func, start_date=10000000, end_date=30000000
    ) -> pd.DataFrame:
        """通过minute_data_alter数据库一天一天计算因子值"""
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index("date")
            sql = sqlConfig("minute_data_stock_alter")
            now_minute_datas = sql.show_tables(full=False)
            now_minute_data = now_minute_datas[-1]
            now_minute_data = pd.Timestamp(now_minute_data)
            if self.daily_factors.index.max() < now_minute_data:
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data}，开始更新……"
                    )
                old_end = datetime.datetime.strftime(
                    self.daily_factors.index.max(), "%Y%m%d"
                )
                now_minute_datas = [i for i in now_minute_datas if i > old_end]
                dfs = []
                for c in tqdm.tqdm(now_minute_datas, desc="桂棹兮兰桨，击空明兮遂流光🌊"):
                    df = self.get_single_day_factor(func, c)
                    dfs.append(df)
                dfs = pd.concat(dfs)
                dfs = dfs.sort_index()
                self.daily_factors = pd.concat([self.daily_factors, dfs])
                self.daily_factors = self.daily_factors.dropna(how="all")
                self.daily_factors = self.daily_factors[
                    self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
                ]
                self.daily_factors.reset_index().to_feather(self.daily_factors_path)
                if not STATES["NO_LOG"]:
                    logger.success("更新已完成")

        except Exception:
            self.minute_to_daily(func, start_date=start_date, end_date=end_date)

    def get_neutral_monthly_factors(self, df, boxcox=False):
        """对月度因子做市值中性化处理"""
        shen = pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5, boxcox=True, plt=False, print_comments=False)
        else:
            shen.run(5, neutralize=True, plt=False, print_comments=False)
        new_factors = shen.factors.copy()
        new_factors = new_factors.set_index(["date", "code"]).unstack()
        new_factors.columns = list(map(lambda x: x[1], list(new_factors.columns)))
        new_factors = new_factors.reset_index()
        add_start_point = new_factors.date.min()
        add_start_point = add_start_point - pd.Timedelta(days=add_start_point.day)
        new_factors.date = new_factors.date.shift(1)
        new_factors.date = new_factors.date.fillna(add_start_point)
        new_factors = new_factors.set_index("date")
        return new_factors

    def get_monthly_factors(self, func, neutralize, boxcox):
        """将日频的因子转化为月频因子"""
        two_parts = self.monthly_factors_path.split(".")
        try:
            self.monthly_factors = pd.read_feather(self.monthly_factors_path)
            self.monthly_factors = self.monthly_factors.set_index("date")
        except Exception:
            the_func = partial(func)
            self.monthly_factors = the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors
                )
            elif boxcox:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors, boxcox=True
                )

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(
        self,
        whole=False,
        daily_func=None,
        monthly_func=None,
        neutralize=False,
        boxcox=False,
    ):
        """执必要的函数，将分钟数据变成月度因子"""
        self.get_daily_factors(daily_func, whole)
        self.get_monthly_factors(monthly_func, neutralize, boxcox)


class run_away_with_me:
    def __init__(
        self,
        minute_files_path=None,
        minute_columns=["date", "open", "high", "low", "close", "amount", "money"],
        daily_factors_path=None,
        monthly_factors_path=None,
    ):
        self.homeplace = HomePlace()
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path = minute_files_path
        else:
            self.minute_files_path = self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files = os.listdir(self.minute_files_path)
        self.minute_files = [i for i in self.minute_files if i.endswith(".mat")]
        self.minute_files = sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns = minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list = []
        # 更新数据用的列表
        self.daily_factors_list_update = []
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        # 最终月度因子表格
        self.monthly_factors = None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path = daily_factors_path
        else:
            self.daily_factors_path = self.homeplace.factor_data_file + "日频_"
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path = monthly_factors_path
        else:
            self.monthly_factors_path = self.homeplace.factor_data_file + "月频_"

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """将几个因子截面标准化之后，因子值相加"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __truediv__(self, selfa):
        """两个一正一副的因子，可以用此方法相减"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac = fac1 - fac2
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __floordiv__(self, selfa):
        """两个因子一正一负，可以用此方法相除"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2 = fac2 - fac2.min()
        fac = fac1 / fac2
        fac = fac.replace(np.inf, np.nan)
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __sub__(self, selfa):
        """用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可"""
        tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def wide_to_long(self, df, i):
        """将宽数据转化为长数据，用于因子表转化和拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", i]
        df = df.set_index(["date", "code"])
        return df

    def de_in_group(self, df, help_names):
        """对每个时间，分别做回归，剔除相关因子"""
        ols_order = "fac~" + "+".join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params["Intercept"]
        df = df[["fac"]]
        return df

    def mat_to_df(self, mat, use_datetime=True):
        """将mat文件变成"""
        mat_path = "/".join([self.minute_files_path, mat])
        df = list(scio.loadmat(mat_path).values())[3]
        df = pd.DataFrame(df, columns=self.minute_columns)
        if use_datetime:
            df.date = pd.to_datetime(df.date.apply(str), format="%Y%m%d")
            df = df.set_index("date")
        return df

    def add_suffix(self, code):
        """给股票代码加上后缀"""
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

    def minute_to_daily(
        self,
        func,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
        update=0,
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """

        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files):
                try:
                    code = self.add_suffix(mat[-10:-4])
                    self.code = code
                    df = self.mat_to_df(mat, use_datetime=True)
                    if add_tr:
                        share = read_daily("AllStock_DailyAShareNum.mat")
                        share_this = share[code].to_frame("sharenum").reset_index()
                        share_this.columns = ["date", "sharenum"]
                        df = df.reset_index()
                        df.columns = ["date"] + list(df.columns)[1:]
                        df = pd.merge(df, share_this, on=["date"], how="left")
                        df = df.assign(tr=df.amount / df.sharenum)
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df.date = df.date.dt.strftime("%Y%m%d")
                    df.date = df.date.astype(int)
                    df = df[(df.date >= start_date) & (df.date <= end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    priclose = df.groupby("date").last()
                    priclose = priclose.shift(1).reset_index()
                    df = pd.concat([priclose, df])
                    the_func = partial(func)
                    date_sets = sorted(list(set(df.date)))
                    ress = []
                    for i in range(len(date_sets) - 19):
                        res = df[
                            (df.date >= date_sets[i]) & (df.date <= date_sets[i + 19])
                        ]
                        res = the_func(res)
                        ress.append(res)
                    ress = pd.concat(ress)
                    if isinstance(ress, pd.DataFrame):
                        if "date" in list(ress.columns):
                            ress = ress.set_index("date").iloc[:, 0]
                        else:
                            ress = ress.iloc[:, 0]
                    else:
                        ress = ress.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(ress)
                    else:
                        self.daily_factors_list_update.append(ress)
                except Exception as e:
                    if not STATES["NO_LOG"]:
                        logger.warning(f"{code} 缺失")
                        logger.error(e)
        else:
            for mat in tqdm.tqdm(self.minute_files):
                try:
                    code = self.add_suffix(mat[-10:-4])
                    self.code = code
                    df = self.mat_to_df(mat, use_datetime=True)
                    if add_tr:
                        share = read_daily("AllStock_DailyAShareNum.mat")
                        share_this = share[code].to_frame("sharenum").reset_index()
                        share_this.columns = ["date", "sharenum"]
                        df = df.reset_index()
                        df.columns = ["date"] + list(df.columns)[1:]
                        df = pd.merge(df, share_this, on=["date"], how="left")
                        df = df.assign(tr=df.amount / df.sharenum)
                    the_func = partial(func)
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df.date = df.date.dt.strftime("%Y%m%d")
                    df.date = df.date.astype(int)
                    df = df[(df.date >= start_date) & (df.date <= end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    date_sets = sorted(list(set(df.date)))
                    ress = []
                    for i in range(len(date_sets) - 19):
                        res = df[
                            (df.date >= date_sets[i]) & (df.date <= date_sets[i + 19])
                        ]
                        res = the_func(res)
                        ress.append(res)
                    ress = pd.concat(ress)
                    if isinstance(ress, pd.DataFrame):
                        if "date" in list(ress.columns):
                            ress = ress.set_index("date").iloc[:, 0]
                        else:
                            ress = ress.iloc[:, 0]
                    else:
                        ress = ress.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(ress)
                    else:
                        self.daily_factors_list_update.append(ress)
                except Exception as e:
                    if not STATES["NO_LOG"]:
                        logger.warning(f"{code} 缺失")
                        logger.error(e)
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors_update.index = pd.to_datetime(
                self.daily_factors_update.index.astype(int), format="%Y%m%d"
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
            self.daily_factors.index = pd.to_datetime(
                self.daily_factors.index.astype(int), format="%Y%m%d"
            )
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def minute_to_daily_whole(
        self, func, start_date=10000000, end_date=30000000, update=0
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """
        for mat in tqdm.tqdm(self.minute_files):
            self.code = self.add_suffix(mat[-10:-4])
            df = self.mat_to_df(mat)
            df.date = df.date.astype(int)
            df = df[(df.date >= start_date) & (df.date <= end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func = partial(func)
            df = func(df)
            if isinstance(df, pd.DataFrame):
                df.columns = [self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df, pd.Series):
                df = df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not STATES["NO_LOG"]:
                    logger.warning(f"df is {df}")
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    @kk.desktop_sender(title="嘿，多日分钟数据处理完啦～🐬")
    def get_daily_factors(
        self,
        func,
        whole=False,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
    ):
        """调用分钟到日度方法，算出日频数据"""
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index("date")
            now_minute_data = self.mat_to_df(self.minute_files[0])
            if self.daily_factors.index.max() < now_minute_data.index.max():
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data.index.max()}，开始更新……"
                    )
                start_date_update = int(
                    datetime.datetime.strftime(
                        self.daily_factors.index.max() + pd.Timedelta("1 day"), "%Y%m%d"
                    )
                )
                end_date_update = int(
                    datetime.datetime.strftime(now_minute_data.index.max(), "%Y%m%d")
                )
                if whole:
                    self.minute_to_daily_whole(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
                else:
                    self.minute_to_daily(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
        except Exception:
            if whole:
                self.minute_to_daily_whole(
                    func, start_date=start_date, end_date=end_date
                )
            else:
                self.minute_to_daily(
                    func,
                    add_priclose=add_priclose,
                    add_tr=add_tr,
                    start_date=start_date,
                    end_date=end_date,
                )

    def get_neutral_monthly_factors(self, df, boxcox=False):
        """对月度因子做市值中性化处理"""
        shen = pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5, boxcox=True, plt=False, print_comments=False)
        else:
            shen.run(5, neutralize=True, plt=False, print_comments=False)
        new_factors = shen.factors.copy()
        new_factors = new_factors.set_index(["date", "code"]).unstack()
        new_factors.columns = list(map(lambda x: x[1], list(new_factors.columns)))
        new_factors = new_factors.reset_index()
        add_start_point = new_factors.date.min()
        add_start_point = add_start_point - pd.Timedelta(days=add_start_point.day)
        new_factors.date = new_factors.date.shift(1)
        new_factors.date = new_factors.date.fillna(add_start_point)
        new_factors = new_factors.set_index("date")
        return new_factors

    def get_monthly_factors(self, func, neutralize, boxcox):
        """将日频的因子转化为月频因子"""
        two_parts = self.monthly_factors_path.split(".")
        try:
            self.monthly_factors = pd.read_feather(self.monthly_factors_path)
            self.monthly_factors = self.monthly_factors.set_index("date")
        except Exception:
            the_func = partial(func)
            self.monthly_factors = the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors
                )
            elif boxcox:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors, boxcox=True
                )

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(
        self,
        whole=False,
        daily_func=None,
        monthly_func=None,
        neutralize=False,
        boxcox=False,
    ):
        """执必要的函数，将分钟数据变成月度因子"""
        self.get_daily_factors(daily_func, whole)
        self.get_monthly_factors(monthly_func, neutralize, boxcox)


class carry_me_to_bathroom:
    def __init__(
        self,
        minute_files_path=None,
        minute_columns=["date", "open", "high", "low", "close", "amount", "money"],
        daily_factors_path=None,
        monthly_factors_path=None,
    ):
        self.homeplace = HomePlace()
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path = minute_files_path
        else:
            self.minute_files_path = self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files = os.listdir(self.minute_files_path)
        self.minute_files = [i for i in self.minute_files if i.endswith(".mat")]
        self.minute_files = sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns = minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list = []
        # 更新数据用的列表
        self.daily_factors_list_update = []
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        # 最终月度因子表格
        self.monthly_factors = None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path = daily_factors_path
        else:
            self.daily_factors_path = self.homeplace.factor_data_file + "日频_"
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path = monthly_factors_path
        else:
            self.monthly_factors_path = self.homeplace.factor_data_file + "月频_"

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """将几个因子截面标准化之后，因子值相加"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __truediv__(self, selfa):
        """两个一正一副的因子，可以用此方法相减"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac = fac1 - fac2
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __floordiv__(self, selfa):
        """两个因子一正一负，可以用此方法相除"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2 = fac2 - fac2.min()
        fac = fac1 / fac2
        fac = fac.replace(np.inf, np.nan)
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __sub__(self, selfa):
        """用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可"""
        tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def wide_to_long(self, df, i):
        """将宽数据转化为长数据，用于因子表转化和拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", i]
        df = df.set_index(["date", "code"])
        return df

    def de_in_group(self, df, help_names):
        """对每个时间，分别做回归，剔除相关因子"""
        ols_order = "fac~" + "+".join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params["Intercept"]
        df = df[["fac"]]
        return df

    def mat_to_df(self, mat, use_datetime=True):
        """将mat文件变成"""
        mat_path = "/".join([self.minute_files_path, mat])
        df = list(scio.loadmat(mat_path).values())[3]
        df = pd.DataFrame(df, columns=self.minute_columns)
        if use_datetime:
            df.date = pd.to_datetime(df.date.apply(str), format="%Y%m%d")
            df = df.set_index("date")
        return df

    def add_suffix(self, code):
        """给股票代码加上后缀"""
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

    def minute_to_daily(
        self,
        func,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
        update=0,
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """

        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files):
                code = self.add_suffix(mat[-10:-4])
                self.code = code
                df = self.mat_to_df(mat, use_datetime=True)
                if add_tr:
                    share = read_daily("AllStock_DailyAShareNum.mat")
                    share_this = share[code].to_frame("sharenum").reset_index()
                    share_this.columns = ["date", "sharenum"]
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df = pd.merge(df, share_this, on=["date"], how="left")
                    df = df.assign(tr=df.amount / df.sharenum)
                df = df.reset_index()
                df.columns = ["date"] + list(df.columns)[1:]
                df.date = df.date.dt.strftime("%Y%m%d")
                df.date = df.date.astype(int)
                df = df[(df.date >= start_date) & (df.date <= end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                priclose = df.groupby("date").last()
                priclose = priclose.shift(1).reset_index()
                df = pd.concat([priclose, df])
                the_func = partial(func)
                date_sets = sorted(list(set(df.date)))
                ress = []
                for i in range(len(date_sets) - 19):
                    res = df[(df.date >= date_sets[i]) & (df.date <= date_sets[i + 19])]
                    res = the_func(res)
                    ress.append(res)
                ress = pd.concat(ress)
                if isinstance(ress, pd.DataFrame):
                    if "date" in list(ress.columns):
                        ress = ress.set_index("date").iloc[:, 0]
                    else:
                        ress = ress.iloc[:, 0]
                else:
                    ress = ress.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(ress)
                else:
                    self.daily_factors_list_update.append(ress)
        else:
            for mat in tqdm.tqdm(self.minute_files):
                code = self.add_suffix(mat[-10:-4])
                self.code = code
                df = self.mat_to_df(mat, use_datetime=True)
                if add_tr:
                    share = read_daily("AllStock_DailyAShareNum.mat")
                    share_this = share[code].to_frame("sharenum").reset_index()
                    share_this.columns = ["date", "sharenum"]
                    df = df.reset_index()
                    df.columns = ["date"] + list(df.columns)[1:]
                    df = pd.merge(df, share_this, on=["date"], how="left")
                    df = df.assign(tr=df.amount / df.sharenum)
                the_func = partial(func)
                df = df.reset_index()
                df.columns = ["date"] + list(df.columns)[1:]
                df.date = df.date.dt.strftime("%Y%m%d")
                df.date = df.date.astype(int)
                df = df[(df.date >= start_date) & (df.date <= end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                date_sets = sorted(list(set(df.date)))
                ress = []
                for i in range(len(date_sets) - 19):
                    res = df[(df.date >= date_sets[i]) & (df.date <= date_sets[i + 19])]
                    res = the_func(res)
                    ress.append(res)
                ress = pd.concat(ress)
                if isinstance(ress, pd.DataFrame):
                    if "date" in list(ress.columns):
                        ress = ress.set_index("date").iloc[:, 0]
                    else:
                        ress = ress.iloc[:, 0]
                else:
                    ress = ress.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(ress)
                else:
                    self.daily_factors_list_update.append(ress)
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors_update.index = pd.to_datetime(
                self.daily_factors_update.index.astype(int), format="%Y%m%d"
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
            self.daily_factors.index = pd.to_datetime(
                self.daily_factors.index.astype(int), format="%Y%m%d"
            )
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def minute_to_daily_whole(
        self, func, start_date=10000000, end_date=30000000, update=0
    ):
        """
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        """
        for mat in tqdm.tqdm(self.minute_files):
            self.code = self.add_suffix(mat[-10:-4])
            df = self.mat_to_df(mat)
            df.date = df.date.astype(int)
            df = df[(df.date >= start_date) & (df.date <= end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func = partial(func)
            df = func(df)
            if isinstance(df, pd.DataFrame):
                df.columns = [self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df, pd.Series):
                df = df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not STATES["NO_LOG"]:
                    logger.warning(f"df is {df}")
        if update:
            self.daily_factors_update = pd.concat(
                self.daily_factors_list_update, axis=1
            )
            self.daily_factors = pd.concat(
                [self.daily_factors, self.daily_factors_update]
            )
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors = self.daily_factors.dropna(how="all")
        self.daily_factors = self.daily_factors[
            self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
        ]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not STATES["NO_LOG"]:
            logger.success("更新已完成")

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    def get_daily_factors(
        self,
        func,
        whole=False,
        add_priclose=False,
        add_tr=False,
        start_date=10000000,
        end_date=30000000,
    ):
        """调用分钟到日度方法，算出日频数据"""
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index("date")
            now_minute_data = self.mat_to_df(self.minute_files[0])
            if self.daily_factors.index.max() < now_minute_data.index.max():
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data.index.max()}，开始更新……"
                    )
                start_date_update = int(
                    datetime.datetime.strftime(
                        self.daily_factors.index.max() + pd.Timedelta("1 day"), "%Y%m%d"
                    )
                )
                end_date_update = int(
                    datetime.datetime.strftime(now_minute_data.index.max(), "%Y%m%d")
                )
                if whole:
                    self.minute_to_daily_whole(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
                else:
                    self.minute_to_daily(
                        func,
                        start_date=start_date_update,
                        end_date=end_date_update,
                        update=1,
                    )
        except Exception:
            if whole:
                self.minute_to_daily_whole(
                    func, start_date=start_date, end_date=end_date
                )
            else:
                self.minute_to_daily(
                    func,
                    add_priclose=add_priclose,
                    add_tr=add_tr,
                    start_date=start_date,
                    end_date=end_date,
                )

    def get_neutral_monthly_factors(self, df, boxcox=False):
        """对月度因子做市值中性化处理"""
        shen = pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5, boxcox=True, plt=False, print_comments=False)
        else:
            shen.run(5, neutralize=True, plt=False, print_comments=False)
        new_factors = shen.factors.copy()
        new_factors = new_factors.set_index(["date", "code"]).unstack()
        new_factors.columns = list(map(lambda x: x[1], list(new_factors.columns)))
        new_factors = new_factors.reset_index()
        add_start_point = new_factors.date.min()
        add_start_point = add_start_point - pd.Timedelta(days=add_start_point.day)
        new_factors.date = new_factors.date.shift(1)
        new_factors.date = new_factors.date.fillna(add_start_point)
        new_factors = new_factors.set_index("date")
        return new_factors

    def get_monthly_factors(self, func, neutralize, boxcox):
        """将日频的因子转化为月频因子"""
        two_parts = self.monthly_factors_path.split(".")
        try:
            self.monthly_factors = pd.read_feather(self.monthly_factors_path)
            self.monthly_factors = self.monthly_factors.set_index("date")
        except Exception:
            the_func = partial(func)
            self.monthly_factors = the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors
                )
            elif boxcox:
                self.monthly_factors = self.get_neutral_monthly_factors(
                    self.monthly_factors, boxcox=True
                )

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(
        self,
        whole=False,
        daily_func=None,
        monthly_func=None,
        neutralize=False,
        boxcox=False,
    ):
        """执必要的函数，将分钟数据变成月度因子"""
        self.get_daily_factors(daily_func, whole)
        self.get_monthly_factors(monthly_func, neutralize, boxcox)


class pure_fallmount(pure_fall):
    """继承自父类，专为做因子截面标准化之后相加和因子剔除其他辅助因子的作用"""

    def __init__(self, monthly_factors):
        """输入月度因子值，以设定新的对象"""
        super(pure_fall, self).__init__()
        self.monthly_factors = monthly_factors

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """返回一个对象，而非一个表格，如需表格请调用对象"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fallmount(fac1)
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __sub__(self, selfa):
        """返回对象，如需表格，请调用对象"""
        tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        new_pure = pure_fallmount(facs)
        return new_pure

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fallmount(xy)
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fallmount(xy)
        return new_pure


class pure_winter:
    def __init__(self):
        self.homeplace = HomePlace()
        # barra因子数据
        self.barras = self.read_h5(
            self.homeplace.barra_data_file + "FactorLoading_Style.h5"
        )

    def __call__(self, fallmount=0):
        """返回纯净因子值"""
        if fallmount == 0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)

    def read_h5(self, path):
        """读入h5文件"""
        res = {}
        a = h5py.File(path)
        for k, v in tqdm.tqdm(list(a.items()), desc="数据加载中……"):
            value = list(v.values())[-1]
            col = [i.decode("utf-8") for i in list(list(v.values())[0])]
            ind = [i.decode("utf-8") for i in list(list(v.values())[1])]
            res[k] = pd.DataFrame(value, columns=col, index=ind)
        return res

    @history_remain(slogan="abandoned")
    def last_month_end(self, x):
        """找到下个月最后一天"""
        x1 = x = x - relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @history_remain(slogan="abandoned")
    def set_factors_df(self, df):
        """传入因子dataframe，应为三列，第一列是时间，第二列是股票代码，第三列是因子值"""
        df1 = df.copy()
        df1.columns = ["date", "code", "fac"]
        df1 = df1.set_index(["date", "code"])
        df1 = df1.unstack().reset_index()
        df1.date = df1.date.apply(self.last_month_end)
        df1 = df1.set_index(["date"]).stack()
        self.factors = df1.copy()

    def set_factors_df_wide(self, df):
        """传入因子数据，时间为索引，代码为列名"""
        df1 = df.copy()
        # df1.index=df1.index-pd.DateOffset(months=1)
        df1 = df1.resample("M").last()
        df1 = df1.stack().reset_index()
        df1.columns = ["date", "code", "fac"]
        self.factors = df1.copy()

    def daily_to_monthly(self, df):
        """将日度的barra因子月度化"""
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df = df.resample("M").last()
        return df

    def get_monthly_barras_industrys(self):
        """将barra因子和行业哑变量变成月度数据"""
        for key, value in self.barras.items():
            self.barras[key] = self.daily_to_monthly(value)

    def wide_to_long(self, df, name):
        """将宽数据变成长数据，便于后续拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", name]
        df = df.set_index(["date", "code"])
        return df

    def get_wide_barras_industrys(self):
        """将barra因子和行业哑变量都变成长数据"""
        for key, value in self.barras.items():
            self.barras[key] = self.wide_to_long(value, key)

    def get_corr_pri_ols_pri(self):
        """拼接barra因子和行业哑变量，生成用于求相关系数和纯净因子的数据表"""
        if self.factors.shape[0] > 1:
            self.factors = self.factors.set_index(["date", "code"])
        self.corr_pri = pd.concat(
            [self.factors] + list(self.barras.values()), axis=1
        ).dropna()

    def get_corr(self):
        """计算每一期的相关系数，再求平均值"""
        self.corr_by_step = self.corr_pri.groupby(["date"]).apply(
            lambda x: x.corr().head(1)
        )
        self.__corr = self.corr_by_step.mean()
        self.__corr.index = [
            "因子自身",
            "贝塔",
            "估值",
            "杠杆",
            "盈利",
            "成长",
            "流动性",
            "反转",
            "波动率",
            "市值",
            "非线性市值",
        ]

    @property
    def corr(self):
        return self.__corr.copy()

    def ols_in_group(self, df):
        """对每个时间段进行回归，并计算残差"""
        xs = list(df.columns)
        xs = [i for i in xs if i != "fac"]
        xs_join = "+".join(xs)
        ols_formula = "fac~" + xs_join
        ols_result = smf.ols(ols_formula, data=df).fit()
        ols_ws = {i: ols_result.params[i] for i in xs}
        ols_b = ols_result.params["Intercept"]
        to_minus = [ols_ws[i] * df[i] for i in xs]
        to_minus = reduce(lambda x, y: x + y, to_minus)
        df = df.assign(snow_fac=df.fac - to_minus - ols_b)
        df = df[["snow_fac"]]
        df = df.rename(columns={"snow_fac": "fac"})
        return df

    def get_snow_fac(self):
        """获得纯净因子"""
        self.snow_fac = self.corr_pri.groupby(["date"]).apply(self.ols_in_group)
        self.snow_fac = self.snow_fac.unstack()
        self.snow_fac.columns = list(map(lambda x: x[1], list(self.snow_fac.columns)))

    def run(self):
        """运行一些必要的函数"""
        self.get_monthly_barras_industrys()
        self.get_wide_barras_industrys()
        self.get_corr_pri_ols_pri()
        self.get_corr()
        self.get_snow_fac()


class pure_coldwinter:
    # DONE: 可以自由添加其他要剔除的因子，或者替换某些要剔除的因子
    def __init__(self):
        self.homeplace = HomePlace()
        # barra因子数据
        styles = os.listdir(homeplace.barra_data_file)
        styles = [i for i in styles if i.endswith(".feather")]
        barras = {}
        for s in styles:
            k = s.split(".")[0]
            v = pd.read_feather(homeplace.barra_data_file + s)
            v.columns = ["date"] + list(v.columns)[1:]
            v = v.set_index("date")
            barras[k] = v
        self.barras = barras

    def __call__(self, fallmount=0):
        """返回纯净因子值"""
        if fallmount == 0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)

    @history_remain(slogan="abandoned")
    def last_month_end(self, x):
        """找到下个月最后一天"""
        x1 = x = x - relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @history_remain(slogan="abandoned")
    def set_factors_df(self, df):
        """传入因子dataframe，应为三列，第一列是时间，第二列是股票代码，第三列是因子值"""
        df1 = df.copy()
        df1.columns = ["date", "code", "fac"]
        df1 = df1.set_index(["date", "code"])
        df1 = df1.unstack().reset_index()
        df1.date = df1.date.apply(self.last_month_end)
        df1 = df1.set_index(["date"]).stack()
        self.factors = df1.copy()

    def set_factors_df_wide(self, df):
        """传入因子数据，时间为索引，代码为列名"""
        df1 = df.copy()
        # df1.index=df1.index-pd.DateOffset(months=1)
        df1 = df1.resample("M").last()
        df1 = df1.stack().reset_index()
        df1.columns = ["date", "code", "fac"]
        self.factors = df1.copy()

    def daily_to_monthly(self, df):
        """将日度的barra因子月度化"""
        df = df.resample("M").last()
        return df

    def get_monthly_barras_industrys(self):
        """将barra因子和行业哑变量变成月度数据"""
        for key, value in self.barras.items():
            self.barras[key] = self.daily_to_monthly(value)

    def wide_to_long(self, df, name):
        """将宽数据变成长数据，便于后续拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", name]
        df = df.set_index(["date", "code"])
        return df

    def get_wide_barras_industrys(self):
        """将barra因子和行业哑变量都变成长数据"""
        for key, value in self.barras.items():
            self.barras[key] = self.wide_to_long(value, key)

    def get_corr_pri_ols_pri(self):
        """拼接barra因子和行业哑变量，生成用于求相关系数和纯净因子的数据表"""
        if self.factors.shape[0] > 1:
            self.factors = self.factors.set_index(["date", "code"])
        self.corr_pri = pd.concat(
            [self.factors] + list(self.barras.values()), axis=1
        ).dropna()

    # DONE: 修改风格因子展示顺序至报告的顺序
    def get_corr(self):
        """计算每一期的相关系数，再求平均值"""
        self.corr_by_step = self.corr_pri.groupby(["date"]).apply(
            lambda x: x.corr().head(1)
        )
        self.__corr = self.corr_by_step.mean()
        self.__corr = self.__corr.rename(
            index={
                "fac": "因子自身",
                "beta": "贝塔",
                "booktoprice": "估值",
                "leverage": "杠杆",
                "earningsyield": "盈利",
                "growth": "成长",
                "liquidity": "流动性",
                "momentum": "动量",
                "residualvolatility": "波动率",
                "size": "市值",
                "nonlinearsize": "非线性市值",
            }
        )
        # self.__corr.index=['因子自身','贝塔','估值','杠杆',
        #                    '盈利','成长','流动性','反转','波动率',
        #                    '市值','非线性市值']

    @property
    def corr(self):
        return self.__corr.copy()

    def ols_in_group(self, df):
        """对每个时间段进行回归，并计算残差"""
        xs = list(df.columns)
        xs = [i for i in xs if i != "fac"]
        xs_join = "+".join(xs)
        ols_formula = "fac~" + xs_join
        ols_result = smf.ols(ols_formula, data=df).fit()
        ols_ws = {i: ols_result.params[i] for i in xs}
        ols_b = ols_result.params["Intercept"]
        to_minus = [ols_ws[i] * df[i] for i in xs]
        to_minus = reduce(lambda x, y: x + y, to_minus)
        df = df.assign(snow_fac=df.fac - to_minus - ols_b)
        df = df[["snow_fac"]]
        df = df.rename(columns={"snow_fac": "fac"})
        return df

    def get_snow_fac(self):
        """获得纯净因子"""
        self.snow_fac = self.corr_pri.groupby(["date"]).apply(self.ols_in_group)
        self.snow_fac = self.snow_fac.unstack()
        self.snow_fac.columns = list(map(lambda x: x[1], list(self.snow_fac.columns)))

    def run(self):
        """运行一些必要的函数"""
        self.get_monthly_barras_industrys()
        self.get_wide_barras_industrys()
        self.get_corr_pri_ols_pri()
        self.get_corr()
        self.get_snow_fac()


class pure_snowtrain(pure_coldwinter):
    """直接返回纯净因子"""

    def __init__(self, factors):
        """直接输入原始因子数据"""
        super(pure_snowtrain, self).__init__()
        self.set_factors_df_wide(factors.copy())
        self.run()

    def __call__(self, fallmount=0):
        """可以直接返回pure_fallmount对象，或纯净因子矩阵"""
        if fallmount == 0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)


class pure_snowtrain_old(pure_winter):
    """直接返回纯净因子"""

    def __init__(self, factors):
        """直接输入原始因子数据"""
        super(pure_snowtrain, self).__init__()
        self.set_factors_df_wide(factors.copy())
        self.run()

    def __call__(self, fallmount=0):
        """可以直接返回pure_fallmount对象，或纯净因子矩阵"""
        if fallmount == 0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)


class pure_moonlight(pure_moon):
    """继承自pure_moon回测框架，使用其中的日频复权收盘价数据，以及换手率数据"""

    def __init__(self):
        """加载全部数据"""
        super(pure_moonlight, self).__init__()
        self.homeplace = HomePlace()
        self.col_and_index()
        self.load_all_files()
        self.judge_month()
        self.get_log_cap()
        # 对数市值
        self.cap_as_factor = (
            self.cap[["date", "code", "cap_size"]].set_index(["date", "code"]).unstack()
        )
        self.cap_as_factor.columns = list(
            map(lambda x: x[1], list(self.cap_as_factor.columns))
        )
        # 传统反转因子ret20
        self.ret20_database = self.homeplace.factor_data_file + "月频_反转因子ret20.feather"
        # 传统换手率因子turn20
        self.turn20_database = (
            self.homeplace.factor_data_file + "月频_换手率因子turn20.feather"
        )
        # 传统波动率因子vol20
        self.vol20_database = self.homeplace.factor_data_file + "月频_波动率因子vol20.feather"
        # #自动更新
        self.get_updated_factors()

    def __call__(self, name):
        """可以通过call方式，直接获取对应因子数据"""
        value = getattr(self, name)
        return value

    def get_ret20(self, pri):
        """计算20日涨跌幅因子"""
        past = pri.iloc[:-20, :]
        future = pri.iloc[20:, :]
        ret20 = (future.to_numpy() - past.to_numpy()) / past.to_numpy()
        df = pd.DataFrame(ret20, columns=pri.columns, index=future.index)
        df = df.resample("M").last()
        return df

    def get_turn20(self, pri):
        """计算20换手率因子"""
        turns = pri.rolling(20).mean()
        turns = turns.resample("M").last().reset_index()
        turns.columns = ["date"] + list(turns.columns)[1:]
        self.factors = turns
        self.get_neutral_factors()
        df = self.factors.copy()
        df = df.set_index(["date", "code"])
        df = df.unstack()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        return df

    def get_vol20(self, pri):
        """计算20日波动率因子"""
        rets = pri.pct_change()
        vol = rets.rolling(20).apply(np.std)
        df = vol.resample("M").last()
        return df

    def update_single_factor_in_database(self, path, pri, func):
        """
        用基础数据库更新因子数据库
        执行顺序为，先读取文件，如果没有，就直接全部计算，然后存储
        如果有，就读出来看看。数据是不是最新的
        如果不是最新的，就将原始数据在上次因子存储的日期处截断
        计算出新的一段时间的因子值，然后追加写入因子文件中
        """
        the_func = partial(func)
        try:
            df = pd.read_feather(path)
            df.columns = ["date"] + list(df.columns)[1:]
            df = df.set_index("date")
            if df.index.max() < pri.index.max():
                to_add = pri[
                    (pri.index > df.index.max()) & (pri.index <= pri.index.max())
                ]
                to_add = the_func(to_add)
                df = pd.concat([df, to_add])
                print("之前数据有点旧了，已为您完成更新")
            else:
                print("数据很新，无需更新")
            df1 = df.reset_index()
            df1.columns = ["date"] + list(df1.columns)[1:]
            df1.to_feather(path)
            return df
        except Exception:
            df = the_func(pri)
            df1 = df.reset_index()
            df1.columns = ["date"] + list(df1.columns)[1:]
            df1.to_feather(path)
            print("新因子建库完成")
            return df

    def get_updated_factors(self):
        """更新因子数据"""
        self.ret20 = self.update_single_factor_in_database(
            self.ret20_database, self.closes, self.get_ret20
        )
        self.turn20 = self.update_single_factor_in_database(
            self.turn20_database, self.turnovers, self.get_turn20
        )
        self.vol20 = self.update_single_factor_in_database(
            self.vol20_database, self.closes, self.get_vol20
        )


class pure_moonnight:
    """封装选股框架"""

    __slots__ = ["shen"]

    def __init__(
        self,
        factors,
        groups_num=10,
        neutralize=False,
        boxcox=False,
        by10=False,
        value_weighted=False,
        y2=False,
        plt_plot=True,
        plotly_plot=False,
        filename="分组净值图",
        time_start=None,
        time_end=None,
        print_comments=True,
        comments_writer=None,
        net_values_writer=None,
        rets_writer=None,
        comments_sheetname=None,
        net_values_sheetname=None,
        rets_sheetname=None,
        on_paper=False,
        sheetname=None,
    ):
        """直接输入因子数据"""
        if isinstance(factors, pure_fallmount):
            factors = factors().copy()
        self.shen = pure_moon()
        self.shen.set_factor_df_date_as_index(factors)
        self.shen.prerpare()
        self.shen.run(
            groups_num=groups_num,
            neutralize=neutralize,
            boxcox=boxcox,
            value_weighted=value_weighted,
            y2=y2,
            plt_plot=plt_plot,
            plotly_plot=plotly_plot,
            filename=filename,
            time_start=time_start,
            time_end=time_end,
            print_comments=print_comments,
            comments_writer=comments_writer,
            net_values_writer=net_values_writer,
            rets_writer=rets_writer,
            comments_sheetname=comments_sheetname,
            net_values_sheetname=net_values_sheetname,
            rets_sheetname=rets_sheetname,
            on_paper=on_paper,
            sheetname=sheetname,
        )

    def __call__(self, fallmount=0):
        """调用则返回因子数据"""
        df = self.shen.factors_out.copy()
        # df=df.set_index(['date', 'code']).unstack()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        if fallmount == 0:
            return df
        else:
            return pure_fallmount(df)


class pure_newyear:
    """转为生成25分组和百分组的收益矩阵而封装"""

    def __init__(self, facx, facy, group_num_single, namex="主", namey="次"):
        """初始化时即进行回测"""
        homex = pure_fallmount(facx)
        homey = pure_fallmount(facy)
        if group_num_single == 5:
            homexy = homex > homey
        elif group_num_single == 10:
            homexy = homex >> homey
        shen = pure_moonnight(
            homexy(), group_num_single**2, plt_plot=False, print_comments=False
        )
        sq = shen.shen.square_rets.copy()
        sq.index = [namex + str(i) for i in list(sq.index)]
        sq.columns = [namey + str(i) for i in list(sq.columns)]
        self.square_rets = sq

    def __call__(self):
        """调用对象时，返回最终结果，正方形的分组年化收益率表"""
        return self.square_rets


class pure_dawn:
    """
    因子切割论的母框架，可以对两个因子进行类似于因子切割的操作
    可用于派生任何"以两个因子生成一个因子"的子类
    使用举例
    cut函数里，必须带有输入变量df,df有两个columns，一个名为'fac1'，一个名为'fac2'，df是最近一个回看期内的数据
    class Cut(pure_dawn):
        def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(Cut,self).__init__(fac1=fac1,fac2=fac2)

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
    """

    def __init__(self, fac1, fac2, *args):
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

    def __call__(self):
        """返回最终月度因子值"""
        return self.fac

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

    def make_monthly_factors_single_code(self, df, func):
        """
        对单一股票来计算月度因子
        func为单月执行的函数，返回值应为月度因子，如一个float或一个list
        df为一个股票的四列表，包含时间、代码、因子1和因子2
        """
        res = {}
        for start, end in zip(self.month_starts, self.month_ends):
            this_month = df[(df.date >= start) & (df.date <= end)]
            res[end] = func(this_month)
        dates = list(res.keys())
        corrs = list(res.values())
        part = pd.DataFrame({"date": dates, "corr": corrs})
        return part

    def get_monthly_factor(self, func):
        """运行自己写的函数，获得月度因子"""
        tqdm.tqdm.pandas(desc="when the dawn comes, tonight will be a memory too.")
        self.fac = self.fac.groupby(["code"]).progress_apply(
            lambda x: self.make_monthly_factors_single_code(x, func)
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
    def run(self, func, backsee=20):
        """运行必要的函数"""
        self.get_fac_long_and_tradedays()
        self.get_month_starts_and_ends(backsee=backsee)
        self.get_monthly_factor(func)


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
        moon = pure_moon()
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
            self.fac = decap_industry(self.fac)
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


#
#
# class pure_cloud(object):
#     '''
#     为了测试其他不同的频率而设计的类，仅考虑了上市满60天这一要素
#     这一回测采取的方案是，对于回测频率n天，将初始资金等分成n笔，每天以1/n的资金调仓
#     每笔资金之间相互独立，最终汇聚成一个收益率序列
#     '''
#     def __init__(self,fac,freq,group=10,boxcox=1,trade_cost=0,print_comments=1,plt_plot=1,plotly_plot=0,
#                  filename='净值走势图',comments_writer=None,nets_writer=None,sheet_name=None):
#         '''n是回测的频率，等分成n份，group是回测的组数，boxcox是是否做行业市值中性化'''
#         self.fac=fac
#         self.freq=freq
#         self.group=group
#         self.boxcox=boxcox
#         self.trade_cost=trade_cost
#         moon=pure_moon()
#         moon.prerpare()
#         ages=moon.ages.copy()
#         ages=(ages>=60)+0
#         self.ages=ages.replace(0,np.nan)
#         self.closes=read_daily(close=1)
#         self.rets=((self.closes.shift(-self.freq)/self.closes-1)*self.ages-trade_cost)/self.freq
#         self.run(print_comments=print_comments,plt_plot=plt_plot,plotly_plot=plotly_plot,filename=filename)
#         if comments_writer:
#             if sheet_name:
#                 self.long_short_comments.to_excel(comments_writer,sheet_name=sheet_name)
#             else:
#                 raise AttributeError('必须制定sheet_name参数🤒')
#         if nets_writer:
#             if sheet_name:
#                 self.group_nets.to_excel(nets_writer,sheet_name=sheet_name)
#             else:
#                 raise AttributeError('必须制定sheet_name参数🤒')
#
#     def comments(self,series,series1):
#         '''对twins中的结果给出评价
#         评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率'''
#         ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
#         duration=(series.index[-1]-series.index[0]).days
#         year=duration/365
#         ret_yearly=(series.iloc[-1]/series.iloc[0])**(1/year)-1
#         max_draw=-(series/series.expanding(1).max()-1).min()
#         vol=np.std(series1)*(250**0.5)
#         sharpe=ret_yearly/vol
#         wins=series1[series1>0]
#         win_rate=len(wins)/len(series1)
#         return pd.Series([ret,ret_yearly,vol,sharpe,max_draw,win_rate],
#                          index=['总收益率','年化收益率','年化波动率','信息比率','最大回撤率','胜率'])
#
#     def run(self,print_comments,plt_plot,plotly_plot,filename):
#         '''对因子值分组并匹配'''
#         if self.boxcox:
#             self.fac=decap_industry(self.fac)
#         self.fac=self.fac.T.apply(lambda x:pd.qcut(x,self.group,labels=False,duplicates='drop')).T
#         self.fac=self.fac.shift(1)
#         self.vs=[(((self.fac==i)+0).replace(0,np.nan)*self.rets).mean(axis=1) for i in range(self.group)]
#         self.group_rets=pd.DataFrame({f'group{k}':list(v) for k,v in zip(range(1,self.group+1),self.vs)},index=self.vs[0].index)
#         self.group_rets=self.group_rets.dropna(how='all')
#         self.group_nets=(self.group_rets+1).cumprod()
#         self.group_nets=self.group_nets.apply(lambda x:x/x.iloc[0])
#         self.one=self.group_nets['group1']
#         self.end=self.group_nets[f'group{self.group}']
#         if self.one.iloc[-1]>self.end.iloc[-1]:
#             self.long_name='group1'
#             self.short_name=f'group{self.group}'
#         else:
#             self.long_name=f'group{self.group}'
#             self.short_name='group1'
#         self.long_short_ret=self.group_rets[self.long_name]-self.group_rets[self.short_name]
#         self.long_short_net=(self.long_short_ret+1).cumprod()
#         self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
#         if self.long_short_net.iloc[-1]<1:
#             self.long_short_ret=self.group_rets[self.short_name]-self.group_rets[self.long_name]
#             self.long_short_net=(self.long_short_ret+1).cumprod()
#             self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
#         self.group_rets=pd.concat([self.group_rets,self.long_short_ret.to_frame('long_short')],axis=1)
#         self.group_nets=pd.concat([self.group_nets,self.long_short_net.to_frame('long_short')],axis=1)
#         self.long_short_comments=self.comments(self.long_short_net,self.long_short_ret)
#         if print_comments:
#             print(self.long_short_comments)
#         if plt_plot:
#             self.group_nets.plot()
#             plt.savefig(filename+'.png')
#             plt.show()
#         if plotly_plot:
#             fig=pe.line(self.group_nets)
#             filename_path=filename+'.html'
#             pio.write_html(fig,filename_path,auto_open=True)


class pure_wood(object):
    """一种因子合成的方法，灵感来源于adaboost算法
    adaboost算法的精神是，找到几个分类效果较差的弱学习器，通过改变不同分类期训练时的样本的权重，
    计算每个学习器的错误概率，通过线性加权组合的方式，让弱分类期进行投票，决定最终分类结果，
    这里取通过计算各个因子多头或空头的错误概率，进而通过错误概率对因子进行加权组合，
    此处方式将分为两种，一种是主副因子的方式，另一种是全等价的方式，
    主副因子即先指定一个主因子（通常为效果更好的那个因子），然后指定若干个副因子，先计算主因子的错误概率，
    找到主因子多头里分类错误的部分，然后通过提高期加权，依次计算后续副因子的错误概率（通常按照因子效果从好到坏排序），
    最终对依次得到的错误概率做运算，然后加权
    全等价方式即不区分主副因子，分别独立计算每个因子的错误概率，然后进行加权"""

    def __init__(self, domain_fac: pd.DataFrame, subdomain_facs: list, group_num=10):
        """声明主副因子和分组数"""
        self.domain_fac = domain_fac
        self.subdomain_facs = subdomain_facs
        self.group_num = group_num
        opens = read_daily(open=1).resample("M").first()
        closes = read_daily(close=1).resample("M").last()
        self.ret_next = closes / opens - 1
        self.ret_next = self.ret_next.shift(-1)
        self.domain_fac = (
            self.domain_fac.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
        )
        self.ret_next = (
            self.ret_next.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
        )
        self.subdomain_facs = [
            i.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
            for i in self.subdomain_facs
        ]
        self.get_all_a()
        self.get_three_new_facs()

    def __call__(self, *args, **kwargs):
        return copy.copy(self.new_facs)

    def get_a_and_new_weight(self, n, fac, weight=None):
        """计算主因子的权重和权重矩阵"""
        fac_at_n = (fac == n) + 0
        ret_at_n = (self.ret_next == n) + 0
        not_nan = fac_at_n + ret_at_n
        not_nan = not_nan[not_nan.index.isin(fac_at_n.index)]
        not_nan = (not_nan > 0) + 0
        wrong = ((ret_at_n - fac_at_n) > 0) + 0
        wrong = wrong[wrong.index.isin(fac_at_n.index)]
        right = ((ret_at_n - fac_at_n) == 0) + 0
        right = right[right.index.isin(fac_at_n.index)]
        wrong = wrong * not_nan
        right = right * not_nan
        wrong = wrong.dropna(how="all")
        right = right.dropna(how="all")
        if isinstance(weight, pd.DataFrame):
            e_rate = (wrong * weight).sum(axis=1)
            a_rate = 0.5 * np.log((1 - e_rate) / e_rate)
            wrong_here = -wrong
            g_df = multidfs_to_one(right, wrong_here)
            on_exp = (g_df.T * a_rate.to_numpy()).T
            with_exp = np.exp(on_exp)
            new_weight = weight * with_exp
            new_weight = (new_weight.T / new_weight.sum(axis=1).to_numpy()).T
        else:
            e_rate = (right.sum(axis=1)) / (right.sum(axis=1) + wrong.sum(axis=1))
            a_rate = 0.5 * np.log((1 - e_rate) / e_rate)
            wrong_here = -wrong
            g_df = multidfs_to_one(right, wrong_here)
            on_exp = (g_df.T * a_rate.to_numpy()).T
            with_exp = np.exp(on_exp)
            new_weight = with_exp.copy()
            new_weight = (new_weight.T / new_weight.sum(axis=1).to_numpy()).T
        return a_rate, new_weight

    def get_all_a(self):
        """计算每个因子的a值"""
        # 第一组部分
        one_a_domain, one_weight = self.get_a_and_new_weight(1, self.domain_fac)
        one_a_list = [one_a_domain]
        for fac in self.subdomain_facs:
            one_new_a, one_weight = self.get_a_and_new_weight(1, fac, one_weight)
            one_a_list.append(one_new_a)
        self.a_list_one = one_a_list
        # 最后一组部分
        end_a_domain, end_weight = self.get_a_and_new_weight(
            self.group_num, self.domain_fac
        )
        end_a_list = [end_a_domain]
        for fac in self.subdomain_facs:
            end_new_a, end_weight = self.get_a_and_new_weight(
                self.group_num, fac, end_weight
            )
            end_a_list.append(end_new_a)
        self.a_list_end = end_a_list

    def get_three_new_facs(self):
        """分别使用第一组加强、最后一组加强、两组平均的方式结合"""
        one_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(
                    [self.domain_fac] + self.subdomain_facs, self.a_list_one
                )
            ]
        )
        end_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(
                    [self.domain_fac] + self.subdomain_facs, self.a_list_end
                )
            ]
        )
        both_fac = one_fac + end_fac
        self.new_facs = [one_fac, end_fac, both_fac]


class pure_fire(object):
    """一种因子合成的方法，灵感来源于adaboost算法
    adaboost算法的精神是，找到几个分类效果较差的弱学习器，通过改变不同分类期训练时的样本的权重，
    计算每个学习器的错误概率，通过线性加权组合的方式，让弱分类期进行投票，决定最终分类结果，
    这里取通过计算各个因子多头或空头的错误概率，进而通过错误概率对因子进行加权组合，
    此处方式将分为两种，一种是主副因子的方式，另一种是全等价的方式，
    主副因子即先指定一个主因子（通常为效果更好的那个因子），然后指定若干个副因子，先计算主因子的错误概率，
    找到主因子多头里分类错误的部分，然后通过提高期加权，依次计算后续副因子的错误概率（通常按照因子效果从好到坏排序），
    最终对依次得到的错误概率做运算，然后加权
    全等价方式即不区分主副因子，分别独立计算每个因子的错误概率，然后进行加权"""

    def __init__(self, facs: list, group_num=10):
        """声明主副因子和分组数"""
        self.facs = facs
        self.group_num = group_num
        opens = read_daily(open=1).resample("M").first()
        closes = read_daily(close=1).resample("M").last()
        self.ret_next = closes / opens - 1
        self.ret_next = self.ret_next.shift(-1)
        self.ret_next = (
            self.ret_next.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
        )
        self.facs = [
            i.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
            for i in self.facs
        ]
        self.get_all_a()
        self.get_three_new_facs()

    def __call__(self, *args, **kwargs):
        return copy.copy(self.new_facs)

    def get_a(self, n, fac):
        """计算主因子的权重和权重矩阵"""
        fac_at_n = (fac == n) + 0
        ret_at_n = (self.ret_next == n) + 0
        not_nan = fac_at_n + ret_at_n
        not_nan = not_nan[not_nan.index.isin(fac_at_n.index)]
        not_nan = (not_nan > 0) + 0
        wrong = ((ret_at_n - fac_at_n) > 0) + 0
        wrong = wrong[wrong.index.isin(fac_at_n.index)]
        right = ((ret_at_n - fac_at_n) == 0) + 0
        right = right[right.index.isin(fac_at_n.index)]
        wrong = wrong * not_nan
        right = right * not_nan
        wrong = wrong.dropna(how="all")
        right = right.dropna(how="all")
        e_rate = (right.sum(axis=1)) / (right.sum(axis=1) + wrong.sum(axis=1))
        a_rate = 0.5 * np.log((1 - e_rate) / e_rate)
        return a_rate

    def get_all_a(self):
        """计算每个因子的a值"""
        # 第一组部分
        self.a_list_one = [self.get_a(1, i) for i in self.facs]
        # 最后一组部分
        self.a_list_end = [self.get_a(self.group_num, i) for i in self.facs]

    def get_three_new_facs(self):
        """分别使用第一组加强、最后一组加强、两组平均的方式结合"""
        one_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(self.facs, self.a_list_one)
            ]
        )
        end_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(self.facs, self.a_list_end)
            ]
        )
        both_fac = one_fac + end_fac
        self.new_facs = [one_fac, end_fac, both_fac]


class pure_moonson(object):
    """行业轮动回测框架"""

    def __init__(self, fac, group_num=5):
        homeplace = HomePlace()
        pindu = (
            pd.read_feather(homeplace.daily_data_file + "各行业行情数据.feather")
            .set_index("date")
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


"""以下是更新数据库的部分"""
"""sql部分"""


def read_minute_mat(file: str = None) -> list[str, pd.DataFrame]:
    code = file[-10:-4]
    if code.startswith("0") or code.startswith("3"):
        code = code + ".SZ"
    elif code.startswith("6"):
        code = code + ".SH"
    elif code.startswith("8"):
        code = code + ".BJ"
    else:
        code = code + ".UN"
    df = list(scio.loadmat(homeplace.minute_data_file + file).values())[3]
    df = pd.DataFrame(
        df, columns=["date", "open", "high", "low", "close", "amount", "money"]
    )
    df = (
        df.groupby(["date"])
        .apply(lambda x: x.assign(num=list(range(1, len(x) + 1))))
        .reset_index(drop=True)
    )
    return code, df


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


"""clickhouse化"""


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


"""数据库1-分钟数据更新"""


class to_one(object):
    def __init__(self, minute_data):
        self.old_path = "/Users/chenzongwei/pythoncode/数据库/分钟数据/"
        self.old_files = os.listdir(self.old_path)
        self.title = ["date", "open", "high", "low", "close", "amount", "money"]
        self.df = minute_data
        logger.info("concat success")
        self.df.columns = [
            "index",
            "high",
            "open",
            "close",
            "volume",
            "windcode",
            "low",
            "amount",
        ]
        self.df = self.unify_dataframe(self.df)
        self.prefix = "UnAdjstedStockMinute_"
        self.suffix = ".mat"
        logger.info("unify success")
        tqdm.tqdm.pandas()

    def refresh_file(self, df):
        code = df.windcode.iloc[0]
        if "." in code:
            code = code.split(".")[0]
        df = df.drop(columns=["windcode"])
        try:
            the_file = [i for i in self.old_files if code in i][0]
            the_file = "/".join([self.old_path, the_file])
            data = list(scio.loadmat(the_file).values())[3]
            data = pd.DataFrame(data, columns=self.title)
            data = pd.concat([data, df])
            scio.savemat(the_file, {"data": data.to_numpy()}, do_compression=True)
        except Exception:
            new_file = self.old_path + "/" + self.prefix + code + self.suffix
            scio.savemat(new_file, {"date": df.to_numpy()}, do_compression=True)
            logger.warning(f"{code} is new")

    def unify_dataframe(self, df):
        df["index"] = pd.to_datetime(df["index"])
        df["index"] = df["index"].dt.strftime("%Y%m%d").astype(int)
        # df['index']=df['index'].apply(int)
        df = df[
            ["index", "windcode", "open", "high", "low", "close", "volume", "amount"]
        ]
        df = df.rename(columns={"index": "date", "volume": "amount", "amount": "money"})
        return df

    def run(self):
        self.df.groupby("windcode").progress_apply(self.refresh_file)
        logger.success("everything is okay")


@retry
def download_single_minute(stock, startdate, enddate):
    try:
        df = pro.stk_mins(
            code=stock,
            freq="1min",
            start_date=startdate,
            end_date=enddate,
            fields="code,tradetime,openprice,highprice,lowprice,closeprice,volume,value",
        )

        return df
    except Exception:
        time.sleep(60)
        df = pro.stk_mins(
            code=stock,
            freq="1min",
            start_date=startdate,
            end_date=enddate,
            fields="code,tradetime,openprice,highprice,lowprice,closeprice,volume,value",
        )
        return df


@retry
def download_stock_list():
    try:
        df = pro.stock_basic(list_status="L", fields="ts_code")
        return df
    except Exception:
        time.sleep(60)
        df = pro.stock_basic(list_status="L", fields="ts_code")
        return df


def database_update_minute_files(
    startdate: str = None, enddate: str = None, to_mat=True, to_clickhouse=False
):
    """
    更新数据库中的分钟数据，startdate形如'20220501'，enddate形如'20220701'
    """
    # 获取股票列表
    homeplace = HomePlace()
    config = pickledb.load(homeplace.update_data_file + "database_config.db", False)
    s = sqlConfig(db_name="minute_data")
    sa = sqlConfig(db_name="minute_data_alter")
    if to_clickhouse:
        chc = ClickHouseClient(database_name="minute_data")
    if startdate:
        ...
    else:
        startdate = config.get("minute_enddate")
        logger.info(
            f"上次更新到{datetime.datetime.strftime(pd.Timestamp(startdate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
        )
    if enddate:
        ...
    else:
        enddate = datetime.datetime.now()
        if enddate.hour < 17:
            ...
        else:
            enddate = enddate + pd.Timedelta(days=1)
        enddate = datetime.datetime.strftime(enddate, "%Y%m%d")
        logger.info(
            f"本次将更新到{datetime.datetime.strftime(pd.Timestamp(enddate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
        )
    data = download_stock_list()
    stocks = list(data.iloc[:, 0])
    calen = download_calendar(startdate=startdate, enddate=enddate)
    calen = list(map(int, sorted(list(set(calen.trade_date)))))

    if to_mat:
        minute_data = []
    minute_data_sql = []
    for stock in tqdm.tqdm(stocks, desc="正在下载分钟数据"):
        df = download_single_minute(stock, startdate, enddate)
        df_sql = df.rename(
            columns={
                "tradetime": "date",
                "openprice": "open",
                "highprice": "high",
                "lowprice": "low",
                "closeprice": "close",
                "volume": "amount",
                "value": "money",
            }
        )
        df_sql = df_sql[["date", "open", "high", "low", "close", "amount", "money"]]
        df_sql = df_sql.sort_values(["date"])
        df_sql.date = pd.to_datetime(df_sql.date).dt.strftime("%Y%m%d")
        df_sql.date = df_sql.date.astype(int)
        df_sql = (
            df_sql.groupby(["date"])
            .apply(lambda x: x.assign(num=list(range(1, len(x) + 1))))
            .reset_index(drop=True)
        )
        df_sql = df_sql.where(df_sql < 1e38, np.nan)
        df_sql = df_sql.where(df_sql > -1e38, np.nan)
        try:
            df_sql.to_sql(
                name=stock,
                con=s.engine,
                if_exists="append",
                index=False,
                dtype={
                    "date": INT,
                    "open": FLOAT(2),
                    "high": FLOAT(2),
                    "low": FLOAT(2),
                    "close": FLOAT(2),
                    "amount": INT,
                    "money": FLOAT(2),
                    "num": INT,
                },
            )
        except Exception:
            if s.get_data(stock).shape[0] == 0:
                df_sql.to_sql(
                    name=stock,
                    con=s.engine,
                    if_exists="replace",
                    index=False,
                    dtype={
                        "date": INT,
                        "open": FLOAT(2),
                        "high": FLOAT(2),
                        "low": FLOAT(2),
                        "close": FLOAT(2),
                        "amount": INT,
                        "money": FLOAT(2),
                        "num": INT,
                    },
                )
            else:
                logger.error(f"{stock}写入mysql时发生错误，请更新完成后复查")
        df_sql = df_sql.assign(code=stock)
        minute_data_sql.append(df_sql)
        if to_mat:
            minute_data.append(df)
    minute_data_sql = pd.concat(minute_data_sql)
    for day in calen[:-1]:
        m = minute_data_sql[minute_data_sql.date == day]
        m = m.drop(columns=["date"])
        m.to_sql(
            name=str(day),
            con=sa.engine,
            if_exists="replace",
            index=False,
            dtype={
                "open": FLOAT(2),
                "high": FLOAT(2),
                "low": FLOAT(2),
                "close": FLOAT(2),
                "amount": INT,
                "money": FLOAT(2),
                "num": INT,
                "code": VARCHAR(9),
            },
        )
    if to_clickhouse:
        minute_data_sql = np.around(minute_data_sql, 2)
        minute_data_sql["date"] = (minute_data_sql["date"] * 100).astype(int).ffill()
        minute_data_sql["open"] = (minute_data_sql["open"] * 100).astype(int).ffill()
        minute_data_sql["high"] = (minute_data_sql["high"] * 100).astype(int).ffill()
        minute_data_sql["low"] = (minute_data_sql["low"] * 100).astype(int).ffill()
        minute_data_sql["close"] = (minute_data_sql["close"] * 100).astype(int).ffill()
        minute_data_sql["amount"] = (
            (minute_data_sql["amount"] * 100).astype(int).ffill()
        )
        minute_data_sql["money"] = (minute_data_sql["money"] * 100).astype(int).ffill()
        minute_data_sql["num"] = (minute_data_sql["num"] * 100).astype(int).ffill()
        minute_data_sql.to_sql(
            "minute_data", chc.engine, if_exists="append", index=False
        )

    if to_mat:
        minute_data = [i.iloc[::-1, :] for i in minute_data]
        minute_data = pd.concat(minute_data)

        one = to_one(minute_data)
        one.run()
    config.set("minute_enddate", enddate)
    config.set("data_refresh", "ready")
    config.dump()
    logger.success(
        f"分钟数据已更新，现在最新的是{datetime.datetime.strftime(pd.Timestamp(enddate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
    )


"""米筐更新分钟数据到clickhouse"""


def database_update_minute_data_to_clickhouse(kind: str) -> pd.DataFrame:
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
    chc = ClickHouseClient("minute_data")
    last_date = max(chc.show_all_dates(f"minute_data_{kind}"))
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
    ts = (
        (np.around(ts.set_index("code"), 2) * 100)
        .ffill()
        .dropna()
        .astype(int)
        .reset_index()
    )
    ts.code = ts.code.str.replace(".XSHE", ".SZ")
    ts.code = ts.code.str.replace(".XSHG", ".SH")
    # 数据写入数据库
    ts.to_sql(f"minute_data_{kind}", chc.engine, if_exists="append", index=False)
    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")


"""米筐更新分钟数据到mysql"""


def database_update_minute_data_to_mysql(kind: str) -> pd.DataFrame:
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
    # 连接4个数据库
    sqls = sqlConfig("minute_data_stock")
    sqlsa = sqlConfig("minute_data_stock_alter")
    sqli = sqlConfig("minute_data_index")
    sqlia = sqlConfig("minute_data_index_alter")
    last_date = max(sqlsa.show_tables(full=False))
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
    ts = (
        (np.around(ts.set_index("code"), 2) * 100)
        .ffill()
        .dropna()
        .astype(int)
        .reset_index()
    )
    ts.code = ts.code.str.replace(".XSHE", ".SZ")
    ts.code = ts.code.str.replace(".XSHG", ".SH")
    codes = list(set(ts.code))
    dates = list(set(ts.date))
    # 数据写入数据库
    fails = []
    # 股票
    if kind == "stock":
        # 写入每只股票一张表
        for code in codes:
            dfi = ts[ts.date == date]
            try:
                dfi.drop(columns=["code"]).to_sql(
                    name=code,
                    con=sqls.engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        "date": INT,
                        "open": INT,
                        "high": INT,
                        "low": INT,
                        "close": INT,
                        "amount": BIGINT,
                        "money": BIGINT,
                        "num": INT,
                    },
                )
            except Exception:
                try:
                    if sqls.get_data(code).shape[0] == 0:
                        dfi.drop(columns=["code"]).to_sql(
                            name=code,
                            con=sqls.engine,
                            if_exists="replace",
                            index=False,
                            dtype={
                                "date": INT,
                                "open": INT,
                                "high": INT,
                                "low": INT,
                                "close": INT,
                                "amount": BIGINT,
                                "money": BIGINT,
                                "num": INT,
                            },
                        )
                except Exception:
                    fails.append(code)
                    logger.warning(f"股票{code}写入失败了，请检查")
        # 把每天写入每天所有股票一张表
        for date in dates:
            dfi = ts[ts.date == date]
            try:
                dfi.drop(columns=["date"]).to_sql(
                    name=str(date),
                    con=sqlsa.engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        "code": VARCHAR(9),
                        "open": INT,
                        "high": INT,
                        "low": INT,
                        "close": INT,
                        "amount": BIGINT,
                        "money": BIGINT,
                        "num": INT,
                    },
                )
            except Exception:
                try:
                    if sqlsa.get_data(date).shape[0] == 0:
                        dfi.drop(columns=["date"]).to_sql(
                            name=str(date),
                            con=sqlsa.engine,
                            if_exists="replace",
                            index=False,
                            dtype={
                                "code": VARCHAR(9),
                                "open": INT,
                                "high": INT,
                                "low": INT,
                                "close": INT,
                                "amount": BIGINT,
                                "money": BIGINT,
                                "num": INT,
                            },
                        )
                except Exception:
                    fails.append(date)
                    logger.warning(f"股票{date}写入失败了，请检查")
    # 指数
    else:
        # 写入每个指数一张表
        for code in codes:
            dfi = ts[ts.date == date]
            try:
                dfi.drop(columns=["code"]).to_sql(
                    name=code,
                    con=sqli.engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        "date": INT,
                        "open": INT,
                        "high": INT,
                        "low": INT,
                        "close": INT,
                        "amount": BIGINT,
                        "money": BIGINT,
                        "num": INT,
                    },
                )
            except Exception:
                try:
                    if sqli.get_data(code).shape[0] == 0:
                        dfi.drop(columns=["code"]).to_sql(
                            name=code,
                            con=sqli.engine,
                            if_exists="replace",
                            index=False,
                            dtype={
                                "date": INT,
                                "open": INT,
                                "high": INT,
                                "low": INT,
                                "close": INT,
                                "amount": BIGINT,
                                "money": BIGINT,
                                "num": INT,
                            },
                        )
                except Exception:
                    fails.append(code)
                    logger.warning(f"指数{code}写入失败了，请检查")
        # 把每天写入每天所有指数一张表
        for date in dates:
            dfi = ts[ts.date == date]
            try:
                dfi.drop(columns=["date"]).to_sql(
                    name=str(date),
                    con=sqlia.engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        "code": VARCHAR(9),
                        "open": INT,
                        "high": INT,
                        "low": INT,
                        "close": INT,
                        "amount": BIGINT,
                        "money": BIGINT,
                        "num": INT,
                    },
                )
            except Exception:
                try:
                    if sqlia.get_data(date).shape[0] == 0:
                        dfi.drop(columns=["date"]).to_sql(
                            name=str(date),
                            con=sqlia.engine,
                            if_exists="replace",
                            index=False,
                            dtype={
                                "code": VARCHAR(9),
                                "open": INT,
                                "high": INT,
                                "low": INT,
                                "close": INT,
                                "amount": BIGINT,
                                "money": BIGINT,
                                "num": INT,
                            },
                        )
                except Exception:
                    fails.append(date)
                    logger.warning(f"指数{date}写入失败了，请检查")

    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")


"""查询流量使用情况"""


def rqdatac_show_used():
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    print(f"今日已使用rqsdk流量{user2}MB")
    return user2


"""日频数据更新"""


@retry
def download_single_daily(day):
    """更新单日的数据"""
    try:
        # 8个价格，交易状态，成交量，
        df1 = pro.a_daily(
            trade_date=day,
            fields=[
                "code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjopen",
                "adjclose",
                "adjhigh",
                "adjlow",
                "tradestatus",
            ],
        )
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2 = pro.daily_basic(
            trade_date=day,
            fields=["ts_code", "trade_date", "turnover_rate_f", "float_share"],
        )
        time.sleep(1)
        return df1, df2
    except Exception:
        time.sleep(60)
        # 8个价格，交易状态，成交量，
        df1 = pro.a_daily(
            trade_date=day,
            fields=[
                "code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjopen",
                "adjclose",
                "adjhigh",
                "adjlow",
                "tradestatus",
            ],
        )
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2 = pro.daily_basic(
            trade_date=day,
            fields=["ts_code", "trade_date", "turnover_rate_f", "float_share"],
        )
        time.sleep(1)
        return df1, df2


@retry
def download_calendar(startdate, enddate):
    """更新单日的数据"""
    try:
        # 交易日历
        df0 = pro.a_calendar(start_date=startdate, end_date=enddate)
        time.sleep(1)
        return df0
    except Exception:
        time.sleep(60)
        # 交易日历
        df0 = pro.a_calendar(start_date=startdate, end_date=enddate)
        time.sleep(1)
        return df0


def database_update_daily_files(startdate: str = None, enddate: str = None):
    """
    更新数据库中的日频数据，startdate形如'20220501'，enddate形如'20220701'
    """
    read_daily.clear_cache()
    homeplace = HomePlace()
    config = pickledb.load(homeplace.update_data_file + "database_config.db", False)
    if startdate:
        ...
    else:
        startdate = config.get("daily_enddate")
        logger.info(
            f"上次更新到{datetime.datetime.strftime(pd.Timestamp(startdate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
        )
    if enddate:
        ...
    else:
        enddate = datetime.datetime.now()
        if enddate.hour < 17:
            enddate = enddate - pd.Timedelta(days=1)
        else:
            ...
        enddate = datetime.datetime.strftime(enddate, "%Y%m%d")
        logger.info(
            f"本次将更新到{datetime.datetime.strftime(pd.Timestamp(enddate),format='%Y-%m-%d')}"
        )
    # 交易日历
    df0 = download_calendar(startdate, enddate)
    tradedates = sorted(list(set(df0.trade_date)))
    if len(tradedates) > 1:
        # 存储每天数据
        df1s = []
        df2s = []
        for day in tqdm.tqdm(tradedates, desc="正在下载日频数据"):
            df1, df2 = download_single_daily(day)
            df1s.append(df1)
            df2s.append(df2)
        # 8个价格，交易状态，成交量，
        df1s = pd.concat(df1s)
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2s = pd.concat(df2s)
    elif len(tradedates) == 1:
        df1s, df2s = download_single_daily(tradedates[0])
    else:
        raise ValueError("从上次更新到这次更新，还没有经过交易日。放假就好好休息吧，别跑代码了🤒")
    df1s.tradestatus = (df1s.tradestatus == "交易") + 0
    df2s = df2s.rename(columns={"ts_code": "code"})
    df1s.trade_date = df1s.trade_date.apply(int)
    df2s.trade_date = df2s.trade_date.apply(int)
    both_codes = list(set(df1s.code) & set(df2s.code))
    df1s = df1s[df1s.code.isin(both_codes)]
    df2s = df2s[df2s.code.isin(both_codes)]
    # st股
    df3 = pro.ashare_st()

    suffix_path = homeplace.update_data_file
    suffix_path_save = homeplace.daily_data_file
    codes = list(scio.loadmat(suffix_path_save + "AllStockCode.mat").values())[3]
    codes = [i[0] for i in codes[0]]
    days = list(scio.loadmat(suffix_path_save + "TradingDate_Daily.mat").values())[3]
    days = [i[0] for i in days]

    def read_mat(path):
        col = list(scio.loadmat(suffix_path_save + "AllStockCode.mat").values())[3]
        index = list(scio.loadmat(suffix_path_save + "TradingDate_Daily.mat").values())[
            3
        ]
        col = [i[0] for i in col[0]]
        if len(index) > 1000:
            index = [i[0] for i in index]
        else:
            index = index[0]
        path = suffix_path_save + path
        data = list(scio.loadmat(path).values())[3]
        data = pd.DataFrame(data, index=index, columns=col)
        return data

    def to_mat(df, row, filename=None, ind="trade_date", col="code"):
        df = df[[ind, col, row]].set_index([ind, col])
        df = df.unstack()
        df.columns = [i[1] for i in list(df.columns)]
        old = read_mat(filename)
        new = pd.concat([old, df])
        scio.savemat(
            suffix_path_save + filename, {"data": new.to_numpy()}, do_compression=True
        )
        logger.success(filename + "已更新")
        return new

    # 股票日行情（未复权高开低收，复权高开低收，交易状态，成交量）
    part1 = df1s.copy()
    # 未复权开盘价
    opens = to_mat(part1, "open", "AllStock_DailyOpen.mat")
    # 未复权最高价
    highs = to_mat(part1, "high", "AllStock_DailyHigh.mat")
    # 未复权最低价
    lows = to_mat(part1, "low", "AllStock_DailyLow.mat")
    # 未复权收盘价
    closes = to_mat(part1, "close", "AllStock_DailyClose.mat")
    # 成交量
    volumes = to_mat(part1, "volume", "AllStock_DailyVolume.mat")
    # 复权开盘价
    diopens = to_mat(part1, "adjopen", "AllStock_DailyOpen_dividend.mat")
    # 复权最高价
    dihighs = to_mat(part1, "adjhigh", "AllStock_DailyHigh_dividend.mat")
    # 复权最低价
    dilows = to_mat(part1, "adjlow", "AllStock_DailyLow_dividend.mat")
    # 复权收盘价
    dicloses = to_mat(part1, "adjclose", "AllStock_DailyClose_dividend.mat")
    # 交易状态
    status = to_mat(part1, "tradestatus", "AllStock_DailyStatus.mat")

    # 换手率
    part2 = df2s[["trade_date", "code", "turnover_rate_f"]]
    part2 = part2.set_index(["trade_date", "code"]).unstack()
    part2.columns = [i[1] for i in list(part2.columns)]
    part2 = part2 / 100
    part2_old = read_mat("AllStock_DailyTR.mat")
    part2_new = pd.concat([part2_old, part2])
    # part2_new=part2_new.dropna(how='all',axis=1)
    part2_new = part2_new[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyTR.mat",
        {"data": part2_new.to_numpy()},
        do_compression=True,
    )
    logger.success("换手率更新完成")

    # #交易日历和股票代码
    # part2_new=part2_new.reset_index()
    # part2_new.columns=['date']+list(part2_new.columns)[1:]
    # part2_new.to_feather('日历与代码暂存.feather')

    # 流通股数
    # 读取新的流通股变动数
    part3 = df2s[["trade_date", "code", "float_share"]]
    part3 = part3.set_index(["trade_date", "code"]).unstack()
    part3.columns = [i[1] for i in list(part3.columns)]
    part3 = part3 * 10000
    part3_old = read_mat("AllStock_DailyAShareNum.mat")
    part3_new = pd.concat([part3_old, part3])
    # part2_new=part2_new.dropna(how='all',axis=1)
    part3_new = part3_new[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyAShareNum.mat",
        {"data": part3_new.to_numpy()},
        do_compression=True,
    )
    logger.success("流通股数更新完成")

    # st
    part4 = df3[["s_info_windcode", "entry_dt", "remove_dt"]]
    part4 = part4.sort_values("s_info_windcode")
    part4.remove_dt = part4.remove_dt.fillna(enddate).astype(int)
    part4 = part4.set_index("s_info_windcode").stack()
    part4 = part4.reset_index().assign(
        he=sorted(list(range(int(part4.shape[0] / 2))) * 2)
    )
    part4 = part4.drop(columns=["level_1"])
    part4.columns = ["code", "date", "he"]
    part4.date = pd.to_datetime(part4.date, format="%Y%m%d")

    def single(df):
        full = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())})
        df = pd.merge(full, df, on=["date"], how="left")
        df = df.fillna(method="ffill")
        return df

    tqdm.tqdm.pandas()
    part4 = part4.groupby(["code", "he"]).progress_apply(single)
    part4.date = part4.date.dt.strftime("%Y%m%d").astype(int)
    part4 = part4[part4.date.isin(list(part2_new.index))]
    part4 = part4.reset_index(drop=True)
    part4 = part4.assign(st=1)

    part4 = part4.drop_duplicates(subset=["date", "code"]).pivot(
        index="date", columns="code", values="st"
    )

    part4_0 = pd.DataFrame(0, columns=part2_new.columns, index=part2_new.index)
    part4_0 = part4_0 + part4
    # old=read_mat('AllStock_DailyST.mat')
    # part4_new=pd.concat([old,part4_0])
    part4_0 = part4_0.replace(np.nan, 0)
    part4_0 = part4_0[part4_0.index.isin(list(part2_new.index))]
    part4_0 = part4_0.T
    part4_0 = part4_0[part4_0.index.isin(list(part2_new.columns))]
    part4_0 = part4_0.T
    # part4_0=part4_0.dropna(how='all',axis=1)
    part4_0 = part4_0[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyST.mat",
        {"data": part4_0.to_numpy()},
        do_compression=True,
    )
    logger.success("st更新完了")

    # 上市天数
    part5_close = pd.read_feather(suffix_path + "BasicFactor_Close.txt").set_index(
        "index"
    )
    part5_close = part5_close[part5_close.index < 20040101]
    part5_close = pd.concat([part5_close, closes])
    # part5_close.reset_index().to_feather(suffix_path+'BasicFactor_Close.txt')
    part5 = np.sign(part5_close).fillna(method="ffill").cumsum()
    part5 = part5[part5.index.isin(list(part2_new.index))]
    part5 = part5.T
    part5 = part5[part5.index.isin(list(part2_new.columns))]
    part5 = part5.T
    # part5=part5.dropna(how='all',axis=1)
    part5 = part5[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyListedDate.mat",
        {"data": part5.to_numpy()},
        do_compression=True,
    )
    logger.success("上市天数更新完了")

    # 交易日历和股票代码
    scio.savemat(
        suffix_path_save + "TradingDate_Daily.mat",
        {"data": part2_new.index.to_numpy()},
        do_compression=True,
    )
    scio.savemat(
        suffix_path_save + "AllStockCode.mat",
        {"data": part2_new.columns.to_numpy()},
        do_compression=True,
    )
    enddate = pd.Timestamp(enddate) + pd.Timedelta(days=1)
    enddate = datetime.datetime.strftime(enddate, "%Y%m%d")
    config.set("daily_enddate", enddate)
    config.dump()
    logger.success("交易日历和股票代码更新完了")
    read_daily.clear_cache()
    logger.success(
        f"日频数据已更新，现在最新的是{datetime.datetime.strftime(pd.Timestamp(enddate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
    )


def add_suffix(code):
    if code.startswith("0") or code.startswith("3"):
        code = code + ".SZ"
    elif code.startswith("6"):
        code = code + ".SH"
    elif code.startswith("8"):
        code = code + ".BJ"
    else:
        code = code + ".UN"
    return code


"""更新个股风格暴露数据（barra）"""


@retry
def download_single_day_style(day):
    """更新单日的数据"""
    try:
        style = pro.RMExposureDayGet(
            trade_date=str(day),
            fields="tradeDate,ticker,BETA,MOMENTUM,SIZE,EARNYILD,RESVOL,GROWTH,BTOP,LEVERAGE,LIQUIDTY,SIZENL",
        )
        time.sleep(1)
        return style
    except Exception:
        time.sleep(60)
        style = pro.RMExposureDayGet(
            trade_date=str(day),
            fields="tradeDate,ticker,BETA,MOMENTUM,SIZE,EARNYILD,RESVOL,GROWTH,BTOP,LEVERAGE,LIQUIDTY,SIZENL",
        )
        time.sleep(1)
        return style


def database_update_barra_files():
    fs = os.listdir(homeplace.barra_data_file)[0]
    fs = pd.read_feather(homeplace.barra_data_file + fs)
    fs.columns = ["date"] + list(fs.columns)[1:]
    fs = fs.set_index("date")
    last_date = fs.index.max()
    last_date = datetime.datetime.strftime(last_date, "%Y%m%d")
    now = datetime.datetime.now()
    if now.hour < 17:
        now = now - pd.Timedelta(days=1)
    now = datetime.datetime.strftime(now, "%Y%m%d")
    logger.info(f"风格暴露数据上次更新到{last_date}，本次将更新到{now}")
    df0 = download_calendar(last_date, now)
    tradedates = sorted(list(set(df0.trade_date)))
    style_names = [
        "beta",
        "momentum",
        "size",
        "residualvolatility",
        "earningsyield",
        "growth",
        "booktoprice",
        "leverage",
        "liquidity",
        "nonlinearsize",
    ]
    ds = {k: [] for k in style_names}
    for t in tqdm.tqdm(tradedates):
        style = download_single_day_style(t)
        style.columns = style.columns.str.lower()
        style = style.rename(
            columns={
                "earnyild": "earningsyield",
                "tradedate": "date",
                "ticker": "code",
                "resvol": "residualvolatility",
                "btop": "booktoprice",
                "sizenl": "nonlinearsize",
                "liquidty": "liquidity",
            }
        )
        style.date = pd.to_datetime(style.date, format="%Y%m%d")
        style.code = style.code.apply(add_suffix)
        sts = list(style.columns)[2:]
        for s in sts:
            ds[s].append(style.pivot(columns="code", index="date", values=s))
    for k, v in ds.items():
        old = pd.read_feather(homeplace.barra_data_file + k + ".feather")
        old.columns = ["date"] + list(old.columns)[1:]
        old = old.set_index("date")
        new = pd.concat(v)
        new = pd.concat([old, new])
        new.reset_index().to_feather(homeplace.barra_data_file + k + ".feather")
    logger.success(f"风格暴露数据已经更新到{now}")


"""更新300、500、1000行情数据"""


@retry
def download_single_index(index_code: str):
    if index_code == "000300.SH":
        file = "沪深300"
    elif index_code == "000905.SH":
        file = "中证500"
    elif index_code == "000852.SH":
        file = "中证1000"
    else:
        file = index_code
    try:
        df = (
            pro.index_daily(ts_code=index_code)
            .sort_values("trade_date")
            .rename(columns={"trade_date": "date"})
        )
        df = df[["date", "close"]]
        df.date = pd.to_datetime(df.date, format="%Y%m%d")
        df = df.set_index("date")
        df = df.resample("M").last()
        df.columns = [file]
        return df
    except Exception:
        logger.warning("封ip了，请等待1分钟")
        time.sleep(60)
        df = (
            pro.index_daily(ts_code=index_code)
            .sort_values("trade_date")
            .rename(columns={"trade_date": "date"})
        )
        df = df[["date", "close"]]
        df.date = pd.to_datetime(df.date, format="%Y%m%d")
        df = df.set_index("date")
        df = df.resample("M").last()
        df.columns = [file]
        return df


def database_update_index_three():
    """读取三大指数的原始行情数据，返回并保存在本地"""
    hs300 = download_single_index("000300.SH")
    zz500 = download_single_index("000905.SH")
    zz1000 = download_single_index("000852.SH")
    res = pd.concat([hs300, zz500, zz1000], axis=1)
    new_date = datetime.datetime.strftime(res.index.max(), "%Y%m%d")
    res.reset_index().to_feather(homeplace.daily_data_file + "3510行情.feather")
    logger.success(f"3510行情数据已经更新至{new_date}")


"""更新申万一级行业的行情"""


@retry
def download_single_industry_price(ind):
    try:
        df = pro.swindex_daily(code=ind)[["trade_date", "close"]].set_index(
            "trade_date"
        )
        df.columns = [ind]
        time.sleep(1)
        return df
    except Exception:
        time.sleep(60)
        df = pro.swindex_daily(code=ind)[["trade_date", "close"]].set_index(
            "trade_date"
        )
        df.columns = [ind]
        time.sleep(1)
        return df


def database_update_industry_prices():
    indus = []
    for ind in list(INDUS_DICT.keys()):
        df = download_single_industry_price(ind=ind)
        indus.append(df)
    indus = pd.concat(indus, axis=1).reset_index()
    indus.columns = ["date"] + list(indus.columns)[1:]
    indus = indus.set_index("date")
    indus = indus.dropna()
    indus.index = pd.to_datetime(indus.index, format="%Y%m%d")
    indus = indus.sort_index()
    indus.reset_index().to_feather("各行业行情数据.feather")
    new_date = datetime.datetime.strftime(indus.index.max(), "%Y%m%d")
    logger.success(f"申万一级行业的行情数据已经更新至{new_date}")


"""更新申万一级行业哑变量"""


@retry
def download_single_industry_member(ind):
    try:
        df = pro.index_member(index_code=ind)
        # time.sleep(1)
        return df
    except Exception:
        time.sleep(60)
        df = pro.index_member(index_code=ind)
        time.sleep(1)
        return df


def 生成每日分类表(df, code, entry, exit, kind):
    """df是要包含任意多列的表格，为dataframe格式，主要内容为，每一行是
    一只股票或一只基金的代码、分类、进入该分类的时间、移除该分类的时间，
    除此之外，还可以包含很多其他内容
    code是股票代码列的列名，为字符串格式；
    entry是股票进入该分类的日期的列名，为字符串格式
    exit是股票退出该分类的日期的列名，为字符串格式
    kind是分类列的列名，为字符串格式"""
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


def database_update_industry_member():
    dfs = []
    for ind in tqdm.tqdm(INDUS_DICT.keys()):
        ff = download_single_industry_member(ind)
        ff = 生成每日分类表(ff, "con_code", "in_date", "out_date", "index_code")
        khere = ff.index_code.iloc[0]
        ff = ff.assign(khere=1)
        ff.columns = ["code", "index_code", "date", khere]
        ff = ff.drop(columns=["index_code"])
        dfs.append(ff)
    res = pd.merge(dfs[0], dfs[1], on=["code", "date"])
    dfs = pd.concat(dfs)
    trs = read_daily(tr=1, start=20040101)
    dfs = dfs[dfs.date.isin(trs.index)]
    dfs = dfs.sort_values(["date", "code"])
    dfs = dfs[["date", "code"] + list(dfs.columns)[2:]]
    dfs = dfs.fillna(0)
    dfs.reset_index(drop=True).to_feather(
        homeplace.daily_data_file + "申万行业2021版哑变量.feather"
    )
    new_date = dfs.date.max()
    new_date = datetime.datetime.strftime(new_date, "%Y%m%d")
    logger.success(f"申万一级行业成分股(哑变量)已经更新至{new_date}")


"""更新3510指数成分股"""


@retry
def download_single_index_member_monthly(code):
    file = homeplace.daily_data_file + INDEX_DICT[code] + "月成分股.feather"
    old = pd.read_feather(file).set_index("index")
    old_date = old.index.max()
    start_date = old_date + pd.Timedelta(days=1)
    end_date = datetime.datetime.now()
    if start_date >= end_date:
        logger.info(f"{INDEX_DICT[code]}月成分股无需更新，上次已经更新到了{start_date}")
    else:
        start_date, end_date = datetime.datetime.strftime(
            start_date, "%Y%m%d"
        ), datetime.datetime.strftime(end_date, "%Y%m%d")
        logger.info(f"{INDEX_DICT[code]}月成分股上次更新到{start_date},本次将更新到{end_date}")
        try:
            a = pro.index_weight(
                index_code=code, start_date=start_date, end_date=end_date
            )
            if a.shape[0] == 0:
                logger.info(f"{INDEX_DICT[code]}月成分股无需更新，上次已经更新到了{start_date}")
            else:
                time.sleep(1)
                a.trade_date = pd.to_datetime(a.trade_date, format="%Y%m%d")
                a = a.sort_values("trade_date").set_index("trade_date")
                a = (
                    a.groupby("con_code")
                    .resample("M")
                    .last()
                    .drop(columns=["con_code"])
                    .reset_index()
                )
                a = a.assign(num=1)
                a = (
                    a[["trade_date", "con_code", "num"]]
                    .rename(columns={"trade_date": "date", "con_code": "code"})
                    .pivot(columns="code", index="date", values="num")
                )
                a = pd.concat([old, a]).fillna(0)
                a.reset_index().to_feather(file)
                logger.success(f"已将{INDEX_DICT[code]}月成分股更新至{end_date}")
        except Exception:
            time.sleep(60)
            a = pro.index_weight(
                index_code=code, start_date=start_date, end_date=end_date
            )
            if a.shape[0] == 0:
                logger.info(f"{INDEX_DICT[code]}月成分股无需更新，上次已经更新到了{start_date}")
            else:
                time.sleep(1)
                a.trade_date = pd.to_datetime(a.trade_date, format="%Y%m%d")
                a = a.sort_values("trade_date").set_index("trade_date")
                a = (
                    a.groupby("con_code")
                    .resample("M")
                    .last()
                    .drop(columns=["con_code"])
                    .reset_index()
                )
                a = a.assign(num=1)
                a = (
                    a[["trade_date", "con_code", "num"]]
                    .rename(columns={"trade_date": "date", "con_code": "code"})
                    .pivot(columns="code", index="date", values="num")
                )
                a = pd.concat([old, a]).fillna(0)
                a.reset_index().to_feather(file)
                logger.success(f"已将{INDEX_DICT[code]}月成分股更新至{end_date}")


def database_update_index_members_monthly():
    for k in list(INDEX_DICT.keys()):
        download_single_index_member_monthly(k)


def download_single_index_member(code):
    file = homeplace.daily_data_file + INDEX_DICT[code] + "日成分股.feather"
    if code.endswith(".SH"):
        code = code[:6] + ".XSHG"
    elif code.endswith(".SZ"):
        code = code[:6] + ".XSHE"
    now = datetime.datetime.now()
    df = rqdatac.index_components(
        code, start_date="20100101", end_date=now, market="cn"
    )
    ress = []
    for k, v in df.items():
        res = pd.DataFrame(1, index=[pd.Timestamp(k)], columns=v)
        ress.append(res)
    ress = pd.concat(ress)
    ress.columns = [convert_code(i)[0] for i in list(ress.columns)]
    tr = np.sign(read_daily(tr=1, start=20100101))
    rt = np.sign(tr + ress)
    now_str = datetime.datetime.strftime(now, "%Y%m%d")
    tr.reset_index().to_feather(file)
    logger.success(f"已将{INDEX_DICT[convert_code(code)[0]]}日成分股更新至{now_str}")


def database_update_index_members():
    for k in list(INDEX_DICT.keys()):
        download_single_index_member(k)


"""保存最终因子值"""


def database_save_final_factors(df: pd.DataFrame, name: str, order: int):
    """保存最终因子的因子值"""
    homeplace = HomePlace()
    path = homeplace.final_factor_file + name + "_" + "多因子" + str(order) + ".feather"
    df.reset_index().to_feather(path)
    final_date = df.index.max()
    final_date = datetime.datetime.strftime(final_date, "%Y%m%d")
    config = pickledb.load(homeplace.update_data_file + "database_config.db", False)
    config.set("data_refresh", "done")
    config.dump()
    logger.success(f"今日计算的因子值保存，最新一天为{final_date}")


"""读取初级&最终因子值"""


def database_read_final_factors(
    name: str = None, order: int = None, output=True, new=False
) -> tuple[pd.DataFrame, str]:
    """根据因子名字，或因子序号，读取最终因子的因子值"""
    homeplace = HomePlace()
    facs = os.listdir(homeplace.final_factor_file)
    if name is None and order is None:
        raise IOError("请指定因子名字或者因子序号")
    elif name is None and order is not None:
        key = "多因子" + str(order)
        ans = [i for i in facs if key in i][0]
    elif name is not None and name is None:
        key = name
        ans = [i for i in facs if key in i]
        if len(ans) > 0:
            ans = ans[0]
        else:
            raise IOError(f"您名字记错了，不存在叫{name}的因子")
    else:
        key1 = name
        key2 = "多因子" + str(order)
        ans1 = [i for i in facs if key1 in i]
        if len(ans1) > 0:
            ans1 = ans1[0]
        else:
            raise IOError(f"您名字记错了，不存在叫{name}的因子")
        ans2 = [i for i in facs if key2 in i][0]
        if ans1 != ans2:
            ans = ans1
            logger.warning("您输入的名字和序号不一致，怀疑您记错了序号，程序默认以名字为准了哈")
        else:
            ans = ans1
    path = homeplace.final_factor_file + ans
    df = pd.read_feather(path)
    df.columns = ["date"] + list(df.columns)[1:]
    df = df.set_index(["date"])
    df = df[sorted(list(df.columns))]
    final_date = df.index.max()
    final_date = datetime.datetime.strftime(final_date, "%Y%m%d")
    if output:
        if new:
            if os.path.exists(ans.split("_")[0]):
                fac_name = (
                    ans.split("_")[0]
                    + "/"
                    + ans.split("_")[0]
                    + "因子"
                    + final_date
                    + "因子值.csv"
                )
            else:
                os.makedirs(ans.split("_")[0])
                fac_name = (
                    ans.split("_")[0]
                    + "/"
                    + ans.split("_")[0]
                    + "因子"
                    + final_date
                    + "因子值.csv"
                )
            df.tail(1).T.to_csv(fac_name)
            logger.success(f"{final_date}的因子值已保存")
        else:
            if os.path.exists(ans.split("_")[0]):
                fac_name = (
                    ans.split("_")[0]
                    + "/"
                    + ans.split("_")[0]
                    + "因子截至"
                    + final_date
                    + "因子值.csv"
                )
            else:
                os.makedirs(ans.split("_")[0])
                fac_name = (
                    ans.split("_")[0]
                    + "/"
                    + ans.split("_")[0]
                    + "因子截至"
                    + final_date
                    + "因子值.csv"
                )
            df.to_csv(fac_name)
            logger.success(f"截至{final_date}的因子值已保存")
        return df, fac_name
    else:
        return df, ""


def database_read_primary_factors(name: str = None) -> pd.DataFrame:
    """根据因子名字，或因子序号，读取初级因子的因子值"""
    homeplace = HomePlace()
    name = name + "_初级.feather"
    df = pd.read_feather(homeplace.factor_data_file + name)
    df = df.rename(columns={list(df.columns)[0]: "date"})
    df = df.set_index("date")
    df = df[sorted(list(df.columns))]
    return df


"""发送邮件的模块"""


class pure_mail(object):
    def __init__(self, host, user, pwd, port=465):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.port = port

    @retry
    def sendemail(self, tolist, subject, body, lastemail_path):
        message = MIMEMultipart()
        message["Form"] = Header(self.user, "utf-8")
        message["To"] = Header(",".join(tolist), "utf-8")
        message["Subject"] = Header(subject, "utf-8")
        message.attach(MIMEText(body, "plain", "utf-8"))

        for path in lastemail_path:
            att1 = MIMEApplication(open(path, "rb").read())
            att1["Content-Type"] = "application/octet-stream"
            att1.add_header(
                "Content-Disposition", "attachment", filename=path.split("/")[-1]
            )
            message.attach(att1)
        try:
            client = smtplib.SMTP_SSL(self.host, self.port)
            login = client.login(self.user, self.pwd)
            if login and login[0] == 235:
                client.sendmail(self.user, tolist, message.as_string())
                logger.success("邮件发送成功")
            else:
                logger.warning("登录失败")
        except Exception as e:
            logger.error(f"发送失败，原因为{e}")
