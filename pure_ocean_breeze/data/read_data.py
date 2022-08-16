__updated__ = "2022-08-16 15:50:56"

import numpy as np
import pandas as pd
import scipy.io as scio
import datetime
from loguru import logger

from cachier import cachier
import pickledb
import rqdatac

rqdatac.init()
from pure_ocean_breeze.state.state import STATES
from pure_ocean_breeze.state.homeplace import HomePlace
from pure_ocean_breeze.state.decorators import *

homeplace = HomePlace()


@cachier()
def read_daily(
    path: str = None,
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    tr: bool = 0,
    sharenum: bool = 0,
    volume: bool = 0,
    unadjust: bool = 0,
    start: int = STATES["START"],
) -> pd.DataFrame:
    """直接读取常用的量价读取日频数据，默认为复权价格，
    在 open,close,high,low,tr,sharenum,volume 中选择一个参数指定为1

    Parameters
    ----------
    path : str, optional
        要读取文件的路径，由于常用的高开低收换手率等都已经封装，因此此处通常为None, by default None
    open : bool, optional
        为1则选择读取开盘价, by default 0
    close : bool, optional
        为1则选择读取收盘价, by default 0
    high : bool, optional
        为1则选择读取最高价, by default 0
    low : bool, optional
        为1则选择读取最低价, by default 0
    tr : bool, optional
        为1则选择读取换手率, by default 0
    sharenum : bool, optional
        为1则选择读取流通股数, by default 0
    volume : bool, optional
        为1则选择读取成交量, by default 0
    unadjust : bool, optional
        为1则将上述价格改为不复权价格, by default 0
    start : int, optional
        起始日期，形如20130101, by default STATES["START"]

    Returns
    -------
    pd.DataFrame
        一个columns为股票代码，index为时间，values为目标数据的pd.DataFrame

    Raises
    ------
    IOError
        open,close,high,low,tr,sharenum,volume 都为0时，将报错
    另：如果数据未更新，可使用read_daily.clear_cache()来清空缓存
    """

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


def read_index_three(day: int = None) -> tuple[pd.DataFrame]:
    """读取三大指数的原始行情数据，返回并保存在本地

    Parameters
    ----------
    day : int, optional
        起始日期，形如20130101, by default None

    Returns
    -------
    tuple[pd.DataFrame]
        分别返回沪深300、中证500、中证1000的行情数据
    """
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


def read_industry_prices(day: int = None, monthly: bool = 1) -> pd.DataFrame:
    """读取申万一级行业指数的日行情或月行情

    Parameters
    ----------
    day : int, optional
        起始日期，形如20130101, by default None
    monthly : bool, optional
        是否为月行情, by default 1

    Returns
    -------
    pd.DataFrame
        申万一级行业的行情数据
    """
    if day is None:
        day = STATES["START"]
    df = pd.read_feather(homeplace.daily_data_file + "各行业行情数据.feather").set_index(
        "date"
    )
    if monthly:
        df = df.resample("M").last()
    return df


def get_industry_dummies(daily: bool = 0, monthly: bool = 0) -> dict:
    """生成31个行业的哑变量矩阵，返回一个字典

    Parameters
    ----------
    daily : bool, optional
        返回日频的哑变量, by default 0
    monthly : bool, optional
        返回月频的哑变量, by default 0

    Returns
    -------
    dict
        各个行业及其哑变量构成的字典

    Raises
    ------
    ValueError
        如果未指定频率，将报错
    """
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


def database_save_final_factors(df: pd.DataFrame, name: str, order: int) -> None:
    """保存最终因子的因子值

    Parameters
    ----------
    df : pd.DataFrame
        最终因子值
    name : str
        因子的名字，如“适度冒险”
    order : int
        因子的序号
    """
    homeplace = HomePlace()
    path = homeplace.final_factor_file + name + "_" + "多因子" + str(order) + ".feather"
    df.reset_index().to_feather(path)
    final_date = df.index.max()
    final_date = datetime.datetime.strftime(final_date, "%Y%m%d")
    config = pickledb.load(homeplace.update_data_file + "database_config.db", False)
    config.set("data_refresh", "done")
    config.dump()
    logger.success(f"今日计算的因子值保存，最新一天为{final_date}")
