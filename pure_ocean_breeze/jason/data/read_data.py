__updated__ = "2023-07-26 16:42:17"

import os
import numpy as np
import pandas as pd
import datetime
from typing import Any, Union, Dict, Tuple

from pure_ocean_breeze.jason.state.states import STATES
from pure_ocean_breeze.jason.state.homeplace import HomePlace
from pure_ocean_breeze.jason.state.decorators import *

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")


def read_daily(
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    vwap: bool = 0,
    tr: bool = 0,
    sharenum: bool = 0,
    total_sharenum: bool = 0,
    amount: bool = 0,
    money: bool = 0,
    flow_cap: bool = 0,
    total_cap: bool = 0,
    adjfactor: bool = 0,
    state: bool = 0,
    state_loose: bool = 0,
    unadjust: bool = 0,
    ret: bool = 0,
    ret_inday: bool = 0,
    ret_night: bool = 0,
    vol: bool = 0,
    vol_inday: bool = 0,
    vol_night: bool = 0,
    swing: bool = 0,
    stop_up: bool = 0,
    stop_down: bool = 0,
    up_down_limit_status:bool=0,
    swindustry_dummy: bool = 0,
    start: Union[int, str] = STATES["START"],
) -> pd.DataFrame:
    """直接读取常用的量价读取日频数据，默认为复权价格，
    在 open,close,high,low,tr,sharenum,volume 中选择一个参数指定为1

    Parameters
    ----------
    open : bool, optional
        为1则选择读取开盘价, by default 0
    close : bool, optional
        为1则选择读取收盘价, by default 0
    high : bool, optional
        为1则选择读取最高价, by default 0
    low : bool, optional
        为1则选择读取最低价, by default 0
    vwap : bool, optional
        为1则选择读取日均成交价, by default 0
    tr : bool, optional
        为1则选择读取换手率, by default 0
    sharenum : bool, optional
        为1则选择读取流通股数, by default 0
    total_sharenum : bool, optional
        为1则表示读取总股数, by default 0
    amount : bool, optional
        为1则选择读取成交量, by default 0
    money : bool, optional
        为1则表示读取成交额, by default 0
    flow_cap : bool, optional
        为1则选择读取流通市值, by default 0
    total_cap : bool, optional
        为1则选择读取总市值, by default 0
    adjfactor : bool, optional
        为1则选择读取复权因子, by default 0
    state : bool, optional
        为1则选择读取当日交易状态是否正常，1表示正常交易，空值则不是, by default 0
    state_loose : bool, optional
        为1则选择读取当日交易状态是否正常，1表示正常交易，空值则不是, by default 0
    unadjust : bool, optional
        为1则将上述价格改为不复权价格, by default 0
    ret : bool, optional
        为1则选择读取日间收益率, by default 0
    ret_inday : bool, optional
        为1则表示读取日内收益率, by default 0
    ret_night : bool, optional
        为1则表示读取隔夜波动率, by default 0
    vol : bool, optional
        为1则选择读取滚动20日日间波动率, by default 0
    vol_inday : bool, optional
        为1则表示读取滚动20日日内收益率波动率, by default 0
    vol_night : bool, optional
        为1则表示读取滚动20日隔夜收益率波动率, by default 0
    swing : bool, optional
        为1则表示读取振幅, by default 0
    stop_up : bool, optional
        为1则表示读取每只股票涨停价, by default 0
    stop_down : bool, optional
        为1则表示读取每只股票跌停价, by default 0
    swindustry_dummy : bool, optional
        为1则表示读取申万一级行业哑变量, by default 0
    start : Union[int,str], optional
        起始日期，形如20130101, by default STATES["START"]

    Returns
    -------
    `pd.DataFrame`
        一个columns为股票代码，index为时间，values为目标数据的pd.DataFrame

    Raises
    ------
    `IOError`
        open,close,high,low,tr,sharenum,volume 都为0时，将报错
    """

    if not unadjust:
        if open:
            df = pd.read_parquet(
                homeplace.daily_data_file + "opens.parquet"
            ) * read_daily(state=1, start=start)
        elif close:
            df = pd.read_parquet(
                homeplace.daily_data_file + "closes.parquet"
            ) * read_daily(state=1, start=start)
        elif high:
            df = pd.read_parquet(
                homeplace.daily_data_file + "highs.parquet"
            ) * read_daily(state=1, start=start)
        elif low:
            df = pd.read_parquet(
                homeplace.daily_data_file + "lows.parquet"
            ) * read_daily(state=1, start=start)
        elif vwap:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "vwaps.parquet")
                * read_daily(adjfactor=1, start=start)
                *read_daily(state=1, start=start)
            )
        elif tr:
            df = pd.read_parquet(homeplace.daily_data_file + "trs.parquet").replace(
                0, np.nan
            ) *read_daily(state=1, start=start)
        elif sharenum:
            df = pd.read_parquet(homeplace.daily_data_file + "sharenums.parquet")
        elif total_sharenum:
            df = pd.read_parquet(homeplace.daily_data_file + "total_sharenums.parquet")
        elif amount:
            df = pd.read_parquet(
                homeplace.daily_data_file + "amounts.parquet"
            ) * read_daily(state=1, start=start)
        elif money:
            df = pd.read_parquet(
                homeplace.daily_data_file + "moneys.parquet"
            )*read_daily(state=1, start=start)
        elif flow_cap:
            df = pd.read_parquet(homeplace.daily_data_file + "flow_caps.parquet")
        elif total_cap:
            df = pd.read_parquet(homeplace.daily_data_file + "total_caps.parquet")
        elif adjfactor:
            df=pd.read_parquet(homeplace.daily_data_file+'adjfactors.parquet')
        elif state:
            df = pd.read_parquet(homeplace.daily_data_file + "states.parquet")
        elif state_loose:
            df = pd.read_parquet(homeplace.daily_data_file + "states_loose.parquet")
        elif ret:
            df = read_daily(close=1, start=start)
            df = df / df.shift(1) - 1
        elif ret_inday:
            df = read_daily(close=1, start=start) / read_daily(open=1, start=start) - 1
        elif ret_night:
            df = (
                read_daily(open=1, start=start)
                / read_daily(close=1, start=start).shift(1)
                - 1
            )
        elif vol:
            df = read_daily(ret=1, start=start)
            df = df.rolling(20, min_periods=10).std()
        elif vol_inday:
            df = read_daily(ret_inday=1, start=start)
            df = df.rolling(20, min_periods=10).std()
        elif vol_night:
            df = read_daily(ret_night=1, start=start)
            df = df.rolling(20, min_periods=10).std()
        elif swing:
            df = (
                read_daily(high=1, start=start) - read_daily(low=1, start=start)
            ) / read_daily(close=1, start=start).shift(1)
        elif stop_up:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "stop_ups.parquet")
                * read_daily(adjfactor=1, start=start)
                * read_daily(state=1, start=start)
            )
        elif stop_down:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "stop_downs.parquet")
                * read_daily(adjfactor=1, start=start)
                * read_daily(state=1, start=start)
            )
        elif up_down_limit_status:
            df=pd.read_parquet(homeplace.daily_data_file+'up_down_limit_status.parquet')
        elif swindustry_dummy:
            df=pd.read_parquet(homeplace.daily_data_file+'sw_industry_level1_dummies.parquet')
        else:
            raise IOError("阁下总得读点什么吧？🤒")
    else:
        if open:
            df = pd.read_parquet(
                homeplace.daily_data_file + "opens_unadj.parquet"
            ) * read_daily(state=1, start=start)
        elif close:
            df = pd.read_parquet(
                homeplace.daily_data_file + "closes_unadj.parquet"
            ) * read_daily(state=1, start=start)
        elif high:
            df = pd.read_parquet(
                homeplace.daily_data_file + "highs_unadj.parquet"
            ) * read_daily(state=1, start=start)
        elif low:
            df = pd.read_parquet(
                homeplace.daily_data_file + "lows_unadj.parquet"
            ) * read_daily(state=1, start=start)
        elif vwap:
            df = pd.read_parquet(
                homeplace.daily_data_file + "vwaps.parquet"
            ) * read_daily(state=1, start=start)
        elif stop_up:
            df = pd.read_parquet(
                homeplace.daily_data_file + "stop_ups.parquet"
            ) * read_daily(state=1, start=start)
        elif stop_down:
            df = pd.read_parquet(
                homeplace.daily_data_file + "stop_downs.parquet"
            ) * read_daily(state=1, start=start)
        else:
            raise IOError("阁下总得读点什么吧？🤒")
    if "date" not in df.columns:
        df = df[df.index >= pd.Timestamp(str(start))]
    return df.dropna(how="all")


def get_industry_dummies(
    daily: bool = 0,
    weekly: bool = 0,
    start: int = STATES["START"],
) -> Dict:
    """生成30/31个行业的哑变量矩阵，返回一个字典

    Parameters
    ----------
    daily : bool, optional
        返回日频的哑变量, by default 0
    weekly : bool, optional
        返回week频的哑变量, by default 0
    start : int, optional
        起始日期, by default STATES["START"]

    Returns
    -------
    `Dict`
        各个行业及其哑变量构成的字典

    Raises
    ------
    `ValueError`
        如果未指定频率，将报错
    """
    homeplace = HomePlace()
    name='sw_industry_level1_dummies.parquet'
    if weekly:
        industry_dummy = pd.read_parquet(homeplace.daily_data_file + name)
        industry_dummy = (
            industry_dummy.set_index("date")
            .groupby("code")
            .resample("W")
            .last()
            .fillna(0)
            .drop(columns=["code"])
            .reset_index()
        )
    elif daily:
        industry_dummy = pd.read_parquet(homeplace.daily_data_file + name).fillna(0)
    else:
        raise ValueError("您总得指定一个频率吧？🤒")
    industry_dummy = industry_dummy[industry_dummy.date >= pd.Timestamp(str(start))]
    ws = list(industry_dummy.columns)[2:]
    ress = {}
    for w in ws:
        df = industry_dummy[["date", "code", w]]
        df = df.pivot(index="date", columns="code", values=w)
        df = df.replace(0, np.nan)
        ress[w] = df
    return ress


def database_read_final_factors(name: str = None) -> pd.DataFrame:
    """根据因子名字，或因子序号，读取最终因子的因子值

    Parameters
    ----------
    name : str, optional
        因子的名字, by default None

    Returns
    -------
    `pd.DataFrame`
        最终因子值和文件路径
    """
    homeplace = HomePlace()
    df=pd.read_parquet(homeplace.final_factor_file+name+'.parquet')
    return df


def database_read_primary_factors(name: str) -> pd.DataFrame:
    """根据因子名字，读取初级因子的因子值

    Parameters
    ----------
    name : str, optional
        因子的名字, by default None

    Returns
    -------
    `pd.DataFrame`
        初级因子的因子值
    """
    homeplace = HomePlace()
    df = pd.read_parquet(homeplace.factor_data_file + name+'.parquet')
    return df

def read_market(
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    start: int = STATES["START"],
    every_stock: bool = 1,
    sh50: bool = 0,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
) -> Union[pd.DataFrame, pd.Series]:
    """读取中证全指日行情数据

    Parameters
    ----------
    open : bool, optional
        读取开盘点数, by default 0
    close : bool, optional
        读取收盘点数, by default 0
    high : bool, optional
        读取最高点数, by default 0
    low : bool, optional
        读取最低点数, by default 0
    start : int, optional
        读取的起始日期, by default STATES["START"]
    every_stock : bool, optional
        是否修改为index是时间，columns是每只股票代码，每一列值都相同的形式, by default 1
    sh50 : bool, optional
        是否读取上证50, by default 0
    hs300 : bool, optional
        是否读取沪深300, by default 0
    zz500 : bool, optional
        是否读取中证500, by default 0
    zz1000 : bool, optional
        是否读取中证1000, by default 0

    Returns
    -------
    Union[pd.DataFrame,pd.Series]
        读取market_index的行情数据        

    Raises
    ------
    IOError
        如果没有指定任何指数，将报错
    """
    homeplace=HomePlace()
    if open:
        # 米筐的第一分钟是集合竞价，第一分钟的收盘价即为当天开盘价
        df = pd.read_parquet(homeplace.daily_data_file + "index_opens.parquet")
    elif close:
        df = pd.read_parquet(homeplace.daily_data_file + "index_closes.parquet")
    elif high:
        df = pd.read_parquet(homeplace.daily_data_file + "index_highs.parquet")
    elif low:
        df = pd.read_parquet(homeplace.daily_data_file + "index_lows.parquet")
    else:
        raise IOError("总得指定一个指标吧？🤒")
    if sh50:
        df = df["000016.SH"]
    elif hs300:
        df = df["000300.SH"]
    elif zz500:
        df = df["000905.SH"]
    elif zz1000:
        df = df["000852.SH"]
    else:
        raise IOError("总得指定一个指数吧？🤒")
    if every_stock:
        tr = read_daily(tr=1, start=start)
        df = pd.DataFrame({k: list(df) for k in list(tr.columns)}, index=df.index)
    return df
