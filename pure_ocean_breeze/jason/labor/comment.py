__updated__ = "2023-03-16 18:52:29"

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
from typing import Tuple
from pure_ocean_breeze.jason.data.read_data import (
    read_market,
)
from pure_ocean_breeze.jason.state.decorators import do_on_dfs
from pure_ocean_breeze.jason.state.states import STATES


def comment_on_rets_and_nets(
    rets: pd.Series, nets: pd.Series, name: str = "绩效", counts_one_year: int = 50
) -> pd.DataFrame:
    """输入月频的收益率序列和净值序列，输出年化收益、年化波动、信息比率、月度胜率和最大回撤率
    输入2个pd.Series，时间是索引

    Parameters
    ----------
    rets : pd.Series
        收益率序列，index为时间
    nets : pd.Series
        净值序列，index为时间
    name : str, optional
        绩效指标列名字, by default '绩效'
    counts_one_year : int
        一年内有多少次交易, by default 50

    Returns
    -------
    `pd.DataFrame`
        包含年化收益、年化波动、信息比率、月度胜率和最大回撤率的评价指标
    """
    duration_nets = (nets.index[-1] - nets.index[0]).days
    year_nets = duration_nets / 365
    ret_yearly = nets.iloc[-1] /year_nets
    max_draw = ((nets.cummax() - nets) / nets.cummax()).max()
    vol = np.std(rets) * (counts_one_year**0.5)
    info_rate = ret_yearly / vol
    win_rate = len(rets[rets > 0]) / len(rets)
    names = "胜率"
    comments = pd.DataFrame(
        {
            "年化收益率": ret_yearly,
            "年化波动率": vol,
            "信息比率": info_rate,
            names: win_rate,
            "最大回撤率": max_draw,
        },
        index=[name],
    ).T
    return comments


def comments_on_twins(
    nets: pd.Series, rets: pd.Series, counts_one_year: int = 50
) -> pd.Series:
    """输入月频的收益率序列和净值序列，给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率
    输入2个pd.Series，时间是索引

    Parameters
    ----------
    nets : pd.Series
        净值序列，index为时间
    rets : pd.Series
        收益率序列，index为时间
    counts_one_year : int
        一年内有多少次交易, by default 50

    Returns
    -------
    `pd.Series`
        包含年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率的评价指标
    """
    series = nets.copy()
    series1 = rets.copy()
    duration = (series.index[-1] - series.index[0]).days
    year = duration / 365
    ret_yearly = series.iloc[-1] /  year
    max_draw = -((series+1) / (series+1).expanding(1).max()-1).min()
    vol = np.std(series1) * (counts_one_year**0.5)
    sharpe = ret_yearly / vol
    wins = series1[series1 > 0]
    win_rate = len(wins) / len(series1)
    return pd.Series(
        [series.iloc[-1], ret_yearly, vol, sharpe, win_rate, max_draw],
        index=["总收益率", "年化收益率", "年化波动率", "信息比率", "胜率", "最大回撤率"],
    )


def comments_on_twins_periods(
    nets: pd.Series, rets: pd.Series, periods: int
) -> pd.Series:
    """输入其他频率的收益率序列和净值序列，给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率
    输入2个pd.Series，时间是索引

    Parameters
    ----------
    nets : pd.Series
        净值序列，index为时间
    rets : pd.Series
        收益率序列，index为时间
    periods : int
        收益率序列的频率，如5天一次，则为5

    Returns
    -------
    `pd.Series`
        包含年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率的评价指标
    """
    series = nets.copy()
    series1 = rets.copy()
    duration = (series.index[-1] - series.index[0]).days
    year = duration / 365
    ret_yearly = series.iloc[-1] / year
    max_draw = -((series+1) / (series+1).expanding(1).max()-1).min()
    vol = np.std(series1) * (252**0.5) * (periods**0.5)
    sharpe = ret_yearly / vol
    wins = series1[series1 > 0]
    win_rate = len(wins) / len(series1)
    return pd.Series(
        [series.iloc[-1], ret_yearly, vol, sharpe, win_rate, max_draw],
        index=["总收益率", "年化收益率", "年化波动率", "信息比率", "胜率", "最大回撤率"],
    )


@do_on_dfs
def make_relative_comments(
    ret_fac: pd.Series,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
    day: int = STATES['START'],
    show_nets: bool = 0,
) -> pd.Series:
    """对于一个给定的收益率序列，计算其相对于某个指数的超额表现

    Parameters
    ----------
    ret_fac : pd.Series
        给定的收益率序列，index为时间
    hs300 : bool, optional
        为1则相对沪深300指数行情, by default 0
    zz500 : bool, optional
        为1则相对中证500指数行情, by default 0
    zz1000 : bool, optional
        为1则相对中证1000指数行情, by default 0
    day : int, optional
        起始日期，形如20130101, by default STATES['START']
    show_nets : bool, optional
        返回值中包括超额净值数据, by default 0

    Returns
    -------
    `pd.Series`
        评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率

    Raises
    ------
    `IOError`
        如果没指定任何一个指数，将报错
    """

    if hs300:
        net_index = read_market(close=1,hs300=1,every_stock=0,start=day).resample("W").last()
    if zz500:
        net_index = read_market(close=1,zz500=1,every_stock=0,start=day).resample("W").last()
    if zz1000:
        net_index = read_market(close=1,zz1000=1,every_stock=0,start=day).resample("W").last()
    if (hs300 + zz500 + zz1000) == 0:
        raise IOError("你总得指定一个股票池吧？")
    ret_index = net_index.pct_change()
    if day is not None:
        ret_index = ret_index[ret_index.index >= pd.Timestamp(day)]
    ret = ret_fac - ret_index
    ret = ret.dropna()
    net = ret.cumsum()
    rtop = pd.Series(0, index=[net.index.min() - pd.DateOffset(weeks=1)])
    net = pd.concat([rtop, net]).resample("W").last()
    ret = pd.concat([rtop, ret]).resample("W").last()
    com = comments_on_twins(net, ret)
    if show_nets:
        return com, net
    else:
        return com


@do_on_dfs
def make_relative_comments_plot(
    ret_fac: pd.Series,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
    day: int = STATES['START'],
) -> pd.Series:
    """对于一个给定的收益率序列，计算其相对于某个指数的超额表现，然后绘图，并返回超额净值序列

    Parameters
    ----------
    ret_fac : pd.Series
        给定的收益率序列，index为时间
    hs300 : bool, optional
        为1则相对沪深300指数行情, by default 0
    zz500 : bool, optional
        为1则相对中证500指数行情, by default 0
    zz1000 : bool, optional
        为1则相对中证1000指数行情, by default 0
    day : int, optional
        起始日期，形如20130101, by default STATES['START']

    Returns
    -------
    `pd.Series`
        超额净值序列

    Raises
    ------
    `IOError`
        如果没指定任何一个指数，将报错
    """
    if hs300:
        net_index = read_market(close=1,hs300=1,every_stock=0,start=day).resample("W").last()
    if zz500:
        net_index = read_market(close=1,zz500=1,every_stock=0,start=day).resample("W").last()
    if zz1000:
        net_index = read_market(close=1,zz1000=1,every_stock=0,start=day).resample("W").last()
    if (hs300 + zz500 + zz1000) == 0:
        raise IOError("你总得指定一个股票池吧？")
    ret_index = net_index.pct_change()
    if day is not None:
        ret_index = ret_index[ret_index.index >= pd.Timestamp(day)]
    ret = ret_fac - ret_index
    ret = ret.dropna()
    net = ret.cumsum()
    rtop = pd.Series(0, index=[net.index.min() - pd.DateOffset(weeks=1)])
    net = pd.concat([rtop, net]).resample("W").last().ffill()
    net.iplot()
    return net
