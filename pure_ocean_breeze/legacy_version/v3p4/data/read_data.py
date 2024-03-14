__updated__ = "2022-11-05 00:13:07"

import os
import numpy as np
import pandas as pd
import datetime
from typing import Union
from loguru import logger

from pure_ocean_breeze.legacy_version.v3p4.state.states import STATES
from pure_ocean_breeze.legacy_version.v3p4.state.homeplace import HomePlace
from pure_ocean_breeze.legacy_version.v3p4.state.decorators import *
from pure_ocean_breeze.legacy_version.v3p4.data.database import ClickHouseClient, Questdb

homeplace = HomePlace()


def read_daily(
    path: str = None,
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    tr: bool = 0,
    sharenum: bool = 0,
    volume: bool = 0,
    age: bool = 0,
    flow_cap: bool = 0,
    st: bool = 0,
    state: bool = 0,
    unadjust: bool = 0,
    ret: bool = 0,
    ret_inday: bool = 0,
    ret_night: bool = 0,
    vol: bool = 0,
    vol_inday: bool = 0,
    vol_night: bool = 0,
    swing: bool = 0,
    pb: bool = 0,
    pe: bool = 0,
    iret: bool = 0,
    ivol: bool = 0,
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
    age : bool, optional
        为1则选择读取上市天数, by default 0
    flow_cap : bool, optional
        为1则选择读取流通市值, by default 0
    st : bool, optional
        为1则选择读取当日是否为st股，1表示是st股，空值则不是, by default 0
    state : bool, optional
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
    pb : bool, optional
        为1则表示读取市净率, by default 0
    pe : bool, optional
        为1则表示读取市盈率, by default 0
    iret : bool, optional
        为1则表示读取20日回归的fama三因子（市场、流通市值、市净率）特质收益率, by default 0
    ivol : bool, optional
        为1则表示读取20日回归的20日fama三因子（市场、流通市值、市净率）特质波动率, by default 0
    start : int, optional
        起始日期，形如20130101, by default STATES["START"]

    Returns
    -------
    `pd.DataFrame`
        一个columns为股票代码，index为时间，values为目标数据的pd.DataFrame

    Raises
    ------
    `IOError`
        open,close,high,low,tr,sharenum,volume 都为0时，将报错
    另：如果数据未更新，可使用read_daily.clear_cache()来清空缓存
    """

    if not unadjust:
        if path:
            return pd.read_feather(homeplace.daily_data_file + path).set_index("date")
        elif open:
            opens = pd.read_feather(homeplace.daily_data_file + "opens.feather")
            df = opens
            df = df.set_index(list(df.columns)[0])
        elif close:
            closes = pd.read_feather(homeplace.daily_data_file + "closes.feather")
            df = closes
            df = df.set_index(list(df.columns)[0])
        elif high:
            highs = pd.read_feather(homeplace.daily_data_file + "highs.feather")
            df = highs
            df = df.set_index(list(df.columns)[0])
        elif low:
            lows = pd.read_feather(homeplace.daily_data_file + "lows.feather")
            df = lows
            df = df.set_index(list(df.columns)[0])
        elif tr:
            trs = pd.read_feather(homeplace.daily_data_file + "trs.feather")
            df = trs
            df = df.set_index(list(df.columns)[0])
        elif sharenum:
            sharenums = pd.read_feather(homeplace.daily_data_file + "sharenums.feather")
            df = sharenums
            df = df.set_index(list(df.columns)[0])
        elif volume:
            volumes = pd.read_feather(homeplace.daily_data_file + "volumes.feather")
            df = volumes
            df = df.set_index(list(df.columns)[0])
        elif age:
            age = pd.read_feather(homeplace.daily_data_file + "ages.feather")
            df = age
            df = df.set_index(list(df.columns)[0])
        elif flow_cap:
            closes = pd.read_feather(homeplace.daily_data_file + "closes_unadj.feather")
            sharenums = pd.read_feather(homeplace.daily_data_file + "sharenums.feather")
            closes = closes.set_index(list(closes.columns)[0])
            sharenums = sharenums.set_index(list(sharenums.columns)[0])
            flow_cap = closes * sharenums
            df = flow_cap
        elif st:
            st = pd.read_feather(homeplace.daily_data_file + "sts.feather")
            df = st
            df = df.set_index(list(df.columns)[0])
        elif state:
            state = pd.read_feather(homeplace.daily_data_file + "states.feather")
            df = state
            df = df.set_index(list(df.columns)[0])
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
            ) / read_daily(close=1, start=start)
        elif pb:
            df = pd.read_feather(homeplace.daily_data_file + "pb.feather")
            df = df.set_index(list(df.columns)[0])
        elif pe:
            df = pd.read_feather(homeplace.daily_data_file + "pe.feather")
            df = df.set_index(list(df.columns)[0])
        elif iret:
            df = pd.read_feather(
                homeplace.daily_data_file + "idiosyncratic_ret.feather"
            )
            df = df.set_index(list(df.columns)[0])
        elif ivol:
            df = read_daily(iret=1, start=start)
            df = df.rolling(20, min_periods=10).std()
        else:
            raise IOError("阁下总得读点什么吧？🤒")
    else:
        if open:
            opens = pd.read_feather(homeplace.daily_data_file + "opens.feather")
            df = opens
            df = df.set_index(list(df.columns)[0])
        elif close:
            closes = pd.read_feather(homeplace.daily_data_file + "closes.feather")
            df = closes
            df = df.set_index(list(df.columns)[0])
        elif high:
            highs = pd.read_feather(homeplace.daily_data_file + "highs.feather")
            df = highs
            df = df.set_index(list(df.columns)[0])
        elif low:
            lows = pd.read_feather(homeplace.daily_data_file + "lows.feather")
            df = lows
            df = df.set_index(list(df.columns)[0])

        else:
            raise IOError("阁下总得读点什么吧？🤒")
    df = df[df.index >= pd.Timestamp(str(start))]
    return df


def read_market(
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    start: int = STATES["START"],
    every_stock: bool = 1,
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

    Returns
    -------
    Union[pd.DataFrame,pd.Series]
        中证全指每天的指数

    Raises
    ------
    IOError
        如果没有指定任何指数，将报错
    """
    try:
        chc = ClickHouseClient("minute_data")
        df = chc.get_data(
            f"select * from minute_data.minute_data_index where code='000985.SH' and date>={start}00 order by date,num"
        )
        df = df.set_index("code")
        df = df / 100
    except Exception:
        qdb = Questdb()
        df = qdb.get_data(
            "select date,num,close from minute_data_index where code='000985.SH'"
        )
        df.num = df.num.astype(int)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    if open:
        # 米筐的第一分钟是集合竞价，第一分钟的收盘价即为当天开盘价
        df = df[df.num == 1].close
    elif close:
        df = df[df.num == 240].close
    elif high:
        df = df[df.num > 1]
        df = df.groupby("date").max()
        df = df.high
    elif low:
        df = df[df.num > 1]
        df = df.groupby("date").min()
        df = df.low
    else:
        raise IOError("总得指定一个指标吧？🤒")
    if every_stock:
        tr = read_daily(tr=1, start=start)
        df = pd.DataFrame({k: list(df) for k in list(tr.columns)}, index=df.index)
    return df


def read_money_flow(
    buy: bool = 0,
    sell: bool = 0,
    exlarge: bool = 0,
    large: bool = 0,
    median: bool = 0,
    small: bool = 0,
    whole: bool = 0,
) -> pd.DataFrame:
    """一键读入资金流向数据，包括超大单、大单、中单、小单的买入和卖出情况

    Parameters
    ----------
    buy : bool, optional
        方向为买, by default 0
    sell : bool, optional
        方向为卖, by default 0
    exlarge : bool, optional
        超大单，金额大于100万，为机构操作, by default 0
    large : bool, optional
        大单，金额在20万到100万之间，为大户特大单, by default 0
    median : bool, optional
        中单，金额在4万到20万之间，为中户大单, by default 0
    small : bool, optional
        小单，金额在4万以下，为散户中单, by default 0
    whole : bool, optional
        读入当天的总交易额, by default 0

    Returns
    -------
    pd.DataFrame
        index为时间，columns为股票代码，values为对应类型订单当日的成交金额

    Raises
    ------
    IOError
        buy和sell必须指定一个，否则会报错
    IOError
        exlarge，large，median和small必须指定一个，否则会报错
    """
    if not whole:
        if buy:
            if exlarge:
                name = "buy_value_exlarge"
            elif large:
                name = "buy_value_large"
            elif median:
                name = "buy_value_med"
            elif small:
                name = "buy_value_small"
            else:
                raise IOError("您总得指定一种规模吧？🤒")
        elif sell:
            if exlarge:
                name = "sell_value_exlarge"
            elif large:
                name = "sell_value_large"
            elif median:
                name = "sell_value_med"
            elif small:
                name = "sell_value_small"
            else:
                raise IOError("您总得指定一种规模吧？🤒")
        else:
            raise IOError("您总得指定一下是买还是卖吧？🤒")
        name = homeplace.daily_data_file + name + ".feather"
        df = pd.read_feather(name).set_index("date")
        return df
    else:
        dfs = [
            pd.read_feather(homeplace.daily_data_file + name + ".feather").set_index(
                "date"
            )
            for name in [
                "buy_value_exlarge",
                "buy_value_large",
                "buy_value_med",
                "buy_value_small",
                "sell_value_exlarge",
                "sell_value_large",
                "sell_value_med",
                "sell_value_small",
            ]
        ]
        dfs = sum(dfs)
        return dfs


def read_index_single(code: str) -> pd.Series:
    try:
        chc = ClickHouseClient("minute_data")
        hs300 = (
            chc.get_data(
                f"select date,num,close FROM minute_data.minute_data_index WHERE code='{code}'"
            )
            / 100
        )
        hs300.date = pd.to_datetime(hs300.date, format="%Y%m%d")
        hs300 = (
            hs300.sort_values(["date", "num"])
            .groupby("date")
            .last()
            .drop(columns=["num"])
            .close
        )
        return hs300
    except Exception:
        qdb = Questdb()
        hs300 = qdb.get_data(
            f"select date,num,close FROM 'minute_data_index' WHERE code='{code}'"
        )
        hs300.date = pd.to_datetime(hs300.date, format="%Y%m%d")
        hs300.num = hs300.num.astype(int)
        hs300 = (
            hs300.sort_values(["date", "num"])
            .groupby("date")
            .last()
            .drop(columns=["num"])
            .close
        )
        return hs300


def read_index_three(day: int = None) -> tuple[pd.DataFrame]:
    """读取三大指数的原始行情数据，返回并保存在本地

    Parameters
    ----------
    day : int, optional
        起始日期，形如20130101, by default None

    Returns
    -------
    `tuple[pd.DataFrame]`
        分别返回沪深300、中证500、中证1000的行情数据
    """
    if day is None:
        day = STATES["START"]

    hs300, zz500, zz1000, zz2000 = (
        read_index_single("000300.SH").resample("M").last(),
        read_index_single("000905.SH").resample("M").last(),
        read_index_single("000852.SH").resample("M").last(),
        read_index_single("399303.SZ").resample("M").last(),
    )
    hs300 = hs300[hs300.index >= pd.Timestamp(str(day))]
    zz500 = zz500[zz500.index >= pd.Timestamp(str(day))]
    zz1000 = zz1000[zz1000.index >= pd.Timestamp(str(day))]
    zz2000 = zz2000[zz2000.index >= pd.Timestamp(str(day))]

    return hs300, zz500, zz1000, zz2000


def read_swindustry_prices(
    day: int = None, monthly: bool = 1, start: int = STATES["START"]
) -> pd.DataFrame:
    """读取申万一级行业指数的日行情或月行情

    Parameters
    ----------
    day : int, optional
        起始日期，形如20130101, by default None
    monthly : bool, optional
        是否为月行情, by default 1

    Returns
    -------
    `pd.DataFrame`
        申万一级行业的行情数据
    """
    if day is None:
        day = STATES["START"]
    df = pd.read_feather(homeplace.daily_data_file + "申万各行业行情数据.feather").set_index(
        "date"
    )
    df = df[df.index >= pd.Timestamp(str(start))]
    if monthly:
        df = df.resample("M").last()
    return df


def read_zxindustry_prices(
    day: int = None, monthly: bool = 1, start: int = STATES["START"]
) -> pd.DataFrame:
    """读取中信一级行业指数的日行情或月行情

    Parameters
    ----------
    day : int, optional
        起始日期，形如20130101, by default None
    monthly : bool, optional
        是否为月行情, by default 1

    Returns
    -------
    `pd.DataFrame`
        申万一级行业的行情数据
    """
    if day is None:
        day = STATES["START"]
    df = pd.read_feather(homeplace.daily_data_file + "中信各行业行情数据.feather").set_index(
        "date"
    )
    df = df[df.index >= pd.Timestamp(str(start))]
    if monthly:
        df = df.resample("M").last()
    return df


def get_industry_dummies(
    daily: bool = 0,
    monthly: bool = 0,
    start: int = STATES["START"],
    swindustry: bool = 0,
    zxindustry: bool = 0,
) -> dict:
    """生成30/31个行业的哑变量矩阵，返回一个字典

    Parameters
    ----------
    daily : bool, optional
        返回日频的哑变量, by default 0
    monthly : bool, optional
        返回月频的哑变量, by default 0
    start : int, optional
        起始日期, by default STATES["START"]
    swindustry : bool, optional
        是否使用申万一级行业, by default 0
    zxindustry : bool, optional
        是否使用中信一级行业, by default 0

    Returns
    -------
    `dict`
        各个行业及其哑变量构成的字典

    Raises
    ------
    `ValueError`
        如果未指定频率，将报错
    """
    homeplace = HomePlace()
    if swindustry:
        name = "申万行业2021版哑变量.feather"
    else:
        name = "中信一级行业哑变量名称版.feather"
    if monthly:
        industry_dummy = pd.read_feather(homeplace.daily_data_file + name)
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
        industry_dummy = pd.read_feather(homeplace.daily_data_file + name).fillna(0)
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


def database_read_final_factors(
    name: str = None, order: int = None, output: bool = 0, new: bool = 0
) -> tuple[pd.DataFrame, str]:
    """根据因子名字，或因子序号，读取最终因子的因子值

    Parameters
    ----------
    name : str, optional
        因子的名字, by default None
    order : int, optional
        因子的序号, by default None
    output : bool, optional
        是否输出到csv文件, by default 0
    new : bool, optional
        是否只输出最新一期的因子值, by default 0

    Returns
    -------
    `tuple[pd.DataFrame,str]`
        最终因子值和文件路径
    """
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
    name = name + "_初级.feather"
    df = pd.read_feather(homeplace.factor_data_file + name)
    df = df.rename(columns={list(df.columns)[0]: "date"})
    df = df.set_index("date")
    df = df[sorted(list(df.columns))]
    return df
