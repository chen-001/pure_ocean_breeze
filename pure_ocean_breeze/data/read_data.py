__updated__ = "2023-07-26 16:42:17"

import os
import numpy as np
import pandas as pd
import datetime
import deprecation
from typing import Any, Union, Dict, Tuple
from loguru import logger

from pure_ocean_breeze import __version__
from pure_ocean_breeze.state.states import STATES
from pure_ocean_breeze.state.homeplace import HomePlace
from pure_ocean_breeze.state.decorators import *
from pure_ocean_breeze.data.database import ClickHouseClient, Questdb

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")


def read_daily(
    path: str = None,
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
    age: bool = 0,
    flow_cap: bool = 0,
    total_cap: bool = 0,
    adjfactor: bool = 0,
    st: bool = 0,
    state: bool = 0,
    unadjust: bool = 0,
    ret: bool = 0,
    ret_inday: bool = 0,
    ret_night: bool = 0,
    vol_daily: bool = 0,
    vol: bool = 0,
    vol_inday: bool = 0,
    vol_night: bool = 0,
    swing: bool = 0,
    pb: bool = 0,
    pe: bool = 0,
    pettm: bool = 0,
    iret: bool = 0,
    ivol: bool = 0,
    illiquidity: bool = 0,
    swindustry_ret: bool = 0,
    zxindustry_ret: bool = 0,
    stop_up: bool = 0,
    stop_down: bool = 0,
    zxindustry_dummy_code: bool = 0,
    zxindustry_dummy_name: bool = 0,
    swindustry_dummy: bool = 0,
    hs300_member_weight: bool = 0,
    zz500_member_weight: bool = 0,
    zz1000_member_weight: bool = 0,
    start: Union[int, str] = STATES["START"],
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
    age : bool, optional
        为1则选择读取上市天数, by default 0
    flow_cap : bool, optional
        为1则选择读取流通市值, by default 0
    total_cap : bool, optional
        为1则选择读取总市值, by default 0
    adjfactor : bool, optional
        为1则选择读取复权因子, by default 0
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
    vol_daily : bool, optional
        为1则表示读取使用分钟收盘价的标准差计算的波动率, by default 0
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
    pettm : bool, optional
        为1则表示读取市盈率, by default 0
    iret : bool, optional
        为1则表示读取20日回归的fama三因子（市场、流通市值、市净率）特质收益率, by default 0
    ivol : bool, optional
        为1则表示读取20日回归的20日fama三因子（市场、流通市值、市净率）特质波动率, by default 0
    illiquidity : bool, optional
        为1则表示读取当日amihud非流动性指标, by default 0
    swindustry_ret : bool, optional
        为1则表示读取每只股票对应申万一级行业当日收益率, by default 0
    zxindustry_ret : bool, optional
        为1则表示读取每只股票对应申万一级行业当日收益率, by default 0
    stop_up : bool, optional
        为1则表示读取每只股票涨停价, by default 0
    stop_down : bool, optional
        为1则表示读取每只股票跌停价, by default 0
    zxindustry_dummy_code : bool, optional
        为1则表示读取中信一级行业哑变量表代码版, by default 0
    zxindustry_dummy_name : bool, optional
        为1则表示读取中信一级行业哑变量表名称版, by default 0
    swindustry_dummy : bool, optional
        为1则表示读取申万一级行业哑变量, by default 0
    hs300_member_weight : bool, optional
        为1则表示读取沪深300成分股权重（月频）, by default 0
    zz500_member_weight : bool, optional
        为1则表示读取中证500成分股权重（月频）, by default 0
    zz1000_member_weight : bool, optional
        为1则表示读取中证1000成分股权重（月频）, by default 0
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
        if path:
            return pd.read_parquet(homeplace.daily_data_file + path)
        elif open:
            opens = pd.read_parquet(
                homeplace.daily_data_file + "opens.parquet"
            ) * read_daily(state=1, start=start)
            df = opens
        elif close:
            closes = pd.read_parquet(
                homeplace.daily_data_file + "closes.parquet"
            ) * read_daily(state=1, start=start)
            df = closes
        elif high:
            highs = pd.read_parquet(
                homeplace.daily_data_file + "highs.parquet"
            ) * read_daily(state=1, start=start)
            df = highs
        elif low:
            lows = pd.read_parquet(
                homeplace.daily_data_file + "lows.parquet"
            ) * read_daily(state=1, start=start)
            df = lows
        elif vwap:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "vwaps.parquet")
                * read_daily(adjfactor=1, start=start)
                * read_daily(state=1, start=start)
            )
        elif tr:
            trs = pd.read_parquet(homeplace.daily_data_file + "trs.parquet").replace(
                0, np.nan
            ) * read_daily(state=1, start=start)
            df = trs
        elif sharenum:
            sharenums = pd.read_parquet(homeplace.daily_data_file + "sharenums.parquet")
            df = sharenums
        elif total_sharenum:
            df = pd.read_parquet(homeplace.daily_data_file + "total_sharenums.parquet")
        elif amount:
            volumes = pd.read_parquet(
                homeplace.daily_data_file + "amounts.parquet"
            ) * read_daily(state=1, start=start)
            df = volumes
        elif money:
            df = pd.read_parquet(
                homeplace.factor_data_file + "日频数据-每日成交额/每日成交额.parquet"
            )
        elif age:
            age = pd.read_parquet(homeplace.daily_data_file + "ages.parquet")
            df = age
        elif flow_cap:
            closes = pd.read_parquet(
                homeplace.daily_data_file + "closes_unadj.parquet"
            ) * read_daily(state=1, start=start)
            sharenums = pd.read_parquet(homeplace.daily_data_file + "sharenums.parquet")
            flow_cap = closes * sharenums
            df = flow_cap
        elif total_cap:
            closes = pd.read_parquet(
                homeplace.daily_data_file + "closes_unadj.parquet"
            ) * read_daily(state=1, start=start)
            sharenums = pd.read_parquet(
                homeplace.daily_data_file + "total_sharenums.parquet"
            )
            flow_cap = closes * sharenums
            df = flow_cap
        elif adjfactor:
            # df=pd.read_parquet(homeplace.daily_data_file+'adjfactors.parquet')
            df = (
                read_daily(close=1, start=start)
                * read_daily(state=1, start=start)
                / read_daily(close=1, start=start, unadjust=1)
                * read_daily(state=1, start=start)
            )
        elif st:
            st = pd.read_parquet(homeplace.daily_data_file + "sts.parquet")
            df = st
        elif state:
            state = pd.read_parquet(homeplace.daily_data_file + "states.parquet")
            state = state.where(state == 1, np.nan)
            df = state
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
        elif vol_daily:
            df = pd.read_parquet(
                homeplace.factor_data_file + "草木皆兵/草木皆兵_初级.parquet"
            ) * read_daily(state=1)
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
        elif pb:
            df = pd.read_parquet(homeplace.daily_data_file + "pb.parquet") * read_daily(
                state=1, start=start
            )
        elif pe:
            df = pd.read_parquet(homeplace.daily_data_file + "pe.parquet") * read_daily(
                state=1, start=start
            )
        elif pettm:
            df = pd.read_parquet(
                homeplace.daily_data_file + "pettm.parquet"
            ) * read_daily(state=1, start=start)
        elif iret:
            df = pd.read_parquet(
                homeplace.daily_data_file + "idiosyncratic_ret.parquet"
            ) * read_daily(state=1, start=start)
        elif ivol:
            df = read_daily(iret=1, start=start)
            df = df.rolling(20, min_periods=10).std()
        elif illiquidity:
            df = pd.read_parquet(
                homeplace.daily_data_file + "illiquidity.parquet"
            ) * read_daily(state=1, start=start)
        elif swindustry_ret:
            df = pd.read_parquet(
                homeplace.daily_data_file + "股票对应申万一级行业每日收益率.parquet"
            ) * read_daily(state=1, start=start)
        elif zxindustry_ret:
            df = pd.read_parquet(
                homeplace.daily_data_file + "股票对应中信一级行业每日收益率.parquet"
            ) * read_daily(state=1, start=start)
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
        elif zxindustry_dummy_code:
            df = pd.read_parquet(homeplace.daily_data_file + "中信一级行业哑变量代码版.parquet")
        elif zxindustry_dummy_name:
            df = pd.read_parquet(homeplace.daily_data_file + "中信一级行业哑变量名称版.parquet")
        elif swindustry_dummy:
            df = pd.read_parquet(homeplace.daily_data_file + "申万行业2021版哑变量.parquet")
        elif hs300_member_weight:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "沪深300成分股权重.parquet")
                .resample("M")
                .last()
            )
        elif zz500_member_weight:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "中证500成分股权重.parquet")
                .resample("M")
                .last()
            )
        elif zz1000_member_weight:
            df = (
                pd.read_parquet(homeplace.daily_data_file + "中证1000成分股权重.parquet")
                .resample("M")
                .last()
            )
        else:
            raise IOError("阁下总得读点什么吧？🤒")
    else:
        if open:
            opens = pd.read_parquet(
                homeplace.daily_data_file + "opens_unadj.parquet"
            ) * read_daily(state=1, start=start)
            df = opens
        elif close:
            closes = pd.read_parquet(
                homeplace.daily_data_file + "closes_unadj.parquet"
            ) * read_daily(state=1, start=start)
            df = closes
        elif high:
            highs = pd.read_parquet(
                homeplace.daily_data_file + "highs_unadj.parquet"
            ) * read_daily(state=1, start=start)
            df = highs
        elif low:
            lows = pd.read_parquet(
                homeplace.daily_data_file + "lows_unadj.parquet"
            ) * read_daily(state=1, start=start)
            df = lows
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


def read_market(
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    start: int = STATES["START"],
    every_stock: bool = 1,
    market_code: str = "000985.SH",
    questdb_host: str = "127.0.0.1",
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
    market_code : str, optional
        选用哪个指数作为市场指数，默认使用中证全指
    questdb_host: str, optional
        questdb的host，使用NAS时改为'192.168.1.3', by default '127.0.0.1'

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
        df = (
            chc.get_data(
                f"select date,num,close,high,low from minute_data.minute_data_index where code='{market_code}' and date>={start}00 order by date,num"
            )
            / 100
        )
    except Exception:
        try:
            qdb = Questdb(host=questdb_host)
            df = qdb.get_data(
                f"select date,num,close,high,low from minute_data_index where code='{market_code}' and cast(date as int)>={start}"
            )
        except Exception:
            qdb = Questdb(host="192.168.1.3")
            df = qdb.get_data(
                f"select date,num,close,high,low from minute_data_index where code='{market_code}' and cast(date as int)>={start}"
            )
        df.num = df.num.astype(int)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    if open:
        # 米筐的第一分钟是集合竞价，第一分钟的收盘价即为当天开盘价
        df = df[df.num == 1].open
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
        name = homeplace.daily_data_file + name + ".parquet"
        df = pd.read_parquet(name)
        return df
    else:
        dfs = [
            pd.read_parquet(homeplace.daily_data_file + name + ".parquet")
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


def read_index_single(code: str, questdb_host: str = "127.0.0.1") -> pd.Series:
    """读取某个指数的日行情收盘价数据

    Parameters
    ----------
    code : str
        指数的wind代码
    questdb_host: str, optional
        questdb的host，使用NAS时改为'192.168.1.3', by default '127.0.0.1'

    Returns
    -------
    pd.Series
        日行情数据
    """
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
        try:
            qdb = Questdb(host=questdb_host)
            hs300 = qdb.get_data(
                f"select date,num,close FROM 'minute_data_index' WHERE code='{code}'"
            )
        except Exception:
            qdb = Questdb(host="192.168.1.3")
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


def read_index_three(day: int = None) -> Tuple[pd.DataFrame]:
    """读取三大指数的原始行情数据，返回并保存在本地

    Parameters
    ----------
    day : int, optional
        起始日期，形如20130101, by default None

    Returns
    -------
    `Tuple[pd.DataFrame]`
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
    df = pd.read_parquet(homeplace.daily_data_file + "申万各行业行情数据.parquet")
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
    df = pd.read_parquet(homeplace.daily_data_file + "中信各行业行情数据.parquet")
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
) -> Dict:
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
    `Dict`
        各个行业及其哑变量构成的字典

    Raises
    ------
    `ValueError`
        如果未指定频率，将报错
    """
    homeplace = HomePlace()
    if swindustry:
        name = "申万行业2021版哑变量.parquet"
    else:
        name = "中信一级行业哑变量名称版.parquet"
    if monthly:
        industry_dummy = pd.read_parquet(homeplace.daily_data_file + name)
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


@deprecation.deprecated(
    deprecated_in="4.0",
    removed_in="5.0",
    current_version=__version__,
    details="由于因子成果数据库升级，3.x版本的因子成果读取函数将下线",
)
def database_read_final_factors(
    name: str = None,
    order: int = None,
    freq: str = "月",
    output: bool = 0,
    new: bool = 0,
) -> Tuple[pd.DataFrame, str]:
    """根据因子名字，或因子序号，读取最终因子的因子值

    Parameters
    ----------
    name : str, optional
        因子的名字, by default None
    order : int, optional
        因子的序号, by default None
    freq : str, optional
        因子的频率，目前支持`'月'`和`'周'`
    output : bool, optional
        是否输出到csv文件, by default 0
    new : bool, optional
        是否只输出最新一期的因子值, by default 0

    Returns
    -------
    `Tuple[pd.DataFrame,str]`
        最终因子值和文件路径
    """
    homeplace = HomePlace()
    facs = os.listdir(homeplace.final_factor_file)
    if name is None and order is None:
        raise IOError("请指定因子名字或者因子序号")
    elif name is None and order is not None:
        key = "多因子" + str(order) + "_" + freq
        ans = [i for i in facs if ((key in i) and (freq in i))][0]
    elif name is not None and name is None:
        key = name
        ans = [i for i in facs if ((key in i) and (freq in i))]
        if len(ans) > 0:
            ans = ans[0]
        else:
            raise IOError(f"您名字记错了，不存在叫{name}的因子")
    else:
        key1 = name
        key2 = "多因子" + str(order) + "_" + freq
        ans1 = [i for i in facs if ((key1 in i) and (freq in i))]
        if len(ans1) > 0:
            ans1 = ans1[0]
        else:
            raise IOError(f"您名字记错了，不存在叫{name}的因子")
        ans2 = [i for i in facs if ((key2 in i) and (freq in i))][0]
        if ans1 != ans2:
            ans = ans1
            logger.warning("您输入的名字和序号不一致，怀疑您记错了序号，程序默认以名字为准了哈")
        else:
            ans = ans1
    path = homeplace.final_factor_file + ans
    df = pd.read_parquet(path)
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
                    + "_"
                    + freq
                    + "频"
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
                    + "_"
                    + freq
                    + "频"
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
                    + "_"
                    + freq
                    + "频"
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
                    + "_"
                    + freq
                    + "频"
                    + "因子值.csv"
                )
            df.to_csv(fac_name)
            logger.success(f"截至{final_date}的因子值已保存")
        return df, fac_name
    else:
        return df, ""


@deprecation.deprecated(
    deprecated_in="4.0",
    removed_in="5.0",
    current_version=__version__,
    details="由于因子成果数据库升级，3.x版本的因子成果读取函数将下线",
)
def database_read_primary_factors(name: str, name2: str = None) -> pd.DataFrame:
    """根据因子名字，读取初级因子的因子值

    Parameters
    ----------
    name : str, optional
        因子的名字, by default None
    name2 : str, optional
        子因子的名字，当有多个分支因子，分别储存时，使用这个参数来指定具体的子因子, by default None

    Returns
    -------
    `pd.DataFrame`
        初级因子的因子值
    """
    homeplace = HomePlace()
    if name2 is None:
        name = name + "/" + name + "_初级.parquet"
    else:
        name = name + "/" + name + "_初级_" + name2 + ".parquet"
    df = pd.read_parquet(homeplace.factor_data_file + name)
    df = df[sorted(list(df.columns))]
    return df


class FactorDone(object):
    def __init__(
        self,
        order: str,
        name: str = None,
        place: str = None,
        son_name: str = None,
        freq: str = "月",
    ) -> None:
        self.homeplace = HomePlace()
        self.order = order
        self.freq = freq
        self.qdb = Questdb()
        try:
            self.factor_infos = self.qdb.get_data(
                f"select * from factor_infos where order='{self.order}' and freq='{self.freq}'"
            )
        except Exception:
            self.factor_infos = pd.DataFrame()
        self.name = name
        self.place = place
        self.son_name = son_name
        if (self.place is None) or (self.name is None):
            final_line = self.qdb.get_data(
                f"select * from factor_infos where order='{self.order}' and freq='{self.freq}'"
            )
            self.name = final_line.name.iloc[0]
            self.place = final_line.place.iloc[0]
        if son_name is None:
            self.file = f"因子{self.order}_{self.name}_{self.freq}_{self.place}.parquet"
            self.son_factors = {}
            if self.factor_infos.shape[0] > 0:
                for row in self.factor_infos.dropna().itertuples():
                    self.son_factors[row.son_name] = FactorDone(
                        order=row.order,
                        name=row.name,
                        place=row.place,
                        son_name=row.son_name,
                        freq=row.freq,
                    )
        else:
            self.file = f"因子{self.order}_{self.name}_{self.son_name}_{self.freq}_{self.place}.parquet"


    def __call__(self, son_name: str = None) -> Union[pd.DataFrame, dict]:
        if son_name is None:
            return pd.read_parquet(self.homeplace.final_factor_file + self.file)
        else:
            return self.son_factors[son_name]()

    def save_factor(self, factor: pd.DataFrame):
        try:
            son_info = self.qdb.get_data(
                f"select * from factor_infos where order='{self.order}' and son_name='{self.son_name}' and freq='{self.freq}'"
            )
        except Exception:
            logger.warning(f"本次为第一次写入{self.name}_{self.son_name}因子")
            son_info = pd.DataFrame()
        if son_info.shape[0] == 0:
            self.qdb.write_via_df(
                pd.DataFrame(
                    {
                        "order": [self.order],
                        "name": [self.name],
                        "place": [self.place],
                        "son_name": [self.son_name],
                        "freq": [self.freq],
                    }
                ),
                "factor_infos",
            )
        factor.to_parquet(self.homeplace.final_factor_file + self.file)
