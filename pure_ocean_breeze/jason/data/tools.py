"""
针对一些不常见的文件格式，读取数据文件的一些工具函数，以及其他数据工具
"""

__updated__ = "2025-07-08 16:09:34"

import os
import pandas as pd
import tqdm.auto
import datetime
import numpy as np
import scipy.stats as ss
from functools import reduce, partial,lru_cache
from typing import Callable, Union, Dict, List, Tuple
import joblib
import polars as pl
import polars_ols as pls
import mpire

from pure_ocean_breeze.jason.state.homeplace import HomePlace
from pure_ocean_breeze.jason.state.decorators import do_on_dfs
import rust_pyfunc as rp

homeplace = HomePlace()

@lru_cache(maxsize=1)
def get_xs():
    """使用lru_cache确保每个进程只加载一次"""
    try:
        homeplace = HomePlace()
        return pd.read_parquet(homeplace.barra_data_file+"barra_industry_weekly_together.parquet")
    except Exception:
        print("无法加载xs数据")
        return pd.DataFrame()

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
def lu生成每日分类表(
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


@do_on_dfs
def drop_duplicates_index(new: pd.DataFrame,keep:str='first') -> pd.DataFrame:
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
        subset=["tmp_name_for_this_function_never_same_to_others"], keep=keep
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
def boom_one(
    df: pd.DataFrame, backsee: int = 5, daily: bool = 0, min_periods: int = None
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = int(backsee * 0.5)
    if not daily:
        df_mean = (
            df.rolling(backsee, min_periods=min_periods).mean().resample("W").last()
        )
    else:
        df_mean = df.rolling(backsee, min_periods=min_periods).mean()
    return df_mean

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


def get_values(df: pd.DataFrame,n_jobs:int=40) -> List[pd.DataFrame]:
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
    if n_jobs>1:
        facs=joblib.Parallel(n_jobs=40)(joblib.delayed(get_value)(df, x) for x in tqdm.auto.tqdm(list(range(num))))
    else:
        facs = list(map(lambda x: get_value(df, x), range(num)))

    return facs


@do_on_dfs
def lu计算连续期数(ret0: pd.Series, point: float = 0) -> pd.Series:
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


@do_on_dfs
def jason_to_wind(df: pd.DataFrame):
    if '.' not in df.columns[0]:
        df1 = df.copy()
        df1.index = pd.to_datetime(df1.index.astype(str))
        df1.columns = [add_suffix(i) for i in df1.columns]
        return df1
    else:
        return df


@do_on_dfs
def wind_to_jason(df: pd.DataFrame):
    if '.' in df.columns[0]:
        df1 = df.copy()
        df1.columns = [i[:6] for i in df1.columns]
        df1.index = df1.index.strftime("%Y%m%d").astype(int)
        return df1
    else:
        return df


@do_on_dfs
def lu计算连续期数2(
    s: Union[pd.Series, pd.DataFrame],
    judge_number: float = 1,
    nan_value: float = np.nan,
) -> Union[pd.Series, pd.DataFrame]:
    """
    <<注意！使用此函数时，目标df的值必须全为1或nan！！！>>
    """

    # 将Series中的值转换为布尔值，1为True，其余为False
    is_one = s == judge_number

    # 计算累计和以标识连续的1区块
    cumulative_sum = is_one.cumsum()

    # 重置每个连续1区块的累计和，为此，我们需要找到每个区块的开始并从cumulative_sum中减去该值
    reset_cumsum = cumulative_sum - cumulative_sum.where(~is_one).ffill().fillna(0)

    # 在每个连续的1区块内，使用cumsum计算连续1的个数
    continuous_ones = reset_cumsum * is_one
    return continuous_ones.replace(0, nan_value)


def lu计算连续期数2片段递增(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return lu计算连续期数2(s) + lu标记连续片段(s)


def lu计算连续期数2片段递减(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return lu计算连续期数2(s) - lu标记连续片段(s)


def lu计算连续期数奇正偶反(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(s, pd.DataFrame):
        return s.apply(lu计算连续期数奇正偶反)
    else:
        s = s.reset_index(drop=True)
        # 用于标记每个连续非NaN片段
        s_diff = (~s.isna()).astype(int).diff().fillna(1).cumsum()

        # 对每个连续片段进行编号（1开始）
        segment_nums = s_diff[s_diff.diff().fillna(1) != 0].cumsum()

        # 将连续片段编号映射回原始序列
        s_mapped = segment_nums.reindex_like(s).ffill().fillna(0)

        # 对每个连续非NaN片段生成顺序序列
        order = s.groupby(s_mapped).cumcount() + 1

        # 计算每个连续非NaN片段的长度
        segment_lengths = s.groupby(s_mapped).transform("count")

        # 对偶数编号的片段进行逆序排列
        s[s.notna()] = np.where(
            s_mapped % 2 == 1, order, segment_lengths[s_mapped == s_mapped] - order + 1
        )
        return s.values


def lu计算连续期数偶正奇反(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(s, pd.DataFrame):
        return s.apply(lu计算连续期数偶正奇反)
    else:
        s = s.reset_index(drop=True)
        # 用于标记每个连续非NaN片段
        s_diff = (~s.isna()).astype(int).diff().fillna(1).cumsum()

        # 对每个连续片段进行编号（1开始）
        segment_nums = s_diff[s_diff.diff().fillna(1) != 0].cumsum()

        # 将连续片段编号映射回原始序列
        s_mapped = segment_nums.reindex_like(s).ffill().fillna(0)

        # 对每个连续非NaN片段生成顺序序列
        order = s.groupby(s_mapped).cumcount() + 1

        # 计算每个连续非NaN片段的长度
        segment_lengths = s.groupby(s_mapped).transform("count")

        # 对偶数编号的片段进行逆序排列
        s[s.notna()] = np.where(
            s_mapped % 2 == 1, segment_lengths[s_mapped == s_mapped] - order + 1, order
        )
        return s.values


def lu计算连续期数长度(
    s: Union[pd.Series, pd.DataFrame], final_mean=1
) -> Union[float, pd.Series, pd.DataFrame]:
    if isinstance(s, pd.DataFrame):
        return s.apply(lambda x: lu计算连续期数长度(x, final_mean))
    else:
        # 标识非 NaN 值
        not_nan = s.notnull()

        # 计算连续非 NaN 值的分组
        groups = not_nan.ne(not_nan.shift()).cumsum()[not_nan]

        # 计算每段连续非 NaN 值的长度
        segment_lengths = groups.value_counts().sort_index()

        # 计算平均长度
        average_length = segment_lengths.mean()
        if not final_mean:
            return segment_lengths.values
        else:
            return segment_lengths.mean()


def lu标记连续片段(
    s: Union[pd.Series, pd.DataFrame], label_nan=0, number_continuous=1
) -> Union[pd.Series, pd.DataFrame]:
    not_nan = ~s.isna()
    segment_starts = not_nan.diff().fillna(
        True
    )  # 对序列首个元素填充True，因为diff会产生NaN

    # 为每个连续片段分配一个唯一标识符
    segments = segment_starts.cumsum()

    # 仅对非NaN片段应用标识符，NaN值保持不变
    if not label_nan:
        segments = segments * np.sign(s.abs() + 1)
    if number_continuous:
        segments = (segments + 1) // 2
    return segments


def lu删去连续片段中的最大值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(s, pd.DataFrame):
        return s.apply(lu删去连续片段中的最大值)
    else:
        # 生成连续片段的标识符
        s_diff = s.isna().astype(int).diff().fillna(0).ne(0).cumsum()

        # 对每个片段使用transform找到最大值
        max_vals = s.groupby(s_diff).transform("max")

        # 将原始序列中等于最大值的元素替换为NaN
        s[s == max_vals] = np.nan
        return s


def lu删去连续片段中的最小值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return -lu删去连续片段中的最大值(-s)


def lu仅保留连续片段中的最大值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(s, pd.DataFrame):
        return s.apply(lu删去连续片段中的最大值)
    else:
        # 生成连续片段的标识符
        s_diff = s.isna().astype(int).diff().fillna(0).ne(0).cumsum()

        # 对每个片段使用transform找到最大值
        max_vals = s.groupby(s_diff).transform("max")

        # 将原始序列中等于最大值的元素替换为NaN
        s[s != max_vals] = np.nan
        return s


def lu仅保留连续片段中的最小值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return -lu仅保留连续片段中的最大值(-s)


def lu删去连续片段中的最大值及其后面的值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(s, pd.DataFrame):
        return s.apply(lu删去连续片段中的最大值及其后面的值)
    else:
        # 生成连续片段的标识符
        s_diff = s.isna().astype(int).diff().fillna(0).ne(0).cumsum()

        # 对每个片段使用transform找到最大值
        max_vals = s.groupby(s_diff).transform("max")

        # 使用cummax标记最大值及其之后的值
        max_flag = (s.groupby(s_diff).cummax() == max_vals).astype(int)

        # 使用cumsum在每个片段内生成标记，从最大值开始累加
        max_flag_cum = max_flag.groupby(s_diff).cumsum()

        # 将标记的值替换为NaN
        s[max_flag_cum > 0] = np.nan
        return s


def lu删去连续片段中的最小值及其后面的值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return -lu删去连续片段中的最大值及其后面的值(-s)


def lu删去连续片段中的最大值及其前面的值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return lu删去连续片段中的最大值及其后面的值(s[::-1])[::-1]


def lu删去连续片段中的最小值及其前面的值(
    s: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    return -lu删去连续片段中的最大值及其前面的值(-s)


@do_on_dfs
def is_pos(
    s: Union[pd.Series, pd.DataFrame], zero_as_pos: bool = 1
) -> Union[pd.Series, pd.DataFrame]:
    if zero_as_pos:
        return np.sign(s).replace(0, 1).replace(-1, np.nan)
    else:
        return np.sign(s).replace(0, np.nan).replace(-1, np.nan)


@do_on_dfs
def is_neg(
    s: Union[pd.Series, pd.DataFrame], zero_as_neg: bool = 1
) -> Union[pd.Series, pd.DataFrame]:
    if zero_as_neg:
        return np.sign(s).replace(0, -1).replace(1, np.nan).replace(-1, 1)
    else:
        return np.sign(s).replace(0, np.nan).replace(1, np.nan).replace(-1, 1)


@do_on_dfs
def get_pos_value(
    s: Union[pd.Series, pd.DataFrame],
    judge_sign: Union[float, pd.Series, pd.DataFrame],
    zero_as_pos: bool = 1,
) -> Union[pd.Series, pd.DataFrame]:
    return s * is_pos(s - judge_sign, zero_as_pos)


@do_on_dfs
def get_neg_value(
    s: Union[pd.Series, pd.DataFrame],
    judge_sign: Union[float, pd.Series, pd.DataFrame],
    zero_as_neg: bool = 1,
) -> Union[pd.Series, pd.DataFrame]:
    return s * is_neg(s - judge_sign, zero_as_neg)


@do_on_dfs
def count_pos_neg(s: Union[pd.Series, pd.DataFrame]):
    print("正数个数:", is_pos(s).sum().sum(), "负数个数:", is_neg(s).sum().sum())




def de_cross_polars(
    y: Union[pd.DataFrame, pl.DataFrame],
    xs: Union[list[pd.DataFrame], list[pl.DataFrame]],
) -> pd.DataFrame:
    """因子正交函数，使用polars库实现
    速度：10个barra因子、2016-2022、大约11.5秒

    Parameters
    ----------
    y : Union[pd.DataFrame, pl.DataFrame]
        要研究的因子，形式与h5存数据的形式相同，index是时间，columns是股票
    xs : Union[list[pd.DataFrame], list[pl.DataFrame]]
        要被正交掉的干扰因子们，传入一个列表，每个都是h5存储的那种形式的df，index是时间，columns是股票

    Returns
    -------
    pd.DataFrame
        正交后的残差，形式与y相同，index是时间，columns是股票
    """
    if isinstance(y, pd.DataFrame):
        y.index.name='date'
        y = pl.from_pandas(y.reset_index())
    if isinstance(xs[0], pd.DataFrame):
        for i in range(len(xs)):
            xs[i].index.name='date'
        xs = [pl.from_pandas(x.reset_index()) for x in xs]
    y = y.unpivot(index="date", variable_name="code").drop_nulls()
    xs = [x.unpivot(index="date", variable_name="code").drop_nulls() for x in xs]
    for num, i in enumerate(xs):
        y = y.join(i, on=["date", "code"], suffix=f"_{num}")
    y = (
        y.select(
            "date",
            "code",
            pl.col("value")
            .least_squares.ols(
                *[pl.col(f"value_{i}") for i in range(len(xs))],
                add_intercept=True,
                mode="residuals",
            )
            .over("date")
            .alias("resid"),
        )
        .pivot("code", index="date", values="resid")
        .sort('date')
        .to_pandas()
        .set_index("date")
    )
    return y


def de_cross(y, x_list):
    """
    因子正交化函数。

    参数：
    y：pandas DataFrame，因变量。
    x_list：包含 pandas DataFrame 的列表，自变量。

    返回：
    pandas DataFrame，正交化后的残差。
    """
    index_list = [df.index for df in [y]+x_list]
    dates = reduce(lambda x, y: x.intersection(y), index_list)
    def one(timestamp):
        y_series=y.loc[timestamp]
        xs=pd.concat([x.loc[timestamp].to_frame(str(num)) for num,x in enumerate(x_list)],axis=1)
        yxs=pd.concat([y_series.to_frame('haha'),xs],axis=1).dropna().replace([np.inf,-np.inf],np.nan)
        betas=rp.ols(yxs[xs.columns].to_numpy(dtype=float),yxs['haha'].to_numpy(dtype=float),False)
        yresi=y_series-sum([betas[i+1]*xs[str(i)] for i in range(len(betas)-1)])-betas[0]
        yresi=yresi.to_frame('haha').T
        yresi.index=[timestamp]
        return yresi
        
    with mpire.WorkerPool(n_jobs=10) as pool:
        residual_df=pd.concat(pool.map_unordered(one, dates)).sort_index()
    return residual_df


def de_cross_special_for_barra_daily_jason(
    y: Union[pd.DataFrame, pl.DataFrame],
) -> pd.DataFrame:
    """因子正交函数，但固定了xs为barra数据
    速度：10个barra因子、2016-2022、大约3.2秒

    Parameters
    ----------
    y : Union[pd.DataFrame, pl.DataFrame]
        要研究的因子，形式与h5存数据的形式相同，index是时间，columns是股票

    Returns
    -------
    pd.DataFrame
        正交后的残差，形式与y相同，index是时间，columns是股票
    """    
    if isinstance(y, pd.DataFrame):
        y.index.name='date'
        y = pl.from_pandas(y.reset_index())
    y = y.unpivot(index="date", variable_name="code").drop_nulls()
    xs = pl.read_parquet(
        homeplace.barra_data_file+"barra_daily_together_jason.parquet" # 我这个数据缺2020-08-04 和 2020-08-05，给你的版本可能不缺？不过测速用无伤大雅
    )
    y = y.join(xs, on=["date", "code"])
    cols = y.columns[3:]
    y = (
        y.select(
            "date",
            "code",
            pl.col("value")
            .least_squares.ols(
                *[pl.col(i) for i in cols],
                add_intercept=True,
                mode="residuals",
            )
            .over("date")
            .alias("resid"),
        )
        .pivot("code", index="date", values="resid")
        .to_pandas()
        .set_index("date")
        .sort_index()
    )
    return y



def de_cross_special_for_barra_weekly_fast(
    y: Union[pd.DataFrame, pl.DataFrame],with_corr:int=1
) -> pd.DataFrame:
    """因子正交函数，但固定了xs为barra数据
    速度：10个barra因子、2016-2022、大约3.2秒

    Parameters
    ----------
    y : Union[pd.DataFrame, pl.DataFrame]
        要研究的因子，形式与h5存数据的形式相同，index是时间，columns是股票

    Returns
    -------
    pd.DataFrame
        正交后的残差，形式与y相同，index是时间，columns是股票
    """
    if isinstance(y, pd.DataFrame):
        y.index.name='date'
    
    y=y.stack().reset_index()
    y.columns=['date','code','fac']
    
    xs=get_xs()
    
    yx=pd.merge(xs,y,on=['date','code'],sort=False)
    
    def ols_sing(df: pd.DataFrame) -> pd.DataFrame:
        betas=rp.ols(df[df.columns[2:-1]].to_numpy(dtype=float),df['fac'].to_numpy(dtype=float),False)
        df.fac=df.fac-betas[0]-betas[1]*df[df.columns[2]]-betas[2]*df[df.columns[3]]-betas[3]*df[df.columns[4]]-betas[4]*df[df.columns[5]]-betas[5]*df[df.columns[6]]-betas[6]*df[df.columns[7]]-betas[7]*df[df.columns[8]]-betas[8]*df[df.columns[9]]-betas[9]*df[df.columns[10]]-betas[10]*df[df.columns[11]]-betas[11]*df[df.columns[12]]
        return df[['date','code','fac']]
    
    yresid=yx.dropna().groupby('date').apply(ols_sing).pivot(index='date',columns='code',values='fac')
    
    return yresid

def de_cross_special_for_barra_weekly(
    y: Union[pd.DataFrame, pl.DataFrame],with_corr:int=1
) -> pd.DataFrame:
    """因子正交函数，但固定了xs为barra数据
    速度：10个barra因子、2016-2022、大约3.2秒

    Parameters
    ----------
    y : Union[pd.DataFrame, pl.DataFrame]
        要研究的因子，形式与h5存数据的形式相同，index是时间，columns是股票

    Returns
    -------
    pd.DataFrame
        正交后的残差，形式与y相同，index是时间，columns是股票
    """    
    if isinstance(y, pd.DataFrame):
        y.index.name='date'
        y = pl.from_pandas(y.reset_index())
    y = y.unpivot(index="date", variable_name="code").drop_nulls()
    xs = pl.read_parquet(
        homeplace.barra_data_file+"barra_industry_weekly_together.parquet" # 我这个数据缺2020-08-04 和 2020-08-05，给你的版本可能不缺？不过测速用无伤大雅
    )
    y = y.join(xs, on=["date", "code"])
    cols = y.columns[3:]
    yresid = (
        y.select(
            "date",
            "code",
            pl.col("value")
            .least_squares.ols(
                *[pl.col(i) for i in cols],
                add_intercept=True,
                mode="residuals",
            )
            .over("date")
            .alias("resid"),
        )
        .pivot("code", index="date", values="resid")
        .to_pandas()
        .set_index("date")
        .sort_index()
    )
    if with_corr:
        colss=y.columns[3:16]
        corr=y[y.columns[:16]].select(*[pl.corr('value',i).over('date').mean().alias(i) for i in colss]).to_pandas()
        corr.index=['相关系数']
        corr=corr.applymap(to_percent)
        return yresid,corr
    else:
        return yresid
    
def de_cross_special_for_barra_weekly1(
    y: Union[pd.DataFrame, pl.DataFrame],with_corr:int=1
) -> pd.DataFrame:
    """因子正交函数，但固定了xs为barra数据
    速度：10个barra因子、2016-2022、大约3.2秒

    Parameters
    ----------
    y : Union[pd.DataFrame, pl.DataFrame]
        要研究的因子，形式与h5存数据的形式相同，index是时间，columns是股票

    Returns
    -------
    pd.DataFrame
        正交后的残差，形式与y相同，index是时间，columns是股票
    """    
    if isinstance(y, pd.DataFrame):
        y.index.name='date'
        y = pl.from_pandas(y.reset_index())
    y = y.unpivot(index="date", variable_name="code").drop_nulls()
    xs = pl.read_parquet(
        homeplace.barra_data_file+"barra_industry_weekly_together1.parquet" # 我这个数据缺2020-08-04 和 2020-08-05，给你的版本可能不缺？不过测速用无伤大雅
    )
    y = y.join(xs, on=["date", "code"])
    cols = y.columns[3:]
    yresid = (
        y.select(
            "date",
            "code",
            pl.col("value")
            .least_squares.ols(
                *[pl.col(i) for i in cols],
                add_intercept=True,
                mode="residuals",
            )
            .over("date")
            .alias("resid"),
        )
        .pivot("code", index="date", values="resid")
        .to_pandas()
        .set_index("date")
        .sort_index()
    )
    if with_corr:
        colss=y.columns[3:14]
        corr=y[y.columns[:14]].select(*[pl.corr('value',i).over('date').mean().alias(i) for i in colss]).to_pandas()
        corr.index=['相关系数']
        corr=corr.applymap(to_percent)
        return yresid,corr
    else:
        return yresid
    
def adjust_afternoon(df: pd.DataFrame,only_inday:int=1) -> pd.DataFrame:
    start='09:30:00' if only_inday else '09:00:00'
    end='14:57:00' if only_inday else '15:00:00'
    if df.index.name=='exchtime':
        df1=df.between_time(start,'11:30:00')
        df2=df.between_time('13:00:00',end)
        df2.index=df2.index-pd.Timedelta(minutes=90)
        df=pd.concat([df1,df2])
    elif 'exchtime' in df.columns:
        df1=df.set_index('exchtime').between_time(start,'11:30:00')
        df2=df.set_index('exchtime').between_time('13:00:00',end)
        df2.index=df2.index-pd.Timedelta(minutes=90)
        df=pd.concat([df1,df2]).reset_index()
    return df

def get_features_factors(
    df: pd.DataFrame,
    with_abs=False,
    with_max_min=False,
    with_corr=True,
    with_percentiles=True,
    with_lag_autocorr=1,
    with_threshold_counts=True,
    with_period_compare=True,
    with_lyapunov_exponent=True,
    with_complexity=True,  # NEW: Optional complexity metrics (slower but informative)
    append_for_corr:pd.DataFrame=None
):
    """
    Extracts a comprehensive set of statistical features and their names from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numerical data for feature extraction.
        with_abs (bool, optional): If True, includes the absolute values of the computed statistics. Default is True.
        with_max_min (bool, optional): If True, includes the maximum and minimum values (and their absolute values if with_abs is True). Default is False.
        with_corr (bool, optional): If True, includes pairwise correlations between columns. Default is True.
        with_percentiles (bool, optional): If True, includes percentile statistics (P5, P25, P75, P95). Default is True.
        with_lag_autocorr (int, optional): Number of autocorrelation lags to compute (1-5). Default is 1.
        with_threshold_counts (bool, optional): If True, counts threshold crossings (>P90, >P95, <P10, <P5). Default is True.
        with_period_compare (bool, optional): If True, compares first vs last period statistics. Default is True.
        with_complexity (bool, optional): If True, includes complexity metrics (Lyapunov exponent, LZ complexity, entropy, max range product). Computationally expensive. Default is False.

    Returns:
        res (list): List of computed feature values in the order specified by 'names'.
        names (list): List of feature names corresponding to the computed values.

    Features extracted include:
        Level 1 - Basic Statistics:
            - Mean, median, standard deviation, skewness, kurtosis
            - Maximum, minimum, range

        Level 2 - Distribution Features:
            - Percentiles (P5, P25, P75, P95)
            - Interquartile range (IQR = P75 - P25)
            - Coefficient of variation (CV = std/mean)

        Level 3 - Temporal Dynamics:
            - Autocorrelations (lag 1 to N)
            - Linear trend (OLS regression slope)
            - First vs last period comparison (e.g., last 1/3 vs first 1/3 mean)

        Level 4 - Threshold Analysis:
            - Threshold crossing counts (>P90, >P95, <P10, <P5)
            - Excess mean above P90
            - Shortfall mean below P10

        Level 5 - Correlations (if with_corr=True and df has multiple columns):
            - Pairwise Pearson correlations
            - Absolute correlations

        Level 6 - Complexity Metrics (if with_complexity=True):
            - Lyapunov exponent (chaos indicator)
            - LZ complexity (sequence compressibility)
            - Shannon entropy (information content)
            - Max range product (normalized)
    """
    # Level 1: Basic statistics
    means = df.mean()
    medians = df.median()
    stds = df.std()
    skews = df.skew()
    kurts = df.kurt()

    # Level 1 with_abs variants (define upfront to avoid reference issues)
    if with_abs:
        means_abs = means.abs()
        medians_abs = medians.abs()
        stds_abs = stds.abs()
        skews_abs = skews.abs()
        kurts_abs = kurts.abs()

    if with_max_min:
        maxs = df.max()
        mins = df.min()
        ranges = maxs - mins  # Range
        if with_abs:
            maxs_abs = maxs.abs()
            mins_abs = mins.abs()
            ranges_abs = ranges.abs()

    # Level 2: Distribution features
    if with_percentiles:
        p5s = df.quantile(0.05)
        p25s = df.quantile(0.25)
        p75s = df.quantile(0.75)
        p95s = df.quantile(0.95)
        iqrs = p75s - p25s  # Interquartile range
        cvs = stds / (means.abs() + 1e-8)  # Coefficient of variation
        if with_abs:
            p5s_abs = p5s.abs()
            p25s_abs = p25s.abs()
            p75s_abs = p75s.abs()
            p95s_abs = p95s.abs()
            iqrs_abs = iqrs.abs()
            cvs_abs = cvs.abs()

    # Level 3: Temporal dynamics
    # Autocorrelations (lag 1 to N, limited to 3 for speed)
    n_lags = min(max(1, with_lag_autocorr), 3)
    autocorrs = []
    autocorrs_abs = []
    for lag in range(1, n_lags + 1):
        ac = rp.corrwith(df.reset_index(drop=True), df.reset_index(drop=True).shift(lag), 0, use_single_thread=True)
        autocorrs.append(ac)
        autocorrs_abs.append(ac.abs())

    # Linear trend (OLS regression slope)
    trends = rp.trend_2d(df.to_numpy(float), 0)
    if with_abs:
        trends_abs = [abs(i) for i in trends]

    # Period comparison (first 1/3 vs last 1/3)
    if with_period_compare:
        n_rows = len(df)
        split_point = n_rows // 3
        first_period_means = df.iloc[:split_point].mean()
        last_period_means = df.iloc[-split_point:].mean()
        period_diffs = last_period_means - first_period_means
        period_ratios = last_period_means / (first_period_means.abs() + 1e-8)
        if with_abs:
            period_diffs_abs = period_diffs.abs()
            period_ratios_abs = period_ratios.abs()

    # Level 4: Threshold analysis
    if with_threshold_counts:
        p90s = df.quantile(0.90)
        p10s = df.quantile(0.10)

        # Mean of values above/below threshold
        mean_above_p90 = df[df > p90s].mean().fillna(0)
        mean_below_p10 = df[df < p10s].mean().fillna(0)

    # Assemble results and names
    res = []
    names = []
    col_names = df.columns.tolist()

    # Helper function to append results and names
    def append_results(series_list, name_suffixes):
        for series, suffix in zip(series_list, name_suffixes):
            res.extend(series.tolist())
            names.extend([f"{col}_{suffix}" for col in col_names])

    # Level 1: Basic statistics
    append_results([means, medians, stds, skews, kurts],
                   ['mean', 'median', 'std', 'skew', 'kurt'])

    if with_max_min:
        append_results([maxs, mins, ranges],
                       ['max', 'min', 'range'])

    # Level 2: Percentiles
    if with_percentiles:
        append_results([p5s, p25s, p75s, p95s, iqrs, cvs],
                       ['p5', 'p25', 'p75', 'p95', 'iqr', 'cv'])

    # Level 3: Autocorrelations
    for lag_idx, (ac, ac_abs) in enumerate(zip(autocorrs, autocorrs_abs)):
        lag_num = lag_idx + 1
        append_results([ac, ac_abs],
                       [f'autocorr{lag_num}', f'autocorr{lag_num}_abs'])

    # Level 3: Trend
    res.extend(trends)
    names.extend([f"{col}_trend" for col in col_names])

    # Level 3: Period comparison
    if with_period_compare:
        append_results([period_diffs, period_ratios],
                       ['period_diff', 'period_ratio'])

    # Level 4: Threshold means
    if with_threshold_counts:
        append_results([mean_above_p90, mean_below_p10],
                       ['mean_above_p90', 'mean_below_p10'])

    # Level 5: Correlations
    if with_corr:
        if append_for_corr is not None:
            df0=pd.concat([df,append_for_corr],axis=1)
        else:
            df0=df.copy()
        corrs_matrix = rp.fast_correlation_matrix_v2_df(df0, max_workers=1)
        n = corrs_matrix.shape[0]
        i_idx, j_idx = np.triu_indices(n, 1)
        row_names = corrs_matrix.index[i_idx]
        col_names_corr = corrs_matrix.columns[j_idx]
        corr_values = corrs_matrix.to_numpy()[i_idx, j_idx]

        corr_names = [f"{row}_corr_{col}" for row, col in zip(row_names, col_names_corr)]
        res.extend(corr_values.tolist())
        names.extend(corr_names)

        if with_abs:
            res.extend(np.abs(corr_values).tolist())
            names.extend([f"{name}_abs" for name in corr_names])

    # Absolute values (for all base statistics)
    if with_abs:
        append_results([means_abs, medians_abs, stds_abs, skews_abs, kurts_abs],
                       ['mean_abs', 'median_abs', 'std_abs', 'skew_abs', 'kurt_abs'])

        if with_max_min:
            append_results([maxs_abs, mins_abs],
                           ['max_abs', 'min_abs'])

        if with_percentiles:
            append_results([p5s_abs, p25s_abs, p75s_abs, p95s_abs, iqrs_abs, cvs_abs],
                           ['p5_abs', 'p25_abs', 'p75_abs', 'p95_abs', 'iqr_abs', 'cv_abs'])

        res.extend(trends_abs)
        names.extend([f"{col}_trend_abs" for col in col_names])

        if with_period_compare:
            append_results([period_diffs_abs, period_ratios_abs],
                           ['period_diff_abs', 'period_ratio_abs'])

    # Level 6: Complexity metrics (computationally expensive, optional)
    if with_lyapunov_exponent:
        # Define complexity calculation functions
        def calc_lyapunov(series: pd.Series):
            try:
                return rp.calculate_lyapunov_exponent(series.to_numpy(float))['lyapunov_exponent']
            except:
                return np.nan
            
        lyapunovs = df.apply(calc_lyapunov)
        append_results([lyapunovs],['lyapunov'])
    
    if with_complexity:

        def calc_lz_complexity(series: pd.Series):
            try:
                return rp.lz_complexity(series.to_numpy(float))
            except:
                return np.nan

        def calc_entropy(series: pd.Series):
            try:
                return rp.calculate_entropy_1d(series.to_numpy(float))
            except:
                return np.nan

        def calc_max_range_product(series: pd.Series):
            try:
                x, y, _ = rp.find_max_range_product(series.to_numpy(float))
                return abs(x - y) / series.shape[0]
            except:
                return np.nan

        # Calculate complexity metrics for each column
        
        lz_complexities = df.apply(calc_lz_complexity)
        entropies = df.apply(calc_entropy)
        max_range_products = df.apply(calc_max_range_product)

        append_results([lz_complexities, entropies, max_range_products],
                        ['lz_complexity', 'entropy_1d', 'max_range_product'])

    return res, names


def common_columns_index(df1:pd.DataFrame, df2:pd.DataFrame):
    common_cols = df1.columns.intersection(df2.columns)
    common_index = df1.index.intersection(df2.index)
    return df1.loc[common_index, common_cols], df2.loc[common_index, common_cols]

def common_columns_indexs(dfs:list[pd.DataFrame]):
    common_cols = dfs[0].columns
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
        common_index = common_index.intersection(df.index)
    return [df.loc[common_index, common_cols] for df in dfs]