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
import scipy.stats as ss
from functools import reduce, partial
from typing import Callable, Union, Dict, List, Tuple
import joblib
import mpire
import statsmodels.formula.api as smf

from pure_ocean_breeze.jason.state.homeplace import HomePlace
from pure_ocean_breeze.jason.state.decorators import do_on_dfs

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
    df1 = df.copy()
    df1.index = pd.to_datetime(df1.index.astype(str))
    df1.columns = [add_suffix(i) for i in df1.columns]
    return df1


@do_on_dfs
def wind_to_jason(df: pd.DataFrame):
    df1 = df.copy()
    df1.columns = [i[:6] for i in df1.columns]
    df1.index = df1.index.strftime("%Y%m%d").astype(int)
    return df1


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



def de_cross(
    y: pd.DataFrame, xs: Union[List[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """使用若干因子对某个因子进行正交化处理

    Parameters
    ----------
    y : pd.DataFrame
        研究的目标，回归中的y
    xs : Union[List[pd.DataFrame],pd.DataFrame]
        用于正交化的若干因子，回归中的x

    Returns
    -------
    pd.DataFrame
        正交化之后的因子
    """
    if not isinstance(xs, list):
        xs = [xs]
    # y = pure_fallmount(y)
    # xs = [pure_fallmount(i) for i in xs]
    # return (y - xs)()
    df=merge_many([y]+xs,how='inner')
    xs_str='+'.join([f'fac{i+2}' for i in range(len(xs))])
    def sing(date:pd.Timestamp):
        df0=df[df.date==date].set_index(['date','code'])
        if df0.shape[0]>0:
            ols=smf.ols('fac1~'+xs_str,data=df0).fit()
            df0.fac1=ols.resid
            return df0[['fac1']]
    dates=list(set(df.date))
    with mpire.WorkerPool(20) as pool:
        dfs=pool.map(sing,dates)
    dfs=pd.concat(dfs).reset_index().pivot(index='date',columns='code',values='fac1')
    return dfs