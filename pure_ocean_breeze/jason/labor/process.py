__updated__ = "2023-10-14 15:31:01"

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import tqdm.auto
import scipy.stats as ss
from scipy.optimize import linprog
import statsmodels.formula.api as smf
# import matplotlib as mpl

# mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

from functools import reduce, lru_cache, partial
from dateutil.relativedelta import relativedelta
from loguru import logger
import datetime
from collections.abc import Iterable
import plotly.express as pe
import plotly.io as pio
from plotly.tools import FigureFactory as FF
import plotly.graph_objects as go
import mpire

from texttable import Texttable

import cufflinks as cf

try:
    cf.set_config_file(offline=True)
except Exception:
    pass
from IPython.display import display
from typing import Callable, Union, Dict, List, Tuple
from pure_ocean_breeze.jason.data.read_data import (
    read_daily,
    read_market,
    get_industry_dummies,
)
from pure_ocean_breeze.jason.state.homeplace import HomePlace

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")
from pure_ocean_breeze.jason.state.states import STATES
from pure_ocean_breeze.jason.state.decorators import do_on_dfs
from pure_ocean_breeze.jason.data.dicts import INDUS_DICT
from pure_ocean_breeze.jason.data.tools import (
    drop_duplicates_index,
    to_percent,
    to_group,
    standardlize,
    select_max,
    select_min,
    merge_many,
)
from pure_ocean_breeze.jason.labor.comment import (
    comments_on_twins,
    make_relative_comments,
    make_relative_comments_plot,
)


@do_on_dfs
def daily_factor_on_industry(df: pd.DataFrame) -> dict:
    """将一个因子变为仅在某个申万一级行业上的股票

    Parameters
    ----------
    df : pd.DataFrame
        全市场的因子值，index是时间，columns是股票代码

    Returns
    -------
    dict
        key为行业代码，value为对应的行业上的因子值
    """
    df1 = df.resample("W").last()
    if df1.shape[0] * 2 > df.shape[0]:
        daily = 0
        weekly = 1
    else:
        daily = 1
        weekly = 0
    start = int(datetime.datetime.strftime(df.index.min(), "%Y%m%d"))
    ress = get_industry_dummies(
        daily=daily,
        weekly=weekly,
        start=start,
    )
    ress = {k: v * df for k, v in ress.items()}
    return ress


@do_on_dfs
def decap(df: pd.DataFrame, daily: bool = 0, monthly: bool = 0) -> pd.DataFrame:
    """对因子做市值中性化

    Parameters
    ----------
    df : pd.DataFrame
        未中性化的因子，index是时间，columns是股票代码
    daily : bool, optional
        未中性化因子是日频的则为1，否则为0, by default 0
    monthly : bool, optional
        未中性化因子是月频的则为1，否则为0, by default 0

    Returns
    -------
    `pd.DataFrame`
        市值中性化之后的因子

    Raises
    ------
    `NotImplementedError`
        如果未指定日频或月频，将报错
    """
    tqdm.auto.tqdm.pandas()
    cap = read_daily(flow_cap=1).stack().reset_index()
    cap.columns = ["date", "code", "cap"]
    cap.cap = ss.boxcox(cap.cap)[0]

    def single(x):
        x.cap = ss.boxcox(x.cap)[0]
        return x

    cap = cap.groupby(["date"]).apply(single)
    cap = cap.set_index(["date", "code"]).unstack()
    cap.columns = [i[1] for i in list(cap.columns)]
    cap_monthly = cap.resample("W").last()
    last = df.resample("W").last()
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


@do_on_dfs
def decap_industry(df: pd.DataFrame) -> pd.DataFrame:
    """对因子做行业市值中性化

    Parameters
    ----------
    df : pd.DataFrame
        未中性化的因子，index是时间，columns是股票代码

    Returns
    -------
    `pd.DataFrame`
        行业市值中性化之后的因子

    Raises
    ------
    `NotImplementedError`
        如果未指定日频或月频，将报错
    """
    start_date = int(datetime.datetime.strftime(df.index.min(), "%Y%m%d"))
    last = df.resample("W").last()
    homeplace = HomePlace()
    if df.shape[0] / last.shape[0] < 2:
        weekly = True
    else:
        daily = True
    if weekly:
        cap = read_daily(flow_cap=1, start=start_date).resample("W").last()
    else:
        cap = read_daily(flow_cap=1, start=start_date)
    cap = cap.stack().reset_index()
    cap.columns = ["date", "code", "cap"]
    cap.cap = ss.boxcox(cap.cap)[0]

    def single(x):
        x.cap = ss.boxcox(x.cap)[0]
        return x

    cap = cap.groupby(["date"]).apply(single)
    df = df.stack().reset_index()
    df.columns = ["date", "code", "fac"]
    df = pd.merge(df, cap, on=["date", "code"])

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

    file_name = "sw_industry_level1_dummies.parquet"

    if weekly:
        industry_dummy = (
            pd.read_parquet(homeplace.daily_data_file + file_name)
            .fillna(0)
            .set_index("date")
            .groupby("code")
            .resample("W")
            .last()
        )
        industry_dummy = industry_dummy.fillna(0).drop(columns=["code"]).reset_index()
        industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
        col = ["code", "date"] + industry_ws
    elif daily:
        industry_dummy = pd.read_parquet(homeplace.daily_data_file + file_name).fillna(
            0
        )
        industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
        col = ["date", "code"] + industry_ws
    else:
        raise NotImplementedError("必须指定频率")
    industry_dummy.columns = col
    df = pd.merge(df, industry_dummy, on=["date", "code"])
    df = df.set_index(["date", "code"])
    tqdm.auto.tqdm.pandas()
    df = df.groupby(["date"]).progress_apply(neutralize_factors)
    df = df.unstack()
    df.columns = [i[1] for i in list(df.columns)]
    return df


@do_on_dfs
def deboth(df: pd.DataFrame) -> pd.DataFrame:
    """通过回测的方式，对月频因子做行业市值中性化

    Parameters
    ----------
    df : pd.DataFrame
        未中性化的因子

    Returns
    -------
    `pd.DataFrame`
        行业市值中性化之后的因子
    """
    shen = pure_moonnight(df, boxcox=1, plt_plot=0, print_comments=0)
    return shen()


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
def boom_four(
    df: pd.DataFrame, backsee: int = 5, daily: bool = 0, min_periods: int = None
) -> Tuple[pd.DataFrame]:
    """生成20天均值，20天标准差，及二者正向z-score合成，正向排序合成，负向z-score合成，负向排序合成这6个因子

    Parameters
    ----------
    df : pd.DataFrame
        原日频因子
    backsee : int, optional
        回看天数, by default 20
    daily : bool, optional
        为1是每天都滚动，为0则仅保留月底值, by default 0
    min_periods : int, optional
        rolling时的最小期, by default backsee的一半

    Returns
    -------
    `Tuple[pd.DataFrame]`
        6个因子的元组
    """
    if min_periods is None:
        min_periods = int(backsee * 0.5)
    if not daily:
        df_mean = (
            df.rolling(backsee, min_periods=min_periods).mean().resample("W").last()
        )
        df_std = df.rolling(backsee, min_periods=min_periods).std().resample("W").last()
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


@do_on_dfs
def add_cross_standardlize(*args: list) -> pd.DataFrame:
    """将众多因子横截面做z-score标准化之后相加

    Returns
    -------
    `pd.DataFrame`
        合成后的因子
    """
    fms = [pure_fallmount(i) for i in args]
    one = fms[0]
    others = fms[1:]
    final = one + others
    return final()


@do_on_dfs
def to_tradeends(df: pd.DataFrame) -> pd.DataFrame:
    """将最后一个自然日改变为最后一个交易日

    Parameters
    ----------
    df : pd.DataFrame
        index为时间，为每个月的最后一天

    Returns
    -------
    `pd.DataFrame`
        修改为交易日标注后的pd.DataFrame
    """
    """"""
    start = df.index.min()
    start = start - pd.DateOffset(weeks=1)
    start = datetime.datetime.strftime(start, "%Y%m%d")
    trs = read_daily(tr=1, start=start)
    trs = trs.assign(tradeends=list(trs.index))
    trs = trs[["tradeends"]]
    trs = trs.resample("W").last()
    df = pd.concat([trs, df], axis=1)
    df = df.set_index(["tradeends"])
    return df


@do_on_dfs
def market_kind(
    df: pd.DataFrame,
    zhuban: bool = 0,
    chuangye: bool = 0,
    kechuang: bool = 0,
    beijing: bool = 0,
) -> pd.DataFrame:
    """与宽基指数成分股的函数类似，限定股票在某个具体板块上

    Parameters
    ----------
    df : pd.DataFrame
        原始全部股票的因子值
    zhuban : bool, optional
        限定在主板范围内, by default 0
    chuangye : bool, optional
        限定在创业板范围内, by default 0
    kechuang : bool, optional
        限定在科创板范围内, by default 0
    beijing : bool, optional
        限定在北交所范围内, by default 0

    Returns
    -------
    `pd.DataFrame`
        限制范围后的因子值，其余为空

    Raises
    ------
    `ValueError`
        如果未指定任何股票池，将报错
    """
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


def show_corr(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    method: str = "pearson",
    plt_plot: bool = 1,
    show_series: bool = 0,
    old_way: bool = 0,
) -> float:
    """展示两个因子的截面相关性

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    method : str, optional
        计算相关系数的方法, by default "pearson"
    plt_plot : bool, optional
        是否画出相关系数的时序变化图, by default 1
    show_series : bool, optional
        返回相关性的序列，而非均值
    old_way : bool, optional
        使用3.x版本的方式求相关系数

    Returns
    -------
    `float`
        平均截面相关系数
    """
    if old_way:
        if method == "spearman":
            corr = show_x_with_func(fac1, fac2, lambda x: x.rank().corr().iloc[0, 1])
        else:
            corr = show_x_with_func(
                fac1, fac2, lambda x: x.corr(method=method).iloc[0, 1]
            )
    else:
        corr = fac1.corrwith(fac2, axis=1, method=method)
    if show_series:
        return corr
    else:
        if plt_plot:
            corr.plot(rot=60)
            plt.show()
        return corr.mean()


def show_corrs(
    factors: List[pd.DataFrame],
    factor_names: List[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
    method: str = "pearson",
) -> pd.DataFrame:
    """展示很多因子两两之间的截面相关性

    Parameters
    ----------
    factors : List[pd.DataFrame]
        所有因子构成的列表, by default None
    factor_names : List[str], optional
        上述因子依次的名字, by default None
    print_bool : bool, optional
        是否打印出两两之间相关系数的表格, by default True
    show_percent : bool, optional
        是否以百分数的形式展示, by default True
    method : str, optional
        计算相关系数的方法, by default "pearson"

    Returns
    -------
    `pd.DataFrame`
        两两之间相关系数的表格
    """
    corrs = []
    for i in range(len(factors)):
        main_i = factors[i]
        follows = factors[i + 1 :]
        corr = [show_corr(main_i, i, plt_plot=False, method=method) for i in follows]
        corr = [np.nan] * (i + 1) + corr
        corrs.append(corr)
    if factor_names is None:
        factor_names = [f"fac{i}" for i in list(range(1, len(factors) + 1))]
    corrs = pd.DataFrame(corrs, columns=factor_names, index=factor_names)
    np.fill_diagonal(corrs.to_numpy(), 1)
    corrs=pd.DataFrame(corrs.fillna(0).to_numpy()+corrs.fillna(0).to_numpy().T-np.diag(np.diag(corrs)),index=corrs.index,columns=corrs.columns)
    if show_percent:
        pcorrs = corrs.applymap(to_percent)
    else:
        pcorrs = corrs.copy()
    if print_bool:
        return pcorrs
    else:
        return corrs


def show_cov(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    plt_plot: bool = 1,
    show_series: bool = 0,
) -> float:
    """展示两个因子的截面相关性

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    method : str, optional
        计算相关系数的方法, by default "spearman"
    plt_plot : bool, optional
        是否画出相关系数的时序变化图, by default 1
    show_series : bool, optional
        返回相关性的序列，而非均值

    Returns
    -------
    `float`
        平均截面相关系数
    """
    cov = show_x_with_func(fac1, fac2, lambda x: x.cov().iloc[0, 1])
    if show_series:
        return cov
    else:
        if plt_plot:
            cov.plot(rot=60)
            plt.show()
        return cov.mean()


def show_x_with_func(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    func: Callable,
) -> pd.Series:
    """展示两个因子的某种截面关系

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    func : Callable
        要对两个因子在截面上的进行的操作

    Returns
    -------
    `pd.Series`
        截面关系
    """
    the_func = partial(func)
    both1 = fac1.stack().reset_index()
    befo1 = fac2.stack().reset_index()
    both1.columns = ["date", "code", "both"]
    befo1.columns = ["date", "code", "befo"]
    twins = pd.merge(both1, befo1, on=["date", "code"]).set_index(["date", "code"])
    corr = twins.groupby("date").apply(the_func)
    return corr


def show_covs(
    factors: List[pd.DataFrame],
    factor_names: List[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
) -> pd.DataFrame:
    """展示很多因子两两之间的截面相关性

    Parameters
    ----------
    factors : List[pd.DataFrame]
        所有因子构成的列表, by default None
    factor_names : List[str], optional
        上述因子依次的名字, by default None
    print_bool : bool, optional
        是否打印出两两之间相关系数的表格, by default True
    show_percent : bool, optional
        是否以百分数的形式展示, by default True

    Returns
    -------
    `pd.DataFrame`
        两两之间相关系数的表格
    """
    corrs = []
    for i in range(len(factors)):
        main_i = factors[i]
        follows = factors[i + 1 :]
        corr = [show_cov(main_i, i, plt_plot=False) for i in follows]
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


@do_on_dfs
def show_corrs_with_old(
    df: pd.DataFrame = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """计算新因子和已有因子的相关系数

    Parameters
    ----------
    df : pd.DataFrame, optional
        新因子, by default None
    method : str, optional
        求相关系数的方法, by default 'pearson'


    Returns
    -------
    pd.DataFrame
        相关系数矩阵
    """
    files = os.listdir(homeplace.final_factor_file)
    names = [i[:-8] for i in files]
    files = [homeplace.final_factor_file + i for i in files]
    files = [pd.read_parquet(i) for i in files]
    if df is not None:
        corrs = show_corrs([df] + files, names, method=method)
    else:
        corrs = show_corrs(files, names, method=method)
    return corrs


@do_on_dfs
def remove_unavailable(df: pd.DataFrame) -> pd.DataFrame:
    """对日频或月频因子值，剔除st股、不正常交易的股票和上市不足60天的股票

    Parameters
    ----------
    df : pd.DataFrame
        因子值，index是时间，columns是股票代码，values是因子值

    Returns
    -------
    pd.DataFrame
        剔除后的因子值
    """
    df0 = df.resample("W").last()
    if df.shape[0] / df0.shape[0] > 2:
        daily = 1
    else:
        daily = 0
    state = read_daily(state=1).replace(0, np.nan)
    if daily:
        df = df * state
    else:
        df = state.resample("W").first() * df
    return df


class frequency_controller(object):
    def __init__(self, freq: str):
        self.homeplace = HomePlace()
        self.freq = freq

        if freq == "M":
            self.counts_one_year = 12
            self.time_shift = pd.DateOffset(months=1)
            self.states_files = (
                self.homeplace.daily_data_file + "states_monthly.parquet"
            )
            self.sts_files = self.homeplace.daily_data_file + "sts_monthly.parquet"
            self.comment_name = "月"
            self.days_in = 20
        elif freq == "W":
            self.counts_one_year = 50
            self.time_shift = pd.DateOffset(weeks=1)
            self.states_files = self.homeplace.daily_data_file + "states_weekly.parquet"
            self.sts_files = self.homeplace.daily_data_file + "sts_weekly.parquet"
            self.comment_name = "周"
            self.days_in = 5
        else:
            raise ValueError("'暂时不支持此频率哈🤒，目前仅支持月频'M'，和周频'W'")

    def next_end(self, x):
        """找到下个周期的最后一天"""
        if self.freq == "M":
            return x + pd.DateOffset(months=1) + pd.tseries.offsets.MonthEnd()
        elif self.freq == "W":
            return x + pd.DateOffset(weeks=1)
        else:
            raise ValueError("'暂时不支持此频率哈🤒，目前仅支持月频'M'，和周频'W'")


class pure_moon(object):
    __slots__ = [
        "homeplace",
        "sts_monthly_file",
        "states_monthly_file",
        "factors",
        "codes",
        "tradedays",
        "ages",
        "amounts",
        "closes",
        "opens",
        "capital",
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
        "big_small_rankic",
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
        "__factors_out",
        "ics",
        "rankics",
        "factor_turnover_rates",
        "factor_turnover_rate",
        "group_rets_std",
        "group_rets_stds",
        "group_rets_skews",
        "group_rets_skew",
        "wind_out",
        "swindustry_dummy",
        "zxindustry_dummy",
        "closes2_monthly",
        "rets_monthly_last",
        "freq_ctrl",
        "freq",
        "factor_cover",
        "factor_cross_skew",
        "factor_cross_skew_after_neu",
        "pos_neg_rate",
        "corr_itself",
        "factor_cross_stds",
        "corr_itself_shift2",
        "rets_all",
        "inner_long_ret_yearly",
        "inner_short_ret_yearly",
        "inner_long_net_values",
        "inner_short_net_values",
        "group_mean_rets_monthly",
        "not_ups",
        "not_downs",
        "group1_ret_yearly",
        "group10_ret_yearly",
        "market_ret",
        "long_minus_market_rets",
        "long_minus_market_nets",
        "inner_rets_long",
        "inner_rets_short",
    ]

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(
        cls,
        freq: str = "W",
        no_read_indu: bool = 0,
    ):
        cls.homeplace = HomePlace()
        cls.freq = freq
        cls.freq_ctrl = frequency_controller(freq)

        def deal_dummy(industry_dummy):
            industry_dummy = industry_dummy.drop(columns=["code"]).reset_index()
            industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
            col = ["code", "date"] + industry_ws
            industry_dummy.columns = col
            industry_dummy = industry_dummy[
                industry_dummy.date >= pd.Timestamp(str(STATES["START"]))
            ]
            return industry_dummy

        if not no_read_indu:
            # week_here
            cls.swindustry_dummy = (
                pd.read_parquet(
                    cls.homeplace.daily_data_file + "sw_industry_level1_dummies.parquet"
                )
                .fillna(0)
                .set_index("date")
                .groupby("code")
                .resample(freq)
                .last()
            )
            cls.swindustry_dummy = deal_dummy(cls.swindustry_dummy)

    @property
    def factors_out(self):
        return self.__factors_out

    def __call__(self):
        """调用对象则返回因子值"""
        return self.factors_out

    @classmethod
    @lru_cache(maxsize=None)
    def set_basic_data(
        cls,
        total_cap: bool = 0,
    ):
        states = read_daily(state=1, start=STATES["START"])
        opens = read_daily(vwap=1, start=STATES["START"])
        closes = read_daily(vwap=1, start=STATES["START"])
        if total_cap:
            capitals = (
                read_daily(total_cap=1, start=STATES["START"]).resample(cls.freq).last()
            )
        else:
            capitals = (
                read_daily(flow_cap=1, start=STATES["START"]).resample(cls.freq).last()
            )
        market=read_market(zz500=1,every_stock=0,close=1).resample(cls.freq).last()
        cls.market_ret=market/market.shift(1)-1
        # 交易状态文件
        cls.states = states
        # Monday vwap
        cls.opens = opens
        # Friday vwap
        cls.closes = closes
        # 月底流通市值数据
        cls.capital = capitals
        cls.opens = cls.opens.replace(0, np.nan)
        cls.closes = cls.closes.replace(0, np.nan)
        cls.states = read_daily(state=1)
        cls.states = cls.states.resample(cls.freq).first()
        # cls.states=np.sign(cls.states.where(cls.states==cls.states.max().max(),np.nan))
        up_downs = read_daily(up_down_limit_status=1)
        cls.not_ups = (
            np.sign(up_downs.where(up_downs != 1, np.nan).abs() + 1)
            .resample(cls.freq)
            .first()
        )
        cls.not_downs = (
            np.sign(up_downs.where(up_downs != -1, np.nan).abs() + 1)
            .resample(cls.freq)
            .last()
        )

    def set_factor_df_date_as_index(self, df: pd.DataFrame):
        """设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间"""
        # week_here
        self.factors = df.resample(self.freq).last().dropna(how="all")
        self.factors = self.factors * self.states
        self.factor_cover = np.sign(self.factors.abs() + 1).sum().sum()
        opens = self.opens[self.opens.index >= self.factors.index.min()]
        total = np.sign(opens.resample(self.freq).last()).sum().sum()
        self.factor_cover = min(self.factor_cover / total, 1)
        self.factor_cross_skew = self.factors.skew(axis=1).mean()
        pos_num = ((self.factors > 0) + 0).sum().sum()
        neg_num = ((self.factors < 0) + 0).sum().sum()
        self.pos_neg_rate = pos_num / (neg_num + pos_num)
        self.corr_itself = show_corr(self.factors, self.factors.shift(1), plt_plot=0)
        self.corr_itself_shift2 = show_corr(
            self.factors, self.factors.shift(2), plt_plot=0
        )
        self.factor_cross_stds = self.factors.std(axis=1)
        

    @classmethod
    @lru_cache(maxsize=None)
    def get_rets_month(cls):
        """计算每月的收益率，并根据每月做出交易状态，做出删减"""
        # week_here
        cls.opens_monthly = cls.opens.resample(cls.freq).first()
        # week_here
        cls.closes_monthly = cls.closes.resample(cls.freq).last()
        cls.rets_monthly = (cls.closes_monthly - cls.opens_monthly) / cls.opens_monthly
        cls.rets_monthly = cls.rets_monthly.stack().reset_index()
        cls.rets_monthly.columns = ["date", "code", "ret"]

    # @classmethod
    def neutralize_factors(self, date):
        """组内对因子进行市值中性化"""
        df=self.factors[self.factors.date==date].set_index(['date','code'])
        industry_codes = list(df.columns)
        industry_codes = [i for i in industry_codes if i.startswith("w")]
        industry_codes_str = "+".join(industry_codes)
        if len(industry_codes_str) > 0:
            ols_result = smf.ols("fac~cap_size+" + industry_codes_str, data=df).fit()
        else:
            ols_result = smf.ols("fac~cap_size", data=df).fit()
        df.fac=ols_result.resid
        df = df[["fac"]]
        return df

    @classmethod
    @lru_cache(maxsize=None)
    def get_log_cap(cls):
        """获得对数市值"""
        cls.cap = cls.capital.stack().reset_index()
        cls.cap.columns = ["date", "code", "cap_size"]
        cls.cap["cap_size"] = np.log(cls.cap["cap_size"])

    def get_neutral_factors(self, only_cap=0):
        """对因子进行行业市值中性化"""
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.factors = pd.merge(
            self.factors, self.cap, how="inner", on=["date", "code"]
        )
        if not only_cap:
            self.factors = pd.merge(
                self.factors, self.swindustry_dummy, on=["date", "code"]
            )

        dates=list(set(self.factors.date))
        with mpire.WorkerPool(20) as pool:
            res=pool.map(self.neutralize_factors,dates)
        self.factors = pd.concat(res)
        self.factors = self.factors.reset_index()
        self.factors = self.factors.pivot(index="date", columns="code", values="fac")

    def get_ic_rankic(cls, df):
        """计算IC和RankIC"""
        df1 = df[["ret", "fac"]]
        ic = df1.corr(method="pearson").iloc[0, 1]
        rankic = df1.rank().corr().iloc[0, 1]
        small_ic=df1[df1.fac<=df1.fac.median()].rank().corr().iloc[0, 1]
        big_ic=df1[df1.fac>=df1.fac.median()].rank().corr().iloc[0, 1]
        df2 = pd.DataFrame({"ic": [ic], "rankic": [rankic],"small_rankic":[small_ic],"big_rankic":[big_ic]})
        return df2

    def get_icir_rankicir(cls, df):
        """计算ICIR和RankICIR"""
        ic = df.ic.mean()
        rankic = df.rankic.mean()
        small_rankic=df.small_rankic.mean()
        big_rankic=df.big_rankic.mean()
        # week_here
        icir = ic / np.std(df.ic) * (cls.freq_ctrl.counts_one_year ** (0.5))
        # week_here
        rankicir = rankic / np.std(df.rankic) * (cls.freq_ctrl.counts_one_year ** (0.5))
        small_rankicir = small_rankic / np.std(df.small_rankic) * (cls.freq_ctrl.counts_one_year ** (0.5))
        big_rankicir = big_rankic / np.std(df.big_rankic) * (cls.freq_ctrl.counts_one_year ** (0.5))
        return pd.DataFrame(
            {"IC": [ic], "ICIR": [icir], "RankIC": [rankic], "RankICIR": [rankicir]},
            index=["评价指标"],
        ),pd.DataFrame({"1-5RankIC":[small_rankic],"1-5ICIR":[small_rankicir],"6-10RankIC":[big_rankic],"6-10ICIR":[big_rankicir]},index=["评价指标"]).T

    def get_ic_icir_and_rank(cls, df):
        """计算IC、ICIR、RankIC、RankICIR"""
        df1 = df.groupby("date").apply(cls.get_ic_rankic)
        cls.ics = df1.ic
        cls.rankics = df1.rankic
        cls.ics = cls.ics.reset_index(drop=True, level=1).to_frame()
        cls.rankics = cls.rankics.reset_index(drop=True, level=1).to_frame()
        df2,df5 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        dura = (df.date.max() - df.date.min()).days / 365
        t_value = df2.iloc[3, 0] * (dura ** (1 / 2))
        df3 = pd.DataFrame({"评价指标": [t_value]}, index=["RankIC.t"])
        df4 = pd.concat([df2, df3])
        df
        return df4,df5

    @classmethod
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

    def get_data(self, groups_num):
        """拼接因子数据和每月收益率数据，并对涨停和跌停股加以处理"""
        self.data = pd.merge(
            self.rets_monthly, self.factors, how="inner", on=["date", "code"]
        )
        self.ic_icir_and_rank,self.big_small_rankic = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby("date").apply(
            lambda x: self.get_groups(x, groups_num)
        )
        self.wind_out = self.data.copy()
        self.factor_turnover_rates = self.data.pivot(
            index="date", columns="code", values="group"
        )
        rates = []
        for i in range(1, groups_num + 1):
            son = (self.factor_turnover_rates == i) + 0
            son1 = son.diff()
            # self.factor_turnover_rates = self.factor_turnover_rates.diff()
            change = ((np.abs(np.sign(son1)) == 1) + 0).sum(axis=1)
            still = (((son1 == 0) + 0) * son).sum(axis=1)
            rate = change / (change + still)
            rates.append(rate.to_frame(f"group{i}"))
        rates = pd.concat(rates, axis=1).fillna(0)
        self.factor_turnover_rates = rates
        self.data = self.data.reset_index(drop=True)

    def to_group_ret(self, l):
        """每一组的年化收益率"""
        # week_here
        ret = l[-1] / len(l) * self.freq_ctrl.counts_one_year
        return ret

    def make_start_to_one(self, l):
        """让净值序列的第一个数变成1"""
        min_date = self.factors.date.min()
        add_date = min_date - pd.DateOffset(weeks=1)
        add_l = pd.Series([0], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    def get_group_rets_net_values(
        self, groups_num=10, value_weighted=False, trade_cost_double_side=0
    ):
        """计算组内每一期的平均收益，生成每日收益率序列和净值序列"""
        if value_weighted:
            cap_value = self.capital.copy()
            # week_here
            cap_value = cap_value.resample(self.freq).last().shift(1)
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
            self.rets_all = self.data.groupby(["date"]).apply(in_g)
            self.group_rets_std = "市值加权暂未设置该功能，敬请期待🌙"
        else:
            self.group_rets = self.data.groupby(["date", "group"]).apply(
                lambda x: x.ret.mean()
            )
            self.rets_all = self.data.groupby(["date"]).apply(lambda x: x.ret.mean())
            self.group_rets_stds = self.data.groupby(["date", "group"]).ret.std()
            self.group_rets_std = (
                self.group_rets_stds.reset_index().groupby("group").mean()
            )
            self.group_rets_skews = self.data.groupby(["date", "group"]).ret.skew()
            self.group_rets_skew = (
                self.group_rets_skews.reset_index().groupby("group").mean()
            )
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[
            self.group_rets.index <= self.factors.date.max()
        ]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix("group")
        self.group_rets = (
            self.group_rets - self.factor_turnover_rates * trade_cost_double_side
        )
        self.rets_all = (
            self.rets_all
            - self.factor_turnover_rates.mean(axis=1) * trade_cost_double_side
        ).dropna()
        self.long_short_rets = (
            self.group_rets["group1"] - self.group_rets["group" + str(groups_num)]
        )
        self.inner_rets_long = self.group_rets.group1 - self.rets_all
        self.inner_rets_short = (
            self.rets_all - self.group_rets["group" + str(groups_num)]
        )
        self.long_short_net_values = self.make_start_to_one(
            self.long_short_rets.cumsum()
        )
        self.long_minus_market_rets=self.group_rets.group1-self.market_ret
        
        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = (
                self.group_rets["group" + str(groups_num)] - self.group_rets["group1"]
            )
            self.long_short_net_values = self.make_start_to_one(
                self.long_short_rets.cumsum()
            )
            self.inner_rets_long = (
                self.group_rets["group" + str(groups_num)] - self.rets_all
            )
            self.inner_rets_short = self.rets_all - self.group_rets.group1
            self.long_minus_market_rets=self.group_rets['group'+str(groups_num)]-self.market_ret
        self.long_minus_market_nets=self.make_start_to_one(self.long_minus_market_rets.dropna().cumsum())
        self.inner_long_net_values = self.make_start_to_one(
            self.inner_rets_long.cumsum()
        )
        self.inner_short_net_values = self.make_start_to_one(
            self.inner_rets_short.cumsum()
        )
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.cumsum()
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

    def get_long_short_comments(self, on_paper=False):
        """计算多空对冲的相关评价指标
        包括年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        # week_here
        self.long_short_ret_yearly = self.long_short_net_values[-1] * (
            self.freq_ctrl.counts_one_year / len(self.long_short_net_values)
        )
        self.inner_long_ret_yearly = self.inner_long_net_values[-1] * (
            self.freq_ctrl.counts_one_year / len(self.inner_long_net_values)
        )
        self.inner_short_ret_yearly = self.inner_short_net_values[-1] * (
            self.freq_ctrl.counts_one_year / len(self.inner_short_net_values)
        )
        self.group1_ret_yearly= self.group_net_values["group1"][-1] * (
            self.freq_ctrl.counts_one_year / len(self.group_net_values.group1)
        )
        self.group10_ret_yearly = self.group_net_values["group10"][-1] * (
            self.freq_ctrl.counts_one_year / len(self.group_net_values.group10)
        )
        # week_here
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (
            self.freq_ctrl.counts_one_year**0.5
        )
        self.long_short_info_ratio = (
            self.long_short_ret_yearly / self.long_short_vol_yearly
        )
        self.long_short_win_times = len(self.long_short_rets[self.long_short_rets > 0])
        self.long_short_win_ratio = self.long_short_win_times / len(
            self.long_short_rets
        )
        self.max_retreat = -(
            (self.long_short_net_values + 1)
            / (self.long_short_net_values + 1).expanding(1).max()
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
                # week_here
                index=[
                    "年化收益率",
                    "年化波动率",
                    "收益波动比",
                    f"{self.freq_ctrl.comment_name}度胜率",
                    "最大回撤率",
                ],
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
                # week_here
                index=[
                    "年化收益率",
                    "年化波动率",
                    "信息比率",
                    f"{self.freq_ctrl.comment_name}度胜率",
                    "最大回撤率",
                ],
            )

    def get_total_comments(self, groups_num):
        """综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、胜率、最大回撤率"""
        rankic = self.rankics.mean()
        rankic_win = self.rankics[self.rankics * rankic > 0]
        rankic_win_ratio = rankic_win.dropna().shape[0] / self.rankics.dropna().shape[0]
        self.factor_cross_skew_after_neu = self.__factors_out.skew(axis=1).mean()
        if self.ic_icir_and_rank.iloc[2, 0] > 0:
            self.factor_turnover_rate = self.factor_turnover_rates[
                f"group{groups_num}"
            ].mean()
        else:
            self.factor_turnover_rate = self.factor_turnover_rates["group1"].mean()
        self.total_comments = pd.concat(
            [
                self.ic_icir_and_rank,
                pd.DataFrame(
                    {"评价指标": [rankic_win_ratio]},
                    index=["RankIC胜率"],
                ),
                self.long_short_comments,
                # week_here
                pd.DataFrame(
                    {
                        "评价指标": [
                            self.factor_turnover_rate,
                            self.factor_cover,
                            self.pos_neg_rate,
                            self.factor_cross_skew,
                            self.inner_long_ret_yearly,
                            self.inner_long_ret_yearly
                            / (
                                self.inner_long_ret_yearly + self.inner_short_ret_yearly
                            ),
                            self.corr_itself,
                        ]
                    },
                    index=[
                        f"多头{self.freq_ctrl.comment_name}均换手",
                        "因子覆盖率",
                        "因子正值占比",
                        "因子截面偏度",
                        "多头超均收益",
                        "多头收益占比",
                        "一阶自相关性",
                    ],
                ),
                self.big_small_rankic,
                pd.DataFrame(
                    {
                        "评价指标": [
                            self.group1_ret_yearly,
                            self.group10_ret_yearly,
                        ]
                    },
                    index=[
                        "1组年化收益",
                        "10组年化收益",
                    ]
                )
            ]
        )
        # print(self.total_comments)
        self.group_mean_rets_monthly = self.group_rets.drop(
            columns=["long_short"]
        ).mean()
        # self.group_mean_rets_monthly = (
        #     self.group_mean_rets_monthly - self.group_mean_rets_monthly.mean()
        # )
        mar=self.market_ret.loc[self.factors_out.index]
        self.group_mean_rets_monthly = (
            self.group_mean_rets_monthly - mar.mean()
        )*self.freq_ctrl.counts_one_year

    def plot_net_values(self, y2, filename, iplot=1, ilegend=1, without_breakpoint=0):
        """使用matplotlib来画图，y2为是否对多空组合采用双y轴"""
        if not iplot:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(33, 8))
            self.group_net_values.plot(secondary_y=y2, rot=60, ax=ax[0])
            self.group_net_values.plot(secondary_y=y2, ax=ax[0])
            b = self.rankics.copy()
            b.index = [int(i.year) if i.month == 1 else "" for i in list(b.index)]
            b.plot(kind="bar", rot=60, ax=ax[1])
            self.factor_cross_stds.plot(rot=60, ax=ax[2])

            filename_path = filename + ".png"
            if not STATES["NO_SAVE"]:
                plt.savefig(filename_path)
        else:

            tris = self.group_net_values.drop(columns=['long_short'])
            if without_breakpoint:
                tris = tris.dropna()
            figs = cf.figures(
                tris,
                [
                    dict(kind="line", y=list(tris.columns)),
                    # dict(kind="bar", y="各组月均超均收益"),
                    # dict(kind="bar", y="rankic"),
                ],
                asList=True,
            )
            comments = (
                self.total_comments.applymap(lambda x: round(x, 4))
                .rename(index={"RankIC均值t值": "RankIC.t"})
                .reset_index()
            )
            here = pd.concat(
                [
                    comments.iloc[:6, :].reset_index(drop=True),
                    comments.iloc[6:12, :].reset_index(drop=True),
                    comments.iloc[12:18, :].reset_index(drop=True),
                    comments.iloc[18:, :].reset_index(drop=True),
                ],
                axis=1,
            )
            here.columns = ["信息系数", "结果", "绩效指标", "结果", "其他指标", "结果","单侧","结果"]
            # here=here.to_numpy().tolist()+[['信息系数','结果','绩效指标','结果']]
            table = FF.create_table(here.iloc[::-1],xgap=0)
            table.update_yaxes(matches=None)
            pic2 = go.Figure(
                go.Bar(
                    y=list(self.group_mean_rets_monthly),
                    x=[
                        i.replace("roup", "")
                        for i in list(self.group_mean_rets_monthly.index)
                    ],
                )
            )
            # table=go.Figure([go.Table(header=dict(values=list(here.columns)),cells=dict(values=here.to_numpy().tolist()))])
            pic3_data = go.Bar(y=list(self.rankics.rankic), x=list(self.rankics.index),marker_color="red")
            pic3 = go.Figure(data=[pic3_data])
            pic5_data = go.Line(
                y=list(self.rankics.rankic.cumsum()),
                x=list(self.rankics.index),
                name="y2",
                yaxis="y2",
            )
            pic5_layout = go.Layout(yaxis2=dict(title="y2", side="right"))
            pic5 = go.Figure(data=[pic5_data], layout=pic5_layout)
            figs.append(table)
            figs = [figs[-1]] + figs[:-1]
            figs.append(pic2)
            figs = [figs[0], figs[1], figs[-1], pic3]
            figs[1].update_layout(
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            figs[3].update_layout(yaxis2=dict(title="y2", side="right"))
            # twins=pd.concat([self.long_short_net_values,self.long_minus_market_nets],axis=1)
            pic4=go.Figure()
            figs.append(pic4)
            figs[4].add_trace(go.Line(x=self.long_short_net_values.index,y=self.long_short_net_values,name='long_short'))
            figs[4].add_trace(go.Line(x=self.long_minus_market_nets.index,y=self.long_minus_market_nets,name='long_minus_market'))
            figs.append(pic5)
            base_layout = cf.tools.get_base_layout(figs)

            sp = cf.subplots(
                figs,
                shape=(2, 14),
                base_layout=base_layout,
                vertical_spacing=0.15,
                horizontal_spacing=0.025,
                shared_yaxes=False,
                specs=[
                    [
                        # None,
                        {"rowspan": 2, "colspan": 5},
                        None,
                        None,
                        None,
                        None,
                        {"rowspan": 2, "colspan": 3},
                        None,
                        None,
                        {"colspan": 3},
                        None,
                        None,
                        {"colspan": 3},
                        None,
                        None,
                    ],
                    [
                        # None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        {"colspan": 3},
                        None,
                        None,
                        {"colspan": 3},
                        None,
                        None,
                    ],
                ],
                subplot_titles=[
                    "净值曲线",
                    "各组月均超均收益",
                    "Rank IC时序图",
                    "绩效指标",
                ],
            )
            sp["layout"].update(showlegend=ilegend,width=1780,height=230,margin=dict(l=0, r=0, b=0, t=0, pad=0))
            # sp['colors']=['#FF5733', '#33FF57', '#3357FF']
            # los=sp['layout']['annotations']
            # los[0]['font']['color']='#000000'
            # los[1]['font']['color']='#000000'
            # los[2]['font']['color']='#000000'
            # los[3]['font']['color']='#000000'
            # los[-1]['font']['color']='#ffffff'
            # los[-2]['font']['color']='#ffffff'
            # los[-3]['font']['color']='#ffffff'
            # los[-4]['font']['color']='#ffffff'
            # los[0]['text']=los[0]['text'][3:-4]
            # los[1]['text']=los[1]['text'][3:-4]
            # los[2]['text']=los[2]['text'][3:-4]
            # los[3]['text']=los[3]['text'][3:-4]
            # los[-1]['text']='<b>'+los[-1]['text']+'</b>'
            # los[-2]['text']='<b>'+los[-2]['text']+'</b>'
            # los[-3]['text']='<b>'+los[-3]['text']+'</b>'
            # los[-4]['text']='<b>'+los[-4]['text']+'</b>'
            # sp['layout']['annotations']=los
            # print(sp['layout']['annotations'])
            # sp['layout']['annotations'][0]['yanchor']='top'

            cf.iplot(sp)
            # tris=pd.concat([self.group_net_values,self.rankics,self.factor_turnover_rates],axis=1).rename(columns={0:'turnover_rate'})
            # sp=plyoo.make_subplots(rows=2,cols=8,vertical_spacing=.15,horizontal_spacing=.03,
            #                specs=[[{'rowspan':2,'colspan':2,'type':'domain'},None,{'rowspan':2,'colspan':4,'type':'xy'},None,None,None,{'colspan':2,'type':'xy'},None],
            #                       [None,None,None,None,None,None,{'colspan':2,'type':'xy'},None]],
            #                subplot_titles=['净值曲线','Rank IC时序图','月换手率','绩效指标'])
            # comments=self.total_comments.applymap(lambda x:round(x,4)).rename(index={'RankIC均值t值':'RankIC.t'}).reset_index()
            # here=pd.concat([comments.iloc[:5,:].reset_index(drop=True),comments.iloc[5:,:].reset_index(drop=True)],axis=1)
            # here.columns=['信息系数','结果','绩效指标','结果']
            # table=FF.create_table(here)
            # sp.add_trace(table)

    def plotly_net_values(self, filename):
        """使用plotly.express画图"""
        fig = pe.line(self.group_net_values)
        filename_path = filename + ".html"
        pio.write_html(fig, filename_path, auto_open=True)

    @classmethod
    @lru_cache(maxsize=None)
    def prerpare(cls):
        """通用数据准备"""
        cls.get_rets_month()

    def run(
        self,
        groups_num=10,
        neutralize=False,
        trade_cost_double_side=0,
        value_weighted=False,
        y2=False,
        plt_plot=True,
        plotly_plot=False,
        filename="分组净值图",
        print_comments=True,
        comments_writer=None,
        net_values_writer=None,
        rets_writer=None,
        comments_sheetname=None,
        net_values_sheetname=None,
        rets_sheetname=None,
        on_paper=False,
        sheetname=None,
        only_cap=0,
        iplot=1,
        ilegend=1,
        without_breakpoint=0,
        beauty_comments=0,
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
            self.get_neutral_factors(only_cap=only_cap)
        self.__factors_out = self.factors.copy()
        self.factors = self.factors.shift(1)
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.get_data(groups_num)
        self.get_group_rets_net_values(
            groups_num=groups_num,
            value_weighted=value_weighted,
            trade_cost_double_side=trade_cost_double_side,
        )
        self.get_long_short_comments(on_paper=on_paper)
        self.get_total_comments(groups_num=groups_num)
        if on_paper:
            group1_ttest = ss.ttest_1samp(self.group_rets.group1, 0).pvalue
            group10_ttest = ss.ttest_1samp(
                self.group_rets[f"group{groups_num}"], 0
            ).pvalue
            group_long_short_ttest = ss.ttest_1samp(self.long_short_rets, 0).pvalue
            group1_ret = self.group_rets.group1.mean()
            group10_ret = self.group_rets[f"group{groups_num}"].mean()
            group_long_short_ret = self.long_short_rets.mean()
            papers = pd.DataFrame(
                {
                    "评价指标": [
                        group1_ttest,
                        group10_ttest,
                        group_long_short_ttest,
                        group1_ret,
                        group10_ret,
                        group_long_short_ret,
                    ]
                },
                index=[
                    "分组1p值",
                    f"分组{groups_num}p值",
                    f"分组1-分组{groups_num}p值",
                    "分组1收益率",
                    f"分组{groups_num}收益率",
                    f"分组1-分组{groups_num}收益率",
                ],
            )
            self.total_comments = pd.concat([papers, self.total_comments])

        if plt_plot:
            if not STATES["NO_PLOT"]:
                if filename:
                    self.plot_net_values(
                        y2=y2,
                        filename=filename,
                        iplot=iplot,
                        ilegend=bool(ilegend),
                        without_breakpoint=without_breakpoint,
                    )
                else:
                    self.plot_net_values(
                        y2=y2,
                        filename=self.factors_file.split(".")[-2].split("/")[-1]
                        + str(groups_num)
                        + "分组",
                        iplot=iplot,
                        ilegend=bool(ilegend),
                        without_breakpoint=without_breakpoint,
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
                tb = Texttable()
                tb.set_cols_width(
                    [8] * 5 + [9] + [8] * 2 + [7] * 2 + [8] + [8] + [9] + [10] * 5
                )
                tb.set_cols_dtype(["f"] * 18)
                tb.header(list(self.total_comments.T.columns))
                tb.add_rows(self.total_comments.T.to_numpy(), header=False)
                print(tb.draw())
        if sheetname:
            if comments_writer:
                if not on_paper:
                    total_comments = self.total_comments.copy()
                    tc = list(total_comments.评价指标)
                    if beauty_comments:
                        tc[0] = str(round(tc[0] * 100, 2)) + "%"
                        tc[1] = str(round(tc[1], 2))
                        tc[2] = str(round(tc[2] * 100, 2)) + "%"
                        tc[3] = str(round(tc[3], 2))
                        tc[4] = str(round(tc[4], 2))
                        tc[5] = str(round(tc[5] * 100, 2)) + "%"
                        tc[6] = str(round(tc[6] * 100, 2)) + "%"
                        tc[7] = str(round(tc[7] * 100, 2)) + "%"
                        tc[8] = str(round(tc[8], 2))
                        tc[9] = str(round(tc[9] * 100, 2)) + "%"
                        tc[10] = str(round(tc[10] * 100, 2)) + "%"
                        tc[11] = str(round(tc[11] * 100, 2)) + "%"
                        tc[12] = str(round(tc[12] * 100, 2)) + "%"
                        tc[13] = str(round(tc[13] * 100, 2)) + "%"
                        tc[14] = str(round(tc[14], 2))
                        tc[15] = str(round(tc[15], 2))
                        tc[16] = str(round(tc[16] * 100, 2)) + "%"
                        tc[17] = str(round(tc[17] * 100, 2)) + "%"
                    tc = tc + list(self.group_mean_rets_monthly)
                    new_total_comments = pd.DataFrame(
                        {sheetname: tc},
                        index=list(total_comments.index)
                        + [f"第{i}组" for i in range(1, groups_num + 1)],
                    )
                    new_total_comments.to_excel(comments_writer, sheet_name=sheetname)
                    rankic_twins = pd.concat(
                        [self.rankics.rankic, self.rankics.rankic.cumsum()], axis=1
                    )
                    rankic_twins.columns = ["RankIC", "RankIC累积"]
                    rankic_twins.to_excel(
                        comments_writer, sheet_name=sheetname + "RankIC"
                    )
                else:
                    self.total_comments.rename(
                        columns={"评价指标": sheetname}
                    ).to_excel(comments_writer, sheet_name=sheetname)
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
                tc[7] = str(round(tc[7] * 100, 2)) + "%"
                tc[8] = str(round(tc[8], 2))
                tc[9] = str(round(tc[9] * 100, 2)) + "%"
                tc[10] = str(round(tc[10] * 100, 2)) + "%"
                tc[11] = str(round(tc[11] * 100, 2)) + "%"
                tc[12] = str(round(tc[12] * 100, 2)) + "%"
                tc[13] = str(round(tc[13] * 100, 2)) + "%"
                tc[14] = str(round(tc[14], 2))
                tc[15] = str(round(tc[15], 2))
                tc[16] = str(round(tc[16] * 100, 2)) + "%"
                tc[17] = str(round(tc[17] * 100, 2)) + "%"
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


@do_on_dfs
class pure_moonnight(object):
    """封装选股框架"""

    __slots__ = ["shen"]

    def __init__(
        self,
        factors: pd.DataFrame,
        groups_num: int = 10,
        freq: str = "W",
        neutralize: bool = 0,
        trade_cost_double_side: float = 0,
        value_weighted: bool = 0,
        y2: bool = 0,
        plt_plot: bool = 1,
        plotly_plot: bool = 0,
        filename: str = "分组净值图",
        time_start: int = None,
        time_end: int = None,
        print_comments: bool = 1,
        comments_writer: pd.ExcelWriter = None,
        net_values_writer: pd.ExcelWriter = None,
        rets_writer: pd.ExcelWriter = None,
        comments_sheetname: str = None,
        net_values_sheetname: str = None,
        rets_sheetname: str = None,
        on_paper: bool = 0,
        sheetname: str = None,
        no_read_indu: bool = 0,
        only_cap: bool = 0,
        iplot: bool = 1,
        ilegend: bool = 0,
        without_breakpoint: bool = 0,
        total_cap: bool = 0,
    ) -> None:
        """一键回测框架，测试单因子的月频调仓的分组表现
        每月月底计算因子值，月初第一天开盘时买入，月末收盘最后一天收盘时卖出
        剔除上市不足60天的，停牌天数超过一半的，st天数超过一半的
        月末收盘跌停的不卖出，月初开盘涨停的不买入
        由最好组和最差组的多空组合构成多空对冲组

        Parameters
        ----------
        factors : pd.DataFrame
            要用于检测的因子值，index是时间，columns是股票代码
        groups_num : int, optional
            分组数量, by default 10
        freq : str, optional
            回测频率, by default 'M'
        neutralize : bool, optional
            对流通市值取自然对数，以完成行业市值中性化, by default 0
        trade_cost_double_side : float, optional
            交易的双边手续费率, by default 0
        value_weighted : bool, optional
            是否用流通市值加权, by default 0
        y2 : bool, optional
            画图时是否启用第二y轴, by default 0
        plt_plot : bool, optional
            将分组净值曲线用matplotlib画出来, by default 1
        plotly_plot : bool, optional
            将分组净值曲线用plotly画出来, by default 0
        filename : str, optional
            分组净值曲线的图保存的名称, by default "分组净值图"
        time_start : int, optional
            回测起始时间, by default None
        time_end : int, optional
            回测终止时间, by default None
        print_comments : bool, optional
            是否打印出评价指标, by default 1
        comments_writer : pd.ExcelWriter, optional
            用于记录评价指标的xlsx文件, by default None
        net_values_writer : pd.ExcelWriter, optional
            用于记录净值序列的xlsx文件, by default None
        rets_writer : pd.ExcelWriter, optional
            用于记录收益率序列的xlsx文件, by default None
        comments_sheetname : str, optional
            在记录评价指标的xlsx文件中，该工作表的名称, by default None
        net_values_sheetname : str, optional
            在记录净值序列的xlsx文件中，该工作表的名称, by default None
        rets_sheetname : str, optional
            在记录收益率序列的xlsx文件中，该工作表的名称, by default None
        on_paper : bool, optional
            使用学术化评价指标, by default 0
        sheetname : str, optional
            各个pd.Excelwriter中工作表的统一名称, by default None
        no_read_indu : bool, optional
            不读入行业数据, by default 0
        only_cap : bool, optional
            仅做市值中性化, by default 0
        iplot : bool, optional
            使用cufflinks呈现回测结果, by default 1
        ilegend : bool, optional
            使用cufflinks绘图时，是否显示图例, by default 1
        without_breakpoint : bool, optional
            画图的时候是否去除间断点, by default 0
        total_cap : bool, optional
            加权和行业市值中性化时使用总市值, by default 0
        """

        if not isinstance(factors, pd.DataFrame):
            factors = factors()
        if comments_writer is None and sheetname is not None:
            from pure_ocean_breeze.state.states import COMMENTS_WRITER

            comments_writer = COMMENTS_WRITER
        if net_values_writer is None and sheetname is not None:
            from pure_ocean_breeze.state.states import NET_VALUES_WRITER

            net_values_writer = NET_VALUES_WRITER
        if not on_paper:
            from pure_ocean_breeze.state.states import ON_PAPER

            on_paper = ON_PAPER
        if time_start is None:
            from pure_ocean_breeze.state.states import MOON_START

            if MOON_START is not None:
                factors = factors[factors.index >= pd.Timestamp(str(MOON_START))]
        else:
            factors = factors[factors.index >= pd.Timestamp(str(time_start))]
        if time_end is None:
            from pure_ocean_breeze.state.states import MOON_END

            if MOON_END is not None:
                factors = factors[factors.index <= pd.Timestamp(str(MOON_END))]
        else:
            factors = factors[factors.index <= pd.Timestamp(str(time_end))]
        if neutralize == 0:
            no_read_indu = 1
        if only_cap + no_read_indu > 0:
            only_cap = no_read_indu = 1
        if iplot:
            print_comments = 0
        if total_cap:
            if freq == "M":
                self.shen = pure_moon_b(
                    freq=freq,
                    no_read_indu=no_read_indu,
                )
            elif freq == "W":
                self.shen = pure_week_b(
                    freq=freq,
                    no_read_indu=no_read_indu,
                )
        else:
            if freq == "M":
                self.shen = pure_moon(
                    freq=freq,
                    no_read_indu=no_read_indu,
                )
            elif freq == "W":
                self.shen = pure_week(
                    freq=freq,
                    no_read_indu=no_read_indu,
                )
            elif freq == "D":
                self.shen = pure_moon(
                    freq=freq,
                    no_read_indu=no_read_indu,
                )
        self.shen.set_basic_data(
            total_cap=total_cap,
        )
        self.shen.set_factor_df_date_as_index(factors)
        self.shen.prerpare()
        self.shen.run(
            groups_num=groups_num,
            neutralize=neutralize,
            trade_cost_double_side=trade_cost_double_side,
            value_weighted=value_weighted,
            y2=y2,
            plt_plot=plt_plot,
            plotly_plot=plotly_plot,
            filename=filename,
            print_comments=print_comments,
            comments_writer=comments_writer,
            net_values_writer=net_values_writer,
            rets_writer=rets_writer,
            comments_sheetname=comments_sheetname,
            net_values_sheetname=net_values_sheetname,
            rets_sheetname=rets_sheetname,
            on_paper=on_paper,
            sheetname=sheetname,
            only_cap=only_cap,
            iplot=iplot,
            ilegend=ilegend,
            without_breakpoint=without_breakpoint,
        )

    def __call__(self) -> pd.DataFrame:
        """如果做了行业市值中性化，则返回行业市值中性化之后的因子数据

        Returns
        -------
        `pd.DataFrame`
            如果做了行业市值中性化，则行业市值中性化之后的因子数据，否则返回原因子数据
        """
        return self.shen.factors_out

    def comments_ten(self) -> pd.DataFrame:
        """对回测的十分组结果分别给出评价

        Returns
        -------
        `pd.DataFrame`
            评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率
        """
        rets_cols = list(self.shen.group_rets.columns)
        rets_cols = rets_cols[:-1]
        coms = []
        for i in rets_cols:
            ret = self.shen.group_rets[i]
            net = self.shen.group_net_values[i]
            com = comments_on_twins(net, ret)
            com = com.to_frame(i)
            coms.append(com)
        df = pd.concat(coms, axis=1)
        return df.T

    def comment_yearly(self) -> pd.DataFrame:
        """对回测的每年表现给出评价

        Returns
        -------
        pd.DataFrame
            各年度的收益率
        """
        df = self.shen.group_net_values.resample("Y").last().pct_change()
        df.index = df.index.year
        return df


class pure_week(pure_moon): ...


class pure_moon_a(pure_moon): ...


class pure_week_a(pure_moon): ...


class pure_moon_b(pure_moon): ...


class pure_week_b(pure_moon): ...


class pure_moon_c(pure_moon): ...


class pure_week_c(pure_moon): ...


class pure_fall(object):
    # DONE：修改为因子文件名可以带“日频_“，也可以不带“日频_“
    def __init__(self, daily_path: str) -> None:
        """一个使用mysql中的分钟数据，来更新因子值的框架

        Parameters
        ----------
        daily_path : str
            日频因子值存储文件的名字，请以'.parquet'结尾
        """
        self.homeplace = HomePlace()
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        self.daily_factors_path = self.homeplace.factor_data_file + "日频_" + daily_path

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

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

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df


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
        tqdm.auto.tqdm.pandas()
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


class pure_coldwinter(object):
    # DONE: 可以自由添加其他要剔除的因子，或者替换某些要剔除的因子

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(
        cls,
        momentum: bool = 1,
        earningsyield: bool = 1,
        growth: bool = 1,
        liquidity: bool = 1,
        size: bool = 1,
        leverage: bool = 1,
        beta: bool = 1,
        nonlinearsize: bool = 1,
        residualvolatility: bool = 1,
        booktoprice: bool = 1,
    ) -> None:
        """读入10种常用风格因子，并可以额外加入其他因子

        Parameters
        ----------
        facs_dict : Dict, optional
            额外加入的因子，名字为key，因子矩阵为value，形如`{'反转': ret20, '换手': tr20}`, by default None
        momentum : bool, optional
            是否删去动量因子, by default 1
        earningsyield : bool, optional
            是否删去盈利因子, by default 1
        growth : bool, optional
            是否删去成长因子, by default 1
        liquidity : bool, optional
            是否删去流动性因子, by default 1
        size : bool, optional
            是否删去规模因子, by default 1
        leverage : bool, optional
            是否删去杠杆因子, by default 1
        beta : bool, optional
            是否删去贝塔因子, by default 1
        nonlinearsize : bool, optional
            是否删去非线性市值因子, by default 1
        residualvolatility : bool, optional
            是否删去残差波动率因子, by default 1
        booktoprice : bool, optional
            是否删去账面市值比因子, by default 1
        """
        cls.homeplace = HomePlace()
        # barra因子数据
        styles = os.listdir(cls.homeplace.barra_data_file)
        styles = [i for i in styles if (i.endswith(".parquet")) and (i[0] != ".")]
        barras = {}
        for s in styles:
            k = s.split(".")[0]
            v = pd.read_parquet(cls.homeplace.barra_data_file + s).resample("W").last()
            barras[k] = v
        rename_dict = {
            "size": "市值",
            "nonlinearsize": "非线性市值",
            "booktoprice": "估值",
            "earningsyield": "盈利",
            "growth": "成长",
            "leverage": "杠杆",
            "liquidity": "流动性",
            "momentum": "动量",
            "residualvolatility": "波动率",
            "beta": "贝塔",
        }
        if momentum == 0:
            barras = {k: v for k, v in barras.items() if k != "momentum"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "momentum"}
        if earningsyield == 0:
            barras = {k: v for k, v in barras.items() if k != "earningsyield"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "earningsyield"}
        if growth == 0:
            barras = {k: v for k, v in barras.items() if k != "growth"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "growth"}
        if liquidity == 0:
            barras = {k: v for k, v in barras.items() if k != "liquidity"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "liquidity"}
        if size == 0:
            barras = {k: v for k, v in barras.items() if k != "size"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "size"}
        if leverage == 0:
            barras = {k: v for k, v in barras.items() if k != "leverage"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "leverage"}
        if beta == 0:
            barras = {k: v for k, v in barras.items() if k != "beta"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "beta"}
        if nonlinearsize == 0:
            barras = {k: v for k, v in barras.items() if k != "nonlinearsize"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "nonlinearsize"}
        if residualvolatility == 0:
            barras = {k: v for k, v in barras.items() if k != "residualvolatility"}
            rename_dict = {
                k: v for k, v in rename_dict.items() if k != "residualvolatility"
            }
        if booktoprice == 0:
            barras = {k: v for k, v in barras.items() if k != "booktoprice"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "booktoprice"}
        facs_dict = {
            "反转_5天收益率均值": boom_one(read_daily(ret=1)),
            "波动_20天收益率标准差": read_daily(ret=1)
            .rolling(20, min_periods=10)
            .std()
            .resample("W")
            .last(),
            "换手_5天换手率均值": boom_one(read_daily(tr=1)),
        }
        barras.update(facs_dict)
        rename_dict.update({k: k for k in facs_dict.keys()})
        cls.barras = barras
        cls.rename_dict = rename_dict
        sort_names = list(rename_dict.values())
        cls.sort_names = sort_names
        cls.barras_together = merge_many(
            list(barras.values()), list(barras.keys()), how="inner"
        )

    def __call__(self):
        """返回纯净因子值"""
        return self.snow_fac

    def set_factors_df_wide(self, df: pd.DataFrame, other_factors: dict = None):
        """传入因子数据，时间为索引，代码为列名"""
        df = df.resample("W").last()
        self.__corr = [
            df.corrwith(i, axis=1).mean() for i in list(self.barras.values())
        ]
        self.__corr = (
            pd.Series(
                self.__corr, index=[self.rename_dict[i] for i in self.barras.keys()]
            )
            .to_frame("相关系数")
            .T
        )
        self.__corr = self.__corr[self.sort_names]
        df = df.stack().reset_index()
        df.columns = ["date", "code", "fac"]
        self.factors = df
        self.corr_pri = pd.merge(df, self.barras_together, on=["date", "code"]).dropna()
        if other_factors is not None:
            other_factors = merge_many(
                list(other_factors.values()), list(other_factors.keys()), how="inner"
            )
            self.corr_pri = pd.merge(self.corr_pri, other_factors, on=["date", "code"])

    @property
    def corr(self) -> pd.DataFrame:
        """因子和10个常用风格因子的相关系数

        Returns
        -------
        pd.DataFrame
            因子和10个常用风格因子的相关系数
        """
        return self.__corr.copy()

    def ols_in_group(self, date):
        """对每个时间段进行回归，并计算残差"""
        df=self.corr_pri[self.corr_pri.date==date].set_index(['date','code'])
        xs = list(df.columns)
        xs = [i for i in xs if i != "fac"]
        xs_join = "+".join(xs)
        ols_formula = "fac~" + xs_join
        ols_result = smf.ols(ols_formula, data=df).fit()
        df.fac = ols_result.resid
        return df[['fac']]

    def get_snow_fac(self):
        """获得纯净因子"""
        dates=list(set(self.corr_pri.date))
        # dates=[self.corr_pri[self.corr_pri.date==i].set_index(['date','code']) for i in dates]
        # print(dates[0])
        with mpire.WorkerPool(20) as pool:
            res=pool.map(self.ols_in_group,dates)
        self.snow_fac = pd.concat(res)
        # self.snow_fac = (
        #     self.corr_pri.set_index(["date", "code"])
        #     .groupby(["date"],as_index=False)
        #     .apply(self.ols_in_group)
        # )
        if 'date' in self.snow_fac.columns:
            self.snow_fac=self.snow_fac.rename(columns={'date':'old_date'})
        # self.snow_fac = self.snow_fac.unstack()
        # self.snow_fac.columns = list(map(lambda x: x[1], list(self.snow_fac.columns)))
        self.snow_fac=self.snow_fac.reset_index().pivot(index='date',columns='code',values='fac')

def ols_in_group(df):
    """对每个时间段进行回归，并计算残差"""
    xs = list(df.columns)
    xs = [i for i in xs if i != "fac"]
    xs_join = "+".join(xs)
    ols_formula = "fac~" + xs_join
    ols_result = smf.ols(ols_formula, data=df).fit()
    df.fac = ols_result.resid
    return df[['fac']]

@do_on_dfs
class pure_snowtrain(object):
    """直接返回纯净因子"""

    def __init__(
        self,
        factors: pd.DataFrame,
        facs_dict: Dict = None,
        momentum: bool = 1,
        earningsyield: bool = 1,
        growth: bool = 1,
        liquidity: bool = 1,
        size: bool = 1,
        leverage: bool = 1,
        beta: bool = 1,
        nonlinearsize: bool = 1,
        residualvolatility: bool = 1,
        booktoprice: bool = 1,
    ) -> None:
        """计算因子值与10种常用风格因子之间的相关性，并进行纯净化，可以额外加入其他因子

        Parameters
        ----------
        factors : pd.DataFrame
            要考察的因子值，index为时间，columns为股票代码，values为因子值
        facs_dict : Dict, optional
            额外加入的因子，名字为key，因子矩阵为value，形如`{'反转': ret20, '换手': tr20}`, by default None
        momentum : bool, optional
            是否删去动量因子, by default 1
        earningsyield : bool, optional
            是否删去盈利因子, by default 1
        growth : bool, optional
            是否删去成长因子, by default 1
        liquidity : bool, optional
            是否删去流动性因子, by default 1
        size : bool, optional
            是否删去规模因子, by default 1
        leverage : bool, optional
            是否删去杠杆因子, by default 1
        beta : bool, optional
            是否删去贝塔因子, by default 1
        nonlinearsize : bool, optional
            是否删去非线性市值因子, by default 1
        residualvolatility : bool, optional
            是否删去残差波动率因子, by default 1
        booktoprice : bool, optional
            是否删去账面市值比因子, by default 1
        """
        self.winter = pure_coldwinter(
            momentum=momentum,
            earningsyield=earningsyield,
            growth=growth,
            liquidity=liquidity,
            size=size,
            leverage=leverage,
            beta=beta,
            nonlinearsize=nonlinearsize,
            residualvolatility=residualvolatility,
            booktoprice=booktoprice,
        )
        self.winter.set_factors_df_wide(factors, facs_dict)
        self.winter.get_snow_fac()
        self.corr = self.winter.corr

    def __call__(self) -> pd.DataFrame:
        """获得纯净化之后的因子值

        Returns
        -------
        pd.DataFrame
            纯净化之后的因子值
        """
        return self.winter.snow_fac.copy()

    def show_corr(self) -> pd.DataFrame:
        """展示因子与barra风格因子的相关系数

        Returns
        -------
        pd.DataFrame
            相关系数表格
        """
        return self.corr.applymap(lambda x: to_percent(x))


class pure_newyear(object):
    """转为生成25分组和百分组的收益矩阵而封装"""

    def __init__(
        self,
        facx: pd.DataFrame,
        facy: pd.DataFrame,
        group_num_single: int,
        trade_cost_double_side: float = 0,
        namex: str = "主",
        namey: str = "次",
    ) -> None:
        """条件双变量排序法，先对所有股票，依照因子facx进行排序
        然后在每个组内，依照facy进行排序，最后统计各个组内的平均收益率

        Parameters
        ----------
        facx : pd.DataFrame
            首先进行排序的因子，通常为控制变量，相当于正交化中的自变量
            index为时间，columns为股票代码，values为因子值
        facy : pd.DataFrame
            在facx的各个组内，依照facy进行排序，为主要要研究的因子
            index为时间，columns为股票代码，values为因子值
        group_num_single : int
            单个因子分成几组，通常为5或10
        trade_cost_double_side : float, optional
            交易的双边手续费率, by default 0
        namex : str, optional
            facx这一因子的名字, by default "主"
        namey : str, optional
            facy这一因子的名字, by default "次"
        """
        homex = pure_fallmount(facx)
        homey = pure_fallmount(facy)
        if group_num_single == 5:
            homexy = homex > homey
        elif group_num_single == 10:
            homexy = homex >> homey
        shen = pure_moonnight(
            homexy(),
            group_num_single**2,
            trade_cost_double_side=trade_cost_double_side,
            plt_plot=False,
            print_comments=False,
        )
        sq = shen.shen.square_rets.copy()
        sq.index = [namex + str(i) for i in list(sq.index)]
        sq.columns = [namey + str(i) for i in list(sq.columns)]
        self.square_rets = sq

    def __call__(self) -> pd.DataFrame:
        """调用对象时，返回最终结果，正方形的分组年化收益率表

        Returns
        -------
        `pd.DataFrame`
            每个组的年化收益率
        """
        return self.square_rets.copy()


class pure_helper(object):
    def __init__(
        self,
        df_main: pd.DataFrame,
        df_helper: pd.DataFrame,
        func: Callable = None,
        group: int = 10,
    ) -> None:
        """使用因子b的值大小，对因子a进行分组，并可以在组内进行某种操作

        Parameters
        ----------
        df_main : pd.DataFrame
            要被分组并进行操作的因子
        df_helper : pd.DataFrame
            用来做分组的依据
        func : Callable, optional
            分组后，组内要进行的操作, by default None
        group : int, optional
            要分的组数, by default 10
        """
        self.df_main = df_main
        self.df_helper = df_helper
        self.func = func
        self.group = group
        if self.func is None:
            self.__data = self.sort_a_with_b()
        else:
            self.__data = self.sort_a_with_b_func()

    @property
    def data(self):
        return self.__data

    def __call__(self) -> pd.DataFrame:
        return self.data

    def sort_a_with_b(self):
        dfb = to_group(self.df_helper, group=self.group)
        dfb = dfb.stack().reset_index()
        dfb.columns = ["date", "code", "group"]
        dfa = self.df_main.stack().reset_index()
        dfa.columns = ["date", "code", "target"]
        df = pd.merge(dfa, dfb, on=["date", "code"])
        return df

    def sort_a_with_b_func(self):
        the_func = partial(self.func)
        df = self.sort_a_with_b().drop(columns=["code"])
        df = (
            df.groupby(["date", "group"])
            .apply(the_func)
            .drop(columns=["group"])
            .reset_index()
        )
        df = df.pivot(index="date", columns="group", values="target")
        df.columns = [f"group{str(int(i+1))}" for i in list(df.columns)]
        return df


@do_on_dfs
def get_group(df: pd.DataFrame, group_num: int = 10) -> pd.DataFrame:
    """使用groupby的方法，将一组因子值改为截面上的分组值，此方法相比qcut的方法更加稳健，但速度更慢一些

    Parameters
    ----------
    df : pd.DataFrame
        因子值，index为时间，columns为股票代码，values为因子值
    group_num : int, optional
        分组的数量, by default 10

    Returns
    -------
    pd.DataFrame
        转化为分组值后的df，index为时间，columns为股票代码，values为分组值
    """
    a = pure_moon(no_read_indu=1)
    df = df.stack().reset_index()
    df.columns = ["date", "code", "fac"]
    df = a.get_groups(df, group_num).pivot(index="date", columns="code", values="group")
    return df


def symmetrically_orthogonalize(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """对多个因子做对称正交，每个因子得到正交其他因子后的结果

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        多个要做正交的因子，每个df都是index为时间，columns为股票代码，values为因子值的df

    Returns
    -------
    list[pd.DataFrame]
        对称正交后的各个因子
    """

    def sing(dfs: list[pd.DataFrame], date: pd.Timestamp):
        dds = []
        for num, i in enumerate(dfs):
            i = i[i.index == date]
            i.index = [f"fac{num}"]
            i = i.T
            dds.append(i)
        dds = pd.concat(dds, axis=1)
        cov = dds.cov()
        d, u = np.linalg.eig(cov)
        d = np.diag(d ** (-0.5))
        new_facs = pd.DataFrame(
            np.dot(dds, np.dot(np.dot(u, d), u.T)), columns=dds.columns, index=dds.index
        )
        new_facs = new_facs.stack().reset_index()
        new_facs.columns = ["code", "fac_number", "fac"]
        new_facs = new_facs.assign(date=date)
        dds = []
        for num, i in enumerate(dfs):
            i = new_facs[new_facs.fac_number == f"fac{num}"]
            i = i.pivot(index="date", columns="code", values="fac")
            dds.append(i)
        return dds

    dfs = [standardlize(i) for i in dfs]
    date_first = max([i.index.min() for i in dfs])
    date_last = min([i.index.max() for i in dfs])
    dfs = [i[(i.index >= date_first) & (i.index <= date_last)] for i in dfs]
    fac_num = len(dfs)
    ddss = [[] for i in range(fac_num)]
    for date in tqdm.auto.tqdm(dfs[0].index):
        dds = sing(dfs, date)
        for num, i in enumerate(dds):
            ddss[num].append(i)
    ds = []
    for i in tqdm.auto.tqdm(ddss):
        ds.append(pd.concat(i))
    return ds


@do_on_dfs
def sun(factor:pd.DataFrame,rolling_5:int=1):
    '''先单因子测试，再测试其与常用风格之间的关系'''
    if rolling_5:
        factor=boom_one(factor)
    shen=pure_moonnight(factor)
    pfi=pure_snowtrain(factor)
    shen=pure_moonnight(pfi,neutralize=1)
    display(pfi.show_corr())