__updated__ = "2022-09-13 16:44:33"

import numpy as np
import pandas as pd
import knockknock as kk
import os
import tqdm
import scipy.stats as ss
import scipy.io as scio
import statsmodels.formula.api as smf
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False
from functools import reduce, lru_cache, partial
from dateutil.relativedelta import relativedelta
from loguru import logger
import datetime
from collections import Iterable
import plotly.express as pe
import plotly.io as pio
from typing import Callable, Union
from pure_ocean_breeze.legacy_version.v3p1.data.read_data import read_daily, get_industry_dummies
from pure_ocean_breeze.legacy_version.v3p1.state.homeplace import HomePlace

homeplace = HomePlace()
from pure_ocean_breeze.legacy_version.v3p1.state.decorators import *
from pure_ocean_breeze.legacy_version.v3p1.state.states import STATES
from pure_ocean_breeze.legacy_version.v3p1.data.database import *
from pure_ocean_breeze.legacy_version.v3p1.data.dicts import INDUS_DICT
from pure_ocean_breeze.legacy_version.v3p1.data.tools import indus_name


def daily_factor_on300500(
    fac: pd.DataFrame,
    hs300: bool = 0,
    zz500: bool = 0,
    zz800: bool = 0,
    zz1000: bool = 0,
    gz2000: bool = 0,
    other: bool = 0,
) -> pd.DataFrame:
    """输入日频或月频因子值，将其限定在某指数成分股的股票池内，
    目前仅支持沪深300、中证500、中证800、中证1000、国证2000成分股，和除沪深300、中证500、中证1000以外的股票的成分股

    Parameters
    ----------
    fac : pd.DataFrame
        未限定股票池的因子值，index为时间，columns为股票代码
    hs300 : bool, optional
        限定股票池为沪深300, by default 0
    zz500 : bool, optional
        限定股票池为中证500, by default 0
    zz800 : bool, optional
        限定股票池为中证800, by default 0
    zz1000 : bool, optional
        限定股票池为中证1000, by default 0
    gz2000 : bool, optional
        限定股票池为国证2000, by default 0
    other : bool, optional
        限定股票池为除沪深300、中证500、中证1000以外的股票的成分股, by default 0

    Returns
    -------
    `pd.DataFrame`
        仅包含成分股后的因子值，非成分股的因子值为空

    Raises
    ------
    `ValueError`
        如果未指定任何一种指数的成分股，将报错
    """
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


def daily_factor_on_swindustry(df: pd.DataFrame) -> dict:
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
    df1 = df.resample("M").last()
    if df1.shape[0] * 2 > df.shape[0]:
        daily = 0
        monthly = 1
    else:
        daily = 1
        monthly = 0
    start = int(datetime.datetime.strftime(df.index.min(), "%Y%m%d"))
    ress = get_industry_dummies(daily=daily, monthly=monthly, start=start)
    ress = {k: v * df for k, v in ress.items()}
    return ress


def group_test_on_swindustry(
    df: pd.DataFrame, group_num: int = 10, net_values_writer: pd.ExcelWriter = None
) -> pd.DataFrame:
    """在申万一级行业上测试每个行业的分组回测

    Parameters
    ----------
    df : pd.DataFrame
        全市场的因子值，index是时间，columns是股票代码
    group_num : int, optional
        分组数量, by default 10
    net_values_writer : pd.ExcelWriter, optional
        用于存储各个行业分组及多空对冲净值序列的excel文件, by default None

    Returns
    -------
    pd.DataFrame
        各个行业的绩效评价汇总
    """
    dfs = daily_factor_on_swindustry(df)
    ks = []
    vs = []
    for k, v in dfs.items():
        shen = pure_moonnight(
            v,
            groups_num=group_num,
            net_values_writer=net_values_writer,
            sheetname=INDUS_DICT[k],
            plt_plot=0,
        )
        ks.append(k)
        vs.append(shen.shen.total_comments.T)
    vs = pd.concat(vs)
    vs.index = [INDUS_DICT[i] for i in ks]
    return vs


def rankic_test_on_swindustry(
    df: pd.DataFrame, excel_name: str = "行业rankic.xlsx", png_name: str = "行业rankic图.png"
) -> pd.DataFrame:
    """专门计算因子值在各个申万一级行业上的Rank IC值，并绘制柱状图

    Parameters
    ----------
    df : pd.DataFrame
        全市场的因子值，index是时间，columns是股票代码
    excel_name : str, optional
        用于保存各个行业Rank IC值的excel文件的名字, by default '行业rankic.xlsx'
    png_name : str, optional
        用于保存各个行业Rank IC值的柱状图的名字, by default '行业rankic图.png'

    Returns
    -------
    pd.DataFrame
        行业名称与对应的Rank IC
    """
    vs = group_test_on_swindustry(df)
    rankics = vs[["RankIC"]].T
    rankics.to_excel(excel_name)
    rankics.plot(kind="bar")
    plt.show()
    plt.savefig(png_name)
    return rankics


def long_test_on_swindustry(
    df: pd.DataFrame,
    nums: list,
    pos: bool = 0,
    neg: bool = 0,
    save_stock_list: bool = 0,
) -> list[dict]:
    """对每个申万一级行业成分股，使用某因子挑选出最多头的n值股票，考察其超额收益绩效、每月超额收益、每月每个行业的多头名单

    Parameters
    ----------
    df : pd.DataFrame
        使用的因子，index为时间，columns为股票代码
    nums : list
        多头想选取的股票的数量，例如[3,4,5]
    pos : bool, optional
        因子方向为正，即Rank IC为正，则指定此处为True, by default 0
    neg : bool, optional
        因子方向为负，即Rank IC为负，则指定此处为False, by default 0
    save_stock_list:bool, optional
        是否保存每月每个行业的多头名单，会降低运行速度, by default 0

    Returns
    -------
    list[dict]
        超额收益绩效、每月超额收益、每月每个行业的多头名单

    Raises
    ------
    IOError
        pos和neg必须有一个为1，否则将报错
    """
    fac = decap_industry(fac, monthly=True)
    industry_dummy = pd.read_feather(
        homeplace.daily_data_file + "申万行业2021版哑变量.feather"
    ).fillna(0)
    inds = list(industry_dummy.columns)
    ret_next = (
        read_daily(close=1).resample("M").last()
        / read_daily(open=1).resample("M").first()
        - 1
    )
    ages = read_daily(age=1)
    ages = (ages >= 60) + 0
    ages = ages.replace(0, np.nan)
    ret_next = ret_next * ages
    ret_next_dummy = 1 - ret_next.isna()

    def save_ind(code, num):
        ind = industry_dummy[["date", "code", code]]
        ind = ind.pivot(index="date", columns="code", values=code)
        ind = ind.resample("M").last()
        ind = ind.replace(0, np.nan)
        fi = ind * fac
        fi = fi.dropna(how="all")
        fi = fi.shift(1)
        fi = fi * ret_next_dummy
        fi = fi.dropna(how="all")

        def sing(x):
            if neg:
                thr = x.nsmallest(num).iloc[-1]
            elif pos:
                thr = x.nlargest(num).iloc[-1]
            else:
                raise IOError("您需要指定一下因子方向🤒")
            x = (x <= thr) + 0
            return x

        fi = fi.T.apply(sing).T
        fi = fi.replace(0, np.nan)
        fi = fi * ret_next
        ret_long = fi.mean(axis=1)
        return ret_long

    ret_longs = {k: {} for k in nums}
    for num in tqdm.tqdm(nums):
        for code in inds[2:]:
            ret_longs[num][code] = save_ind(code, num)

    coms = {
        k: indus_name(pd.concat(v, axis=1).dropna(how="all").T).T
        for k, v in ret_longs.items()
    }
    indus = indus.resample("M").last().pct_change()
    rets = {k: (v - indus_name(indus.T).T).dropna(how="all") for k, v in coms.items()}
    nets = {k: (v + 1).cumprod() for k, v in rets.items()}
    nets = {
        k: v.apply(lambda x: x.dropna() / x.dropna().iloc[0]) for k, v in nets.items()
    }

    def comments_on_twins(nets: pd.Series, rets: pd.Series) -> pd.Series:
        series = nets.copy()
        series1 = rets.copy()
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

    w = pd.ExcelWriter("各个申万一级行业多头超额绩效.xlsx")

    def com_all(df1, df2, num):
        cs = []
        for ind in list(df1.columns):
            c = comments_on_twins(df2[ind].dropna(), df1[ind].dropna()).to_frame(ind)
            cs.append(c)
        res = pd.concat(cs, axis=1).T
        res.to_excel(w, sheet_name=str(num))
        return res

    coms_finals = {k: com_all(rets[k], nets[k], k) for k in rets.keys()}
    w.save()
    w.close()

    rets_save = {k: v.dropna() for k, v in rets.items() if k in nums}
    u = pd.ExcelWriter("各个申万一级行业每月超额收益率.xlsx")
    for k, v in rets_save.items():
        v.to_excel(u, sheet_name=str(k))
    u.save()
    u.close()

    if save_stock_list:

        def save_ind_stocks(code, num):
            ind = industry_dummy[["date", "code", code]]
            ind = ind.pivot(index="date", columns="code", values=code)
            ind = ind.replace(0, np.nan)
            fi = ind * fac
            fi = fi.dropna(how="all")
            fi = fi.shift(1)
            fi = fi * ret_next_dummy
            fi = fi.dropna(how="all")

            def sing(x):
                if neg:
                    thr = x.nsmallest(num)
                elif pos:
                    thr = x.nlargest(num)
                else:
                    raise IOError("您需要指定一下因子方向🤒")
                return tuple(thr.index)

            fi = fi.T.apply(sing)
            return fi

        stocks_longs = {k: {} for k in nums}
        for num in tqdm.tqdm(nums):
            for code in inds[2:]:
                stocks_longs[num][code] = save_ind_stocks(code, num)

        for num in nums:
            w1 = pd.ExcelWriter(f"各个申万一级行业买{num}只的股票名单.xlsx")
            for k, v in stocks_longs[num].items():
                v = v.T
                v.index = v.index.strftime("%Y/%m/%d")
                v.to_excel(w1, sheet_name=INDUS_DICT[k])
            w1.save()
            w1.close()

        return [coms_finals, rets_save, stocks_longs]
    else:
        return [coms_finals, rets_save]


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


@kk.desktop_sender(title="嘿，行业中性化做完啦～🛁")
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
def decap_industry(
    df: pd.DataFrame, daily: bool = 0, monthly: bool = 0
) -> pd.DataFrame:
    """对因子做行业市值中性化

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
        行业市值中性化之后的因子

    Raises
    ------
    `NotImplementedError`
        如果未指定日频或月频，将报错
    """
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
    if daily == 0 and monthly == 0:
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
    else:
        raise NotImplementedError("必须指定频率")
    industry_dummy.columns = col
    df = pd.merge(df, industry_dummy, on=["date", "code"])
    df = df.set_index(["date", "code"])
    tqdm.tqdm.pandas()
    df = df.groupby(["date"]).progress_apply(neutralize_factors)
    df = df.unstack()
    df.columns = [i[1] for i in list(df.columns)]
    return df


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
    x = np.sum(df.to_numpy().flatten())
    if np.isnan(x):
        print("存在空值")
        return True
    else:
        print("不存在空值")
        return False


def boom_four(
    df: pd.DataFrame, backsee: int = 20, daily: bool = 0, min_periods: int = None
) -> tuple[pd.DataFrame]:
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
    `tuple[pd.DataFrame]`
        6个因子的元组
    """
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


def get_abs(df: pd.DataFrame, median: bool = 0, square: bool = 0) -> pd.DataFrame:
    """均值距离化：计算因子与截面均值的距离

    Parameters
    ----------
    df : pd.DataFrame
        未均值距离化的因子，index为时间，columns为股票代码
    median : bool, optional
        为1则计算到中位数的距离, by default 0
    square : bool, optional
        为1则计算距离的平方, by default 0

    Returns
    -------
    `pd.DataFrame`
        _description_
    """
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
    trs = read_daily(tr=1)
    trs = trs.assign(tradeends=list(trs.index))
    trs = trs[["tradeends"]]
    trs = trs.resample("M").last()
    df = pd.concat([trs, df], axis=1)
    df = df.set_index(["tradeends"])
    return df


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


def show_corr(
    fac1: pd.DataFrame, fac2: pd.DataFrame, method: str = "spearman", plt_plot: bool = 1
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

    Returns
    -------
    `float`
        平均截面相关系数
    """
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
    factors: list[pd.DataFrame],
    factor_names: list[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
) -> pd.DataFrame:
    """展示很多因子两两之间的截面相关性

    Parameters
    ----------
    factors : list[pd.DataFrame]
        所有因子构成的列表, by default None
    factor_names : list[str], optional
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


def get_list_std(delta_sts: list[pd.DataFrame]) -> pd.DataFrame:
    """同一天多个因子，计算这些因子在当天的标准差

    Parameters
    ----------
    delta_sts : list[pd.DataFrame]
        多个因子构成的list，每个因子index为时间，columns为股票代码

    Returns
    -------
    `pd.DataFrame`
        每天每只股票多个因子的标准差
    """
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum(delta_sts_std)
    delta_sts_std = delta_sts_std**0.5 / len(delta_sts)
    return delta_sts_std


class pure_moon(object):
    __slots__ = [
        "homeplace" "path_prefix",
        "codes_path",
        "tradedays_path",
        "ages_path",
        "sts_path",
        "states_path",
        "opens_path",
        "closes_path",
        # "highs_path",
        # "lows_path",
        "pricloses_path",
        "flowshares_path",
        # "amounts_path",
        # "turnovers_path",
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
        # cls.highs_path = "Allstock_DailyHigh_dividend.mat"
        # 复权最低价数据文件
        # cls.lows_path = "Allstock_DailyLow_dividend.mat"
        # 不复权收盘价数据文件
        cls.pricloses_path = "AllStock_DailyClose.mat"
        # 流通股本数据文件
        cls.flowshares_path = "AllStock_DailyAShareNum.mat"
        # 成交量数据文件
        # cls.amounts_path = "AllStock_DailyVolume.mat"
        # 换手率数据文件
        # cls.turnovers_path = "AllStock_DailyTR.mat"
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
        cls.industry_dummy = (
            pd.read_feather(cls.homeplace.daily_data_file + "申万行业2021版哑变量.feather")
            .set_index("date")
            .groupby("code")
            .resample("M")
            .last()
        )
        cls.industry_dummy = cls.industry_dummy.drop(columns=["code"]).reset_index()
        cls.industry_ws = [f"w{i}" for i in range(1, cls.industry_dummy.shape[1] - 1)]
        col = ["code", "date"] + cls.industry_ws
        cls.industry_dummy.columns = col
        cls.industry_dummy = cls.industry_dummy[
            cls.industry_dummy.date >= pd.Timestamp("2010-01-01")
        ]

    def __call__(self, fallmount=0):
        """调用对象则返回因子值"""
        df = self.factors_out.copy()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        if fallmount == 0:
            return df
        else:
            return pure_fallmount(df)

    @params_setter(slogan=None)
    def set_factor_file(self, factors_file):
        """设置因子文件的路径，因子文件列名应为股票代码，索引为时间"""
        self.factors_file = factors_file
        self.factors = pd.read_feather(self.factors_file)
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

    @params_setter(slogan=None)
    def set_factor_df_date_as_index(self, df):
        """设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间"""
        df = df.reset_index()
        df.columns = ["date"] + list(df.columns)[1:]
        self.factors = df
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

    @params_setter(slogan=None)
    def set_factor_df_wide(self, df):
        """从dataframe读入因子宽数据"""
        if isinstance(df, pure_fallmount):
            df = df()
        self.factors = df.copy()
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

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
        cls.codes = list(map(lambda x: x[0], cls.codes))
        cls.tradedays = cls.tradedays[0].tolist()

    @classmethod
    @tool_box(slogan=None)
    def loadmat(cls, path):
        """重写一个加载mat文件的函数，以使代码更简洁"""
        return list(scio.loadmat(path).values())[3]

    @classmethod
    @tool_box(slogan=None)
    def make_df(cls, data):
        """将读入的数据，和股票代码与时间拼接，做成dataframe"""
        data = pd.DataFrame(data, columns=cls.codes, index=cls.tradedays)
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
        data = data[data.index >= pd.Timestamp("2010-01-01")]
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
    @tool_box(slogan=None)
    def judge_month_st_by10(cls, df):
        """比较一个月内正常交易的天数，如果少于10天，就删除本月"""
        normal_count = len(df[df != 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    @classmethod
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
    @tool_box(slogan=None)
    def judge_month_state_by10(cls, df):
        """比较一个月内正常交易天数，如果少于10天，就删除本月"""
        normal_count = len(df[df == 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    @classmethod
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
    @tool_box(slogan=None)
    def write_feather(cls, df, path):
        """将算出来的数据存入本地，以免造成重复运算"""
        df1 = df.copy()
        df1 = df1.reset_index()
        df1.to_feather(path)

    @classmethod
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
        self.factors = pd.merge(self.factors, self.industry_dummy, on=["date", "code"])
        self.factors = self.factors.set_index(["date", "code"])
        self.factors = self.factors.groupby(["date"]).apply(self.neutralize_factors)
        self.factors = self.factors.reset_index()

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
    @tool_box(slogan=None)
    def get_ic_rankic(cls, df):
        """计算IC和RankIC"""
        df1 = df[["ret", "fac"]]
        ic = df1.corr(method="pearson").iloc[0, 1]
        rankic = df1.corr(method="spearman").iloc[0, 1]
        df2 = pd.DataFrame({"ic": [ic], "rankic": [rankic]})
        return df2

    @classmethod
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
    @tool_box(slogan=None)
    def next_month_end(cls, x):
        """找到下个月最后一天"""
        x1 = x = x + relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @classmethod
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

    @main_process(slogan=None)
    def select_data_time(self, time_start, time_end):
        """筛选特定的时间段"""
        if time_start:
            self.data = self.data[self.data.date >= time_start]
        if time_end:
            self.data = self.data[self.data.date <= time_end]

    @tool_box(slogan=None)
    def make_start_to_one(self, l):
        """让净值序列的第一个数变成1"""
        min_date = self.factors.date.min()
        add_date = min_date - relativedelta(days=min_date.day)
        add_l = pd.Series([1], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    @tool_box(slogan=None)
    def to_group_ret(self, l):
        """每一组的年化收益率"""
        ret = l[-1] ** (12 / len(l)) - 1
        return ret

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

    @main_process(slogan=None)
    def get_total_comments(self):
        """综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        self.total_comments = pd.concat(
            [self.ic_icir_and_rank, self.long_short_comments]
        )

    @main_process(slogan=None)
    def plot_net_values(self, y2, filename):
        """使用matplotlib来画图，y2为是否对多空组合采用双y轴"""
        self.group_net_values.plot(secondary_y=y2, rot=60)
        filename_path = filename + ".png"
        if not STATES["NO_SAVE"]:
            plt.savefig(filename_path)

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


class pure_moonnight(object):
    """封装选股框架"""

    __slots__ = ["shen"]

    def __init__(
        self,
        factors: pd.DataFrame,
        groups_num: int = 10,
        neutralize: bool = 0,
        boxcox: bool = 1,
        by10: bool = 0,
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
        neutralize : bool, optional
            对流通市值取自然对数，以完成行业市值中性化, by default 0
        boxcox : bool, optional
            对流通市值做截面boxcox变换，以完成行业市值中性化, by default 1
        by10 : bool, optional
            每天st和停牌状态月度化时，以10天作为标准, by default 0
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
            回测起始时间（此参数已废弃，请在因子上直接截断）, by default None
        time_end : int, optional
            回测终止时间（此参数已废弃，请在因子上直接截断）, by default None
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
        """

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

    def __call__(self) -> pd.DataFrame:
        """如果做了行业市值中性化，则返回行业市值中性化之后的因子数据

        Returns
        -------
        `pd.DataFrame`
            如果做了行业市值中性化，则行业市值中性化之后的因子数据，否则返回原因子数据
        """
        df = self.shen.factors_out.copy()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        return df


class pure_fall(object):
    # DONE：修改为因子文件名可以带“日频_“，也可以不带“日频_“
    def __init__(self, daily_path: str) -> None:
        """一个使用mysql中的分钟数据，来更新因子值的框架

        Parameters
        ----------
        daily_path : str
            日频因子值存储文件的名字，请以'.feather'结尾
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

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    def get_single_day_factor(self, func: Callable, day: int) -> pd.DataFrame:
        """计算单日的因子值，通过sql数据库，读取单日的数据，然后计算因子值"""
        sql = sqlConfig("minute_data_stock_alter")
        df = sql.get_data(str(day))
        the_func = partial(func)
        df = df.groupby(["code"]).apply(the_func).to_frame()
        df.columns = [str(day)]
        df = df.T
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        return df

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors_alter(self, func: Callable) -> None:
        """用mysql逐日更新分钟数据构造的因子

        Parameters
        ----------
        func : Callable
            构造分钟数据使用的函数

        Raises
        ------
        `IOError`
            如果没有历史因子数据，将报错
        """
        """通过minute_data_stock_alter数据库一天一天计算因子值"""
        try:
            try:
                self.daily_factors = pd.read_feather(self.daily_factors_path)
            except Exception:
                self.daily_factors_path = self.daily_factors_path.split("日频_")
                self.daily_factors_path = (
                    self.daily_factors_path[0] + self.daily_factors_path[1]
                )
                self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.drop_duplicates(subset=['date'],keep='last')
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
            raise IOError(
                "您还没有该因子的初级数据，暂时不能更新。请先使用pure_fall_frequent或pure_fall_flexible计算历史因子值。"
            )


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


class pure_fall_frequent(object):
    """对单只股票单日进行操作"""

    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
        clickhouse: bool = 0,
        postgresql: bool = 0,
        questdb: bool = 0,
    ) -> None:
        """基于clickhouse的分钟数据，计算因子值，每天的因子值只用到当日的数据

        Parameters
        ----------
        factor_file : str
            用于保存因子值的文件名，需为feather文件，以'.feather'结尾
        startdate : int, optional
            起始时间，形如20121231，为开区间, by default None
        enddate : int, optional
            截止时间，形如20220814，为闭区间，为空则计算到最近数据, by default None
        kind : str, optional
            类型为股票还是指数，指数为'index', by default "stock"
        clickhouse : bool, optional
            使用clickhouse作为数据源，如果postgresql与本参数都为0，将依然从clickhouse中读取, by default 0
        postgresql : bool, optional
            使用postgresql作为数据源, by default 0
        questdb : bool, optional
            使用questdb作为数据源, by default 0
        """
        homeplace = HomePlace()
        self.kind = kind
        if clickhouse == 0 and postgresql == 0 and questdb == 0:
            clickhouse = 1
        self.clickhouse = clickhouse
        self.postgresql = postgresql
        self.questdb = questdb
        if clickhouse == 1:
            # 连接clickhouse
            self.chc = ClickHouseClient("minute_data")
        elif postgresql == 1:
            self.chc = PostgreSQL("minute_data")
        else:
            self.chc = Questdb()
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = pd.read_feather(self.factor_file)
            factor_old.columns = ["date"] + list(factor_old.columns)[1:]
            factor_old = factor_old.drop_duplicates(subset=["date"])
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

    def __call__(self) -> pd.DataFrame:
        """获得经运算产生的因子

        Returns
        -------
        `pd.DataFrame`
            经运算产生的因子值
        """
        return self.factor.copy()

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func: Callable,
        fields: str = "*",
        chunksize: int = 10,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        """每次抽取chunksize天的截面上全部股票的分钟数据
        对每天的股票的数据计算因子值

        Parameters
        ----------
        func : Callable
            用于计算因子值的函数
        fields : str, optional
            股票数据涉及到哪些字段，排除不必要的字段，可以节约读取数据的时间，形如'date,code,num,close,amount,open'
            提取出的数据，自动按照code,date,num排序，因此code,date,num是必不可少的字段, by default "*"
        chunksize : int, optional
            每次读取的截面上的天数, by default 10
        show_time : bool, optional
            展示每次读取数据所需要的时间, by default 0
        tqdm_inside : bool, optional
            将进度条加在内部，而非外部，建议仅chunksize较大时使用, by default 0
        """
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
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
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
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
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
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                tqdm.tqdm.pandas()
                df = df.groupby(["date", "code"]).progress_apply(the_func)
                df = df.to_frame("fac").reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            else:
                # 开始计算因子值
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                df = df.groupby(["date", "code"]).apply(the_func)
                df = df.to_frame("fac").reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            self.factor_new = df
            # 拼接新的和旧的
            self.factor = (
                pd.concat([self.factor_old, self.factor_new])
                .sort_index()
                .drop_duplicates()
            )
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"补充{self.dates_new[0]}截止到{new_end_date}的因子值计算完了")
        else:
            self.factor = self.factor_old
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")


class pure_fall_flexible(object):
    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
        clickhouse: bool = 0,
        postgresql: bool = 0,
        questdb: bool = 0,
    ) -> None:
        """基于clickhouse的分钟数据，计算因子值，每天的因子值用到多日的数据，或者用到截面的数据
        对一段时间的截面数据进行操作，在get_daily_factors的func函数中
        请写入df=df.groupby([xxx]).apply(fff)之类的语句
        然后单独定义一个函数，作为要apply的fff，可以在apply上加进度条

        Parameters
        ----------
        factor_file : str
            用于存储因子的文件名称，请以'.feather'结尾
        startdate : int, optional
            计算因子的起始日期，形如20220816, by default None
        enddate : int, optional
            计算因子的终止日期，形如20220816, by default None
        kind : str, optional
            指定计算股票还是指数，指数则为'index', by default "stock"
        clickhouse : bool, optional
            使用clickhouse作为数据源，如果postgresql与本参数都为0，将依然从clickhouse中读取, by default 0
        postgresql : bool, optional
            使用postgresql作为数据源, by default 0
        questdb : bool, optional
            使用questdb作为数据源, by default 0
        """
        homeplace = HomePlace()
        self.kind = kind
        if clickhouse == 0 and postgresql == 0 and questdb == 0:
            clickhouse = 1
        self.clickhouse = clickhouse
        self.postgresql = postgresql
        self.questdb = questdb
        if clickhouse == 1:
            # 连接clickhouse
            self.chc = ClickHouseClient("minute_data")
        elif postgresql == 1:
            self.chc = PostgreSQL("minute_data")
        else:
            self.chc = Questdb()
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = pd.read_feather(self.factor_file)
            factor_old.columns = ["date"] + list(factor_old.columns)[1:]
            factor_old = factor_old.drop_duplicates(subset=["date"])
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

    def __call__(self) -> pd.DataFrame:
        """直接返回因子值的pd.DataFrame

        Returns
        -------
        `pd.DataFrame`
            计算出的因子值
        """
        return self.factor.copy()

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func: Callable,
        fields: str = "*",
        chunksize: int = 250,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        """每次抽取chunksize天的截面上全部股票的分钟数据
        依照定义的函数计算因子值

        Parameters
        ----------
        func : Callable
            用于计算因子值的函数
        fields : str, optional
            股票数据涉及到哪些字段，排除不必要的字段，可以节约读取数据的时间，形如'date,code,num,close,amount,open'
            提取出的数据，自动按照code,date,num排序，因此code,date,num是必不可少的字段, by default "*"
        chunksize : int, optional
            每次读取的截面上的天数, by default 10
        show_time : bool, optional
            展示每次读取数据所需要的时间, by default 0
        tqdm_inside : bool, optional
            将进度条加在内部，而非外部，建议仅chunksize较大时使用, by default 0
        """
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
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
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
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
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
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
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
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
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
            self.factor = (
                pd.concat([self.factor_old, self.factor_new])
                .sort_index()
                .drop_duplicates()
            )
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"补充{self.dates_new[0]}截止到{new_end_date}的因子值计算完了")
        else:
            self.factor = self.factor_old
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")


class pure_coldwinter(object):
    # DONE: 可以自由添加其他要剔除的因子，或者替换某些要剔除的因子
    def __init__(
        self,
        facs_dict: dict = None,
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
        facs_dict : dict, optional
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
        self.homeplace = HomePlace()
        # barra因子数据
        styles = os.listdir(self.homeplace.barra_data_file)
        styles = [i for i in styles if i.endswith(".feather")]
        barras = {}
        for s in styles:
            k = s.split(".")[0]
            v = pd.read_feather(self.homeplace.barra_data_file + s)
            v.columns = ["date"] + list(v.columns)[1:]
            v = v.set_index("date")
            barras[k] = v
        rename_dict = {
            "fac": "因子自身",
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
        if facs_dict is not None:
            barras.update(facs_dict)
        self.barras = barras
        self.rename_dict = rename_dict
        sort_names = list(rename_dict.values())
        if facs_dict is not None:
            sort_names = sort_names + list(facs_dict.keys())
        sort_names = [i for i in sort_names if i != "因子自身"]
        self.sort_names = sort_names

    def __call__(self):
        """返回纯净因子值"""
        return self.snow_fac

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
        self.__corr = self.__corr.rename(index=self.rename_dict)
        self.__corr = self.__corr.to_frame("相关系数").T

        self.__corr = self.__corr[self.sort_names]
        self.__corr = self.__corr.T

    @property
    def corr(self) -> pd.DataFrame:
        """因子和10个常用风格因子的相关系数

        Returns
        -------
        pd.DataFrame
            因子和10个常用风格因子的相关系数
        """
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


class pure_snowtrain(object):
    """直接返回纯净因子"""

    def __init__(
        self,
        factors: pd.DataFrame,
        facs_dict: dict = None,
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
        facs_dict : dict, optional
            _description_, by default None
        momentum : bool, optional
            _description_, by default 1
        earningsyield : bool, optional
            _description_, by default 1
        growth : bool, optional
            _description_, by default 1
        liquidity : bool, optional
            _description_, by default 1
        size : bool, optional
            _description_, by default 1
        leverage : bool, optional
            _description_, by default 1
        beta : bool, optional
            _description_, by default 1
        nonlinearsize : bool, optional
            _description_, by default 1
        residualvolatility : bool, optional
            _description_, by default 1
        booktoprice : bool, optional
            _description_, by default 1
        """
        self.winter = pure_coldwinter(
            facs_dict=facs_dict,
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
        self.winter.set_factors_df_wide(factors.copy())
        self.winter.run()
        self.corr = self.winter.corr

    def __call__(self) -> pd.DataFrame:
        """获得纯净化之后的因子值

        Returns
        -------
        pd.DataFrame
            纯净化之后的因子值
        """
        return self.winter.snow_fac.copy()


class pure_newyear(object):
    """转为生成25分组和百分组的收益矩阵而封装"""

    def __init__(
        self,
        facx: pd.DataFrame,
        facy: pd.DataFrame,
        group_num_single: int,
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
            homexy(), group_num_single**2, plt_plot=False, print_comments=False
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
    def run(self, func: Callable, backsee: int = 20) -> None:
        """执行计算的框架，产生因子值

        Parameters
        ----------
        func : Callable
            每个月要进行的操作
        backsee : int, optional
            回看期，即每个月月底对过去多少天进行计算, by default 20
        """
        self.get_fac_long_and_tradedays()
        self.get_month_starts_and_ends(backsee=backsee)
        self.get_monthly_factor(func)
