__updated__ = "2025-03-19 02:14:33"

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tqdm.auto
import scipy.stats as ss

import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

from functools import reduce, lru_cache
from loguru import logger
from plotly.tools import FigureFactory as FF
import plotly.graph_objects as go

import cufflinks as cf

try:
    cf.set_config_file(offline=True)
except Exception:
    pass
from pure_ocean_breeze.jason.data.read_data import (
    read_daily,
    read_market,
)
from pure_ocean_breeze.jason.state.homeplace import HomePlace

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")
from pure_ocean_breeze.jason.state.states import STATES
from pure_ocean_breeze.jason.state.decorators import do_on_dfs
from pure_ocean_breeze.jason.data.tools import (
    to_percent,
    standardlize,
    boom_one,
    de_cross_special_for_barra_weekly_fast,
)


@do_on_dfs
def add_cross_standardlize(*args: list) -> pd.DataFrame:
    """将众多因子横截面做z-score标准化之后相加

    Returns
    -------
    `pd.DataFrame`
        合成后的因子
    """
    res = reduce(lambda x, y: x + y, [standardlize(i) for i in args])
    return res


def show_corr(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    method: str = "pearson",
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
    corr = fac1.corrwith(fac2, axis=1, method=method)
    if show_series:
        return corr
    else:
        if plt_plot:
            corr.plot(rot=60)
            plt.show()
        return corr.mean()


def show_corrs(
    factors: list[pd.DataFrame],
    factor_names: list[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
    method: str = "pearson",
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



class frequency_controller(object):
    def __init__(self, freq: str):
        self.homeplace = HomePlace()
        self.freq = freq
        self.counts_one_year = 50
        self.time_shift = pd.DateOffset(weeks=1)
        self.comment_name = "周"
        self.days_in = 5



class pure_moon(object):
    __slots__ = [
        "homeplace",
        "factors",
        "tradedays",
        "ages",
        "amounts",
        "closes",
        "opens",
        "states",
        "opens_monthly",
        "closes_monthly",
        "rets_monthly",
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
        "long_short_comments",
        "total_comments",
        "__factors_out",
        "ics",
        "rankics",
        "rets_monthly_last",
        "freq_ctrl",
        "freq",
        "factor_cover",
        "factor_cross_skew",
        "pos_neg_rate",
        "corr_itself",
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
        "big_rankics",
        "small_rankics",
        'longside_ret_eachyear',
        'longside_ret',
    ]

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(
        cls,
        freq: str = "W",
    ):
        cls.homeplace = HomePlace()
        cls.freq = freq
        cls.freq_ctrl = frequency_controller(freq)

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
    ):
        states = read_daily(state=1, start=STATES["START"])
        opens = read_daily(vwap=1, start=STATES["START"])
        closes = read_daily(vwap=1, start=STATES["START"])
        market=read_market(zz500=1,every_stock=0,close=1).resample(cls.freq).last()
        cls.market_ret=market/market.shift(1)-1
        # 交易状态文件
        cls.states = states
        # Monday vwap
        cls.opens = opens
        # Friday vwap
        cls.closes = closes
        cls.opens = cls.opens.replace(0, np.nan)
        cls.closes = cls.closes.replace(0, np.nan)
        cls.states = read_daily(state=1)
        cls.states = cls.states.resample(cls.freq).first()
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
        self.factors = (self.factors * self.states).dropna(how="all")
        self.factor_cover = self.factors.count().sum()
        total = self.opens.resample(self.freq).last().reindex(self.factors.index).count().sum()
        self.factor_cover = min(self.factor_cover / total, 1)
        self.factor_cross_skew = self.factors.skew(axis=1).mean()
        pos_num = ((self.factors > 0) + 0).sum().sum()
        neg_num = ((self.factors < 0) + 0).sum().sum()
        self.pos_neg_rate = pos_num / (neg_num + pos_num)
        self.corr_itself = show_corr(self.factors, self.factors.shift(1),plt_plot=0)
        

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


    def get_ic_rankic(cls, df):
        """计算IC和RankIC"""
        df1 = df[["ret", "fac"]]
        rankic = df1.rank().corr().iloc[0, 1]
        small_ic=df1[df1.fac<=df1.fac.median()].rank().corr().iloc[0, 1]
        big_ic=df1[df1.fac>=df1.fac.median()].rank().corr().iloc[0, 1]
        df2 = pd.DataFrame({"rankic": [rankic],"small_rankic":[small_ic],"big_rankic":[big_ic]})
        return df2

    def get_icir_rankicir(cls, df):
        """计算ICIR和RankICIR"""
        rankic = df.rankic.mean()
        small_rankic=df.small_rankic.mean()
        big_rankic=df.big_rankic.mean()
        return pd.DataFrame(
            {"IC": [rankic]},
            index=["评价指标"],
        ),pd.DataFrame({"1-5IC":[small_rankic],"6-10IC":[big_rankic]},index=["评价指标"]).T

    def get_ic_icir_and_rank(cls, df):
        """计算IC、ICIR、RankIC、RankICIR"""
        df1 = df.groupby("date").apply(cls.get_ic_rankic)
        cls.rankics = df1.rankic
        cls.rankics = cls.rankics.reset_index(drop=True, level=1).to_frame()
        cls.small_rankics = df1.small_rankic.reset_index(drop=True, level=1).to_frame()
        cls.big_rankics = df1.big_rankic.reset_index(drop=True, level=1).to_frame()
        df2,df5 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        return df2,df5

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

    def get_group_rets_net_values(self, groups_num=10):
        """计算组内每一期的平均收益，生成每日收益率序列和净值序列"""
        
        self.group_rets = self.data.groupby(["date", "group"]).apply(
            lambda x: x.ret.mean()
        )
        self.rets_all = self.data.groupby(["date"]).apply(lambda x: x.ret.mean())
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[
            self.group_rets.index <= self.factors.date.max()
        ]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix("group")
        self.rets_all=self.rets_all.dropna()
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

    def get_long_short_comments(self):
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
        
        # week_here
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (
            self.freq_ctrl.counts_one_year**0.5
        )
        self.long_short_info_ratio = (
            self.long_short_ret_yearly / self.long_short_vol_yearly
        )
        self.long_short_comments = pd.DataFrame(
            {
                "评价指标": [
                    self.long_short_ret_yearly,
                    self.long_short_vol_yearly,
                    self.long_short_info_ratio,
                ]
            },
            # week_here
            index=[
                "收益率",
                "波动率",
                "信息比",
            ],
        )

    def get_total_comments(self):
        """综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、胜率、最大回撤率"""
        self.group_mean_rets_monthly = self.group_rets.drop(
            columns=["long_short"]
        ).mean()
        mar=self.market_ret.reindex(self.factors_out.index)
        self.group_mean_rets_monthly = (
            self.group_mean_rets_monthly - mar.mean()
        )*self.freq_ctrl.counts_one_year
        self.group1_ret_yearly= self.group_mean_rets_monthly.loc['group1']
        self.group10_ret_yearly = self.group_mean_rets_monthly.loc['group10']
        if self.group1_ret_yearly>self.group10_ret_yearly:
            self.longside_ret=self.group_rets.group1-mar
        else:
            self.longside_ret=self.group_rets.group10-mar
        self.longside_ret_eachyear=self.longside_ret.resample('Y').mean()*self.freq_ctrl.counts_one_year
        self.total_comments = pd.concat(
            [
                self.ic_icir_and_rank,
                self.long_short_comments,
                # week_here
                pd.DataFrame(
                    {
                        "评价指标": [
                            self.pos_neg_rate,
                            self.factor_cross_skew,
                            self.corr_itself,
                            self.factor_cover,
                        ]
                    },
                    index=[
                        "正值占比",
                        "截面偏度",
                        "自相关性",
                        "覆盖率",
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
                        "1组收益",
                        "10组收益",
                    ]
                )
            ]
        )
        

    def plot_net_values(self, ilegend=1, without_breakpoint=0):

        tris = self.group_net_values.drop(columns=['long_short'])
        if without_breakpoint:
            tris = tris.dropna()
        figs = cf.figures(
            tris,
            [
                dict(kind="line", y=list(tris.columns)),
            ],
            asList=True,
        )
        comments = (
            self.total_comments.applymap(lambda x: round(x, 4))
            .reset_index()
        )
        here = pd.concat(
            [
                comments.iloc[1:7, :].reset_index(drop=True),
                comments.iloc[[7,0,8,9,10,11], :].reset_index(drop=True),
            ],
            axis=1,
        )
        here.columns = ["绩效", "结果", "多与空", "结果"]
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
                name='各组收益'
            )
        )
        # table=go.Figure([go.Table(header=dict(values=list(here.columns)),cells=dict(values=here.to_numpy().tolist()))])
        if self.group1_ret_yearly>self.group10_ret_yearly:
            pic3_data = go.Bar(y=list(self.small_rankics.small_rankic), x=list(self.small_rankics.index),marker_color="red")
            pic3 = go.Figure(data=[pic3_data])
            pic5_data1 = go.Scatter(
                y=list(self.small_rankics.small_rankic.cumsum()),
                x=list(self.small_rankics.index),
                name="多头ic",
                yaxis="y2",
                mode="lines",
                line=dict(color="blue"),
            )
        else:
            pic3_data = go.Bar(y=list(self.big_rankics.big_rankic), x=list(self.big_rankics.index),marker_color="red")
            pic3 = go.Figure(data=[pic3_data])
            pic5_data1 = go.Scatter(
                y=list(self.big_rankics.big_rankic.cumsum()),
                x=list(self.big_rankics.index),
                name="多头ic",
                yaxis="y2",
                mode="lines",
                line=dict(color="blue"),
            )
        pic5_data2 = go.Scatter(
            y=list(self.rankics.rankic.cumsum()),
            x=list(self.rankics.index),
            mode="lines",
            name="rankic",
            yaxis="y2",
            line=dict(color="red"),
        )
        figs.append(table)
        figs = [figs[-1]] + figs[:-1]
        figs.append(pic2)
        figs = [figs[0], figs[1], figs[-1], go.Figure()]
        figs[3].add_trace(pic5_data1)
        figs[3].add_trace(pic5_data2)

        pic4=go.Figure()
        figs.append(pic4)
        figs[4].add_trace(go.Bar(x=[str(i) for i in self.longside_ret_eachyear.index.year],y=self.longside_ret_eachyear,name='各年收益'))            
        figs[1].update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        ) 
        
        base_layout = cf.tools.get_base_layout(figs)

        sp = cf.subplots(
            figs,
            shape=(3, 10),
            base_layout=base_layout,
            vertical_spacing=0.15,
            horizontal_spacing=0.045,
            shared_yaxes=False,
            specs=[
                [
                    {"rowspan": 3, "colspan": 3},
                    None,
                    None,
                    {"rowspan": 3, "colspan": 3},
                    None,
                    None,
                    {"colspan": 3},
                    None,
                    None,
                    None,
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    {"colspan": 3},
                    None,
                    None,
                    None,
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    {"colspan": 3},
                    None,
                    None,
                    None,
                ],
            ],
            subplot_titles=[
                "净值曲线",
                "各组月均超均收益",
                "Rank IC时序图",
            ],
        )
        sp["layout"].update(showlegend=ilegend,width=1100,height=200,margin=dict(l=0, r=0, b=0, t=0, pad=0),font=dict(size=12),legend=dict(
            # 可选值：
            # 'left', 'center', 'right' 
            # 'top', 'middle', 'bottom'
            xanchor="right",    # 图例的x锚点
            yanchor="top",     # 图例的y锚点
            x=1,              # x位置（0到1之间）
            y=1               # y位置（0到1之间）
        ))
        cf.iplot(sp)


    @classmethod
    @lru_cache(maxsize=None)
    def prerpare(cls):
        """通用数据准备"""
        cls.get_rets_month()

    def run(
        self,
        groups_num=10,
        ilegend=1,
        without_breakpoint=0,
        show_more_than=0.025,
    ):
        """运行回测部分"""
        self.__factors_out = self.factors.copy()
        self.factors = self.factors.shift(1)
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.get_data(groups_num)
        self.get_group_rets_net_values(groups_num=groups_num)
        self.get_long_short_comments()
        self.get_total_comments()

        if (show_more_than is None) or (show_more_than < max(self.group1_ret_yearly,self.group10_ret_yearly)):
            self.plot_net_values(
                ilegend=bool(ilegend),
                without_breakpoint=without_breakpoint,
            )
        else:
            logger.info(f'多头收益率为{round(max(self.group1_ret_yearly,self.group10_ret_yearly),3)}, ic为{round(self.rankics.rankic.mean(),3)}，表现太差，不展示了')
            # plt.show()


@do_on_dfs
class pure_moonnight(object):
    """封装选股框架"""

    __slots__ = ["shen"]

    def __init__(
        self,
        factors: pd.DataFrame,
        groups_num: int = 10,
        freq: str = "W",
        time_start: int = None,
        time_end: int = None,
        ilegend: bool = 1,
        without_breakpoint: bool = 0,
        show_more_than: float = 0.025,
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
        time_start : int, optional
            回测起始时间, by default None
        time_end : int, optional
            回测终止时间, by default None
        ilegend : bool, optional
            使用cufflinks绘图时，是否显示图例, by default 1
        without_breakpoint : bool, optional
            画图的时候是否去除间断点, by default 0
        show_more_than : float, optional
            展示收益率大于多少的因子，默认展示大于0.025的因子, by default 0.025
        """
        if time_start is not None:
            factors = factors[factors.index >= pd.Timestamp(str(time_start))]
        if time_end is not None:
            factors = factors[factors.index <= pd.Timestamp(str(time_end))]
        self.shen = pure_moon(freq=freq)
        self.shen.set_basic_data()
        self.shen.set_factor_df_date_as_index(factors)
        self.shen.prerpare()
        self.shen.run(
            groups_num=groups_num,
            ilegend=ilegend,
            without_breakpoint=without_breakpoint,
            show_more_than=show_more_than,
        )




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
    a = pure_moon()
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
def sun(factor:pd.DataFrame,rolling_days:int=10,with_pri:bool=1,time_start:int=20170101,show_more_than:float=0.025):
    '''先单因子测试，再测试其与常用风格之间的关系'''
    ractor=boom_one(factor.rank(axis=1),rolling_days)
    if with_pri:
        factor=boom_one(factor,rolling_days)
    pfi=de_cross_special_for_barra_weekly_fast(ractor)
    logger.info('这是中性化之后的表现')
    shen=pure_moonnight(pfi,time_start=time_start,show_more_than=show_more_than)
    if max(shen.shen.group1_ret_yearly,shen.shen.group10_ret_yearly) > show_more_than:
        logger.info('这是原始值')
        shen=pure_moonnight(factor,time_start=time_start,show_more_than=None)
    return bool(max(shen.shen.group1_ret_yearly,shen.shen.group10_ret_yearly) > show_more_than)
        # display(pfi[1])