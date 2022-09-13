import copy
import warnings
warnings.filterwarnings('ignore')
import os
import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
import statsmodels.formula.api as smf
from functools import partial, reduce
from collections import Iterable
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.express as pe
import plotly.io as pio
import scipy.stats as ss
from loguru import logger
import time
from functools import lru_cache,wraps
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import h5py
from cachier import cachier
import pickle
import redis
import swifter
import knockknock as kk
import alphalens as al

NO_LOG = False
NO_COMMENT = False
NO_SAVE = False
NO_PLOT = False
NO_SUMMER = False

if NO_SUMMER:
    ...
else:
    print('欢迎使用芷琦哥的回测框架👋')


class HomePlace(object):
    def __init__(self):
        user_file=os.path.expanduser('~')+'/'
        path_file=open(user_file+'paths.settings','rb')
        paths=pickle.load(path_file)
        self.__dict__=paths

class params_setter(object):
    '''用于标注设置参数部分的装饰器'''
    def __init__(self,slogan=None):
        if not slogan:
            slogan='这是设置参数类型的函数\n'
        self.slogan=slogan
        self.box={}

    def __call__(self,func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__]=func
        self.func=func

        def wrapper(*args,**kwargs):
            func(*args,**kwargs)
            # if not NO_LOG:
            #     logger.info(f'{func.__name__} has been called $ kind of params_setter')
        return wrapper

class main_process(object):
    '''用于标记主逻辑过程的装饰器'''
    def __init__(self,slogan=None):
        if not slogan:
            slogan='这是主逻辑过程的函数\n'
        self.slogan=slogan
        self.box={}

    def __call__(self,func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__]=func

        def wrapper(*args,**kwargs):
            func(*args,**kwargs)
            # if NO_LOG:
            #     logger.success(f'{func.__name__} has been called $ kind of main_process')
        return wrapper

class tool_box(object):
    '''用于标注工具箱部分的装饰器'''
    def __init__(self,slogan=None):
        if not slogan:
            slogan='这是工具箱的函数\n'
        self.slogan=slogan
        self.box={}

    def __call__(self,func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__]=func

        def wrapper(*args,**kwargs):
            res=func(*args,**kwargs)
            # logger.success(f'{func.__name__} has been called $ kind of tool_box')
            return res
        return wrapper

class history_remain(object):
    '''用于历史遗留部分的装饰器'''
    def __init__(self,slogan=None):
        if not slogan:
            slogan='这是历史遗留的函数\n'
        self.slogan=slogan
        self.box={}

    def __call__(self,func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__]=func

        def wrapper(*args,**kwargs):
            func(*args,**kwargs)
            # logger.success(f'{func.__name__} has been called $ kind of history_remain')
        return wrapper



@cachier()
def read_daily(path=None,open=0,close=0,high=0,low=0,tr=0,sharenum=0,volume=0,unadjust=0):
    '''读取日频数据,使用read_daily.clear_cache()来清空缓存'''
    def read_mat(path):
        homeplace=HomePlace()
        col=list(scio.loadmat(homeplace.daily_data_file+'AllStockCode.mat').values())[3]
        index=list(scio.loadmat(homeplace.daily_data_file+'TradingDate_Daily.mat').values())[3]
        col=[i[0] for i in col[0]]
        index=index[0].tolist()
        path=homeplace.daily_data_file+path
        data=list(scio.loadmat(path).values())[3]
        data=pd.DataFrame(data,index=index,columns=col)
        data.index=pd.to_datetime(data.index,format='%Y%m%d')
        data=data.replace(0,np.nan)
        return data
    if not unadjust:
        if path:
            return read_mat(path)
        elif open:
            trs=read_mat('AllStock_DailyTR.mat')
            opens=read_mat('AllStock_DailyOpen_dividend.mat')
            return np.sign(trs)*opens
        elif close:
            trs=read_mat('AllStock_DailyTR.mat')
            closes=read_mat('AllStock_DailyClose_dividend.mat')
            return np.sign(trs)*closes
        elif high:
            trs=read_mat('AllStock_DailyTR.mat')
            highs=read_mat('AllStock_DailyHigh_dividend.mat')
            return np.sign(trs)*highs
        elif low:
            trs=read_mat('AllStock_DailyTR.mat')
            lows=read_mat('AllStock_DailyLow_dividend.mat')
            return np.sign(trs)*lows
        elif tr:
            trs=read_mat('AllStock_DailyTR.mat')
            return trs
        elif sharenum:
            sharenums=read_mat('AllStock_DailyAShareNum.mat')
            return sharenums
        elif volume:
            volumes=read_mat('AllStock_DailyVolume.mat')
            return volumes
        else:
            raise IOError('阁下总得读点什么吧？🤒')
    else:
        if path:
            return read_mat(path)
        elif open:
            trs=read_mat('AllStock_DailyTR.mat')
            opens=read_mat('AllStock_DailyOpen.mat')
            return np.sign(trs)*opens
        elif close:
            trs=read_mat('AllStock_DailyTR.mat')
            closes=read_mat('AllStock_DailyClose.mat')
            return np.sign(trs)*closes
        elif high:
            trs=read_mat('AllStock_DailyTR.mat')
            highs=read_mat('AllStock_DailyHigh.mat')
            return np.sign(trs)*highs
        elif low:
            trs=read_mat('AllStock_DailyTR.mat')
            lows=read_mat('AllStock_DailyLow.mat')
            return np.sign(trs)*lows
        elif tr:
            trs=read_mat('AllStock_DailyTR.mat')
            return trs
        elif sharenum:
            sharenums=read_mat('AllStock_DailyAShareNum.mat')
            return sharenums
        elif volume:
            volumes=read_mat('AllStock_DailyVolume.mat')
            return volumes
        else:
            raise IOError('阁下总得读点什么吧？🤒')

def read_market(full=False,wide=True,open=0,high=0,low=0,close=0,amount=0,money=0):
    '''读取wind全A日行情，如果为full，则直接返回原始表格，如果full为False，则返回部分数据
    如果wide为True，则返回方阵形式，index是时间，columns是股票代码，每一列的数据都一样，这么做是指为了便于与个股运算'''
    market=pd.read_excel(homeplace.daily_data_file+'wind全A日行情.xlsx')
    market=market.drop(columns=['代码','名称'])
    market.columns=['date','open','high','low','close','amount','money']
    market.money=market.money*1000000
    if full:
        return market
    else:
        if wide:
            tr=read_daily(tr=1)
            tr=np.abs(np.sign(tr)).replace(1,0).fillna(0)
            if open:
                market=market[['date','open']]
                market.date=pd.to_datetime(market.date)
                market=market[market.date.isin(list(tr.index))]
                market=market.set_index('date')
                market=market['open']
                tr_one=tr.iloc[:,0]
                market=market+tr_one
                market=market.fillna(method='ffill')
                market=pd.DataFrame({k:list(market) for k in list(tr.columns)},index=tr.index)
            elif high:
                market=market[['date','high']]
                market.date=pd.to_datetime(market.date)
                market=market[market.date.isin(list(tr.index))]
                market=market.set_index('date')
                market=market['high']
                tr_one=tr.iloc[:,0]
                market=market+tr_one
                market=market.fillna(method='ffill')
                market=pd.DataFrame({k:list(market) for k in list(tr.columns)},index=tr.index)
            elif low:
                market=market[['date','low']]
                market.date=pd.to_datetime(market.date)
                market=market[market.date.isin(list(tr.index))]
                market=market.set_index('date')
                market=market['date','low']
                tr_one=tr.iloc[:,0]
                market=market+tr_one
                market=market.fillna(method='ffill')
                market=pd.DataFrame({k:list(market) for k in list(tr.columns)},index=tr.index)
            elif close:
                market=market[['date','close']]
                market.date=pd.to_datetime(market.date)
                market=market[market.date.isin(list(tr.index))]
                market=market.set_index('date')
                market=market['close']
                tr_one=tr.iloc[:,0]
                market=market+tr_one
                market=market.fillna(method='ffill')
                market=pd.DataFrame({k:list(market) for k in list(tr.columns)},index=tr.index)
            elif amount:
                market=market[['date','amount']]
                market.date=pd.to_datetime(market.date)
                market=market[market.date.isin(list(tr.index))]
                market=market.set_index('date')
                market=market['amount']
                tr_one=tr.iloc[:,0]
                market=market+tr_one
                market=market.fillna(method='ffill')
                market=pd.DataFrame({k:list(market) for k in list(tr.columns)},index=tr.index)
            elif money:
                market=market[['date','money']]
                market.date=pd.to_datetime(market.date)
                market=market[market.date.isin(list(tr.index))]
                market=market.set_index('date')
                market=market['money']
                tr_one=tr.iloc[:,0]
                market=market+tr_one
                market=market.fillna(method='ffill')
                market=pd.DataFrame({k:list(market) for k in list(tr.columns)},index=tr.index)
            else:
                raise IOError('您总得读点什么吧？🤒')
            return market
        else:
            cols=[varname.nameof(i) for i in [open,high,low,close,amount,money] if i==1]
            market=market[['date']+cols]
            return market

def read_h5(path):
    '''读取h5文件中的所有字典'''
    import tqdm
    import h5py
    import pandas as pd
    res={}
    a=h5py.File(path)
    for k,v in tqdm.tqdm(list(a.items()),desc='数据加载中……'):
        value=list(v.values())[-1]
        col=[i.decode('utf-8') for i in list(list(v.values())[0])]
        ind=[i.decode('utf-8') for i in list(list(v.values())[1])]
        res[k]=pd.DataFrame(value,columns=col,index=ind)
    return res

def get_value(df,n):
    '''很多因子计算时，会一次性生成很多值，使用时只取出一个值'''
    def get_value_single(x,n):
        try:
            return x[n]
        except Exception:
            return np.nan
    df=df.applymap(lambda x:get_value_single(x,n))
    return df

def comment_on_rets_and_nets(rets,nets,name):
    '''
    输入收益率序列和净值序列，输出年化收益、年化波动、信息比率、月度胜率和最大回撤率
    输入2个pd.Series，时间是索引
    '''
    duration_nets=(nets.index[-1]-nets.index[0]).days
    year_nets=duration_nets/365
    ret_yearly=(nets.iloc[-1]/nets.iloc[0])**(1/year_nets)-1
    max_draw=((nets.cummax()-nets)/nets.cummax()).max()
    vol=np.std(rets)*(12**0.5)
    info_rate=ret_yearly/vol
    win_rate=len(rets[rets>0])/len(rets)
    comments=pd.DataFrame({
        '年化收益率':ret_yearly,'年化波动率':vol,'信息比率':info_rate,'月度胜率':win_rate,'最大回撤率':max_draw
    },index=[name]).T
    return comments

def comments_on_twins(series,series1):
    '''对twins中的结果给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率'''
    ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
    duration=(series.index[-1]-series.index[0]).days
    year=duration/365
    ret_yearly=(series.iloc[-1]/series.iloc[0])**(1/year)-1
    max_draw=-(series/series.expanding(1).max()-1).min()
    vol=np.std(series1)*(12**0.5)
    sharpe=ret_yearly/vol
    wins=series1[series1>0]
    win_rate=len(wins)/len(series1)
    return pd.Series([ret,ret_yearly,vol,sharpe,win_rate,max_draw],
                     index=['总收益率','年化收益率','年化波动率','信息比率','胜率','最大回撤率'])

def comments_on_twins_periods(series,series1,periods=None):
    '''对twins中的结果给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率'''
    ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
    duration=(series.index[-1]-series.index[0]).days
    year=duration/365
    ret_yearly=(series.iloc[-1]/series.iloc[0])**(1/year)-1
    max_draw=-(series/series.expanding(1).max()-1).min()
    vol=np.std(series1)*(252**0.5)*(periods**0.5)
    sharpe=ret_yearly/vol
    wins=series1[series1>0]
    win_rate=len(wins)/len(series1)
    return pd.Series([ret,ret_yearly,vol,sharpe,win_rate,max_draw],
                     index=['总收益率','年化收益率','年化波动率','信息比率','胜率','最大回撤率'])

def daily_factor_on300500(fac,hs300=False,zz500=False,zz800=False,zz1000=False,gz2000=False,other=False):
    '''输入日频因子，把日频因子变为仅在300或者500上的股票池'''
    last=fac.resample('M').last()
    homeplace=HomePlace()
    if fac.shape[0]/last.shape[0]>2:
        if hs300:
            df=pd.read_feather(homeplace.daily_data_file+'沪深300成分股.feather').set_index('index').replace(0,np.nan)
            df=df*fac
            df=df.dropna(how='all')
        elif zz500:
            df=pd.read_feather(homeplace.daily_data_file+'中证500成分股.feather').set_index('index').replace(0,np.nan)
            df=df*fac
            df=df.dropna(how='all')
        elif zz800:
            df1=pd.read_feather(homeplace.daily_data_file+'沪深300成分股.feather').set_index('index')
            df2=pd.read_feather(homeplace.daily_data_file+'中证500成分股.feather').set_index('index')
            df=df1+df2
            df=df.replace(0,np.nan)
            df=df*fac
            df=df.dropna(how='all')
        elif zz1000:
            df=pd.read_feather(homeplace.daily_data_file+'中证1000成分股.feather').set_index('index').replace(0,np.nan)
            df=df*fac
            df=df.dropna(how='all')
        elif gz2000:
            df=pd.read_feather(homeplace.daily_data_file+'国证2000成分股.feather').set_index('index').replace(0,np.nan)
            df=df*fac
            df=df.dropna(how='all')
        elif other:
            tr=read_daily(tr=1).fillna(0).replace(0,1)
            tr=np.sign(tr)
            df1=(tr*pd.read_feather(homeplace.daily_data_file+'沪深300成分股.feather').set_index('index')).fillna(0)
            df2=(tr*pd.read_feather(homeplace.daily_data_file+'中证500成分股.feather').set_index('index')).fillna(0)
            df3=(tr*pd.read_feather(homeplace.daily_data_file+'中证1000成分股.feather').set_index('index')).fillna(0)
            df=(1-df1)*(1-df2)*(1-df3)*tr
            df=df.replace(0,np.nan)*fac
            df=df.dropna(how='all')
        else:
            raise ValueError('总得指定一下是哪个成分股吧🤒')
    else:
        if hs300:
            df=pd.read_feather(homeplace.daily_data_file+'沪深300成分股.feather').set_index('index').replace(0,np.nan).resample('M').last()
            df=df*fac
            df=df.dropna(how='all')
        elif zz500:
            df=pd.read_feather(homeplace.daily_data_file+'中证500成分股.feather').set_index('index').replace(0,np.nan).resample('M').last()
            df=df*fac
            df=df.dropna(how='all')
        elif zz800:
            df1=pd.read_feather(homeplace.daily_data_file+'沪深300成分股.feather').set_index('index').resample('M').last()
            df2=pd.read_feather(homeplace.daily_data_file+'中证500成分股.feather').set_index('index').resample('M').last()
            df=df1+df2
            df=df.replace(0,np.nan)
            df=df*fac
            df=df.dropna(how='all')
        elif zz1000:
            df=pd.read_feather(homeplace.daily_data_file+'中证1000成分股.feather').set_index('index').replace(0,np.nan).resample('M').last()
            df=df*fac
            df=df.dropna(how='all')
        elif gz2000:
            df=pd.read_feather(homeplace.daily_data_file+'国证2000成分股.feather').set_index('index').replace(0,np.nan).resample('M').last()
            df=df*fac
            df=df.dropna(how='all')
        elif other:
            tr=read_daily(tr=1).fillna(0).replace(0,1).resample('M').last()
            tr=np.sign(tr)
            df1=(tr*pd.read_feather(homeplace.daily_data_file+'沪深300成分股.feather').set_index('index').resample('M').last()).fillna(0)
            df2=(tr*pd.read_feather(homeplace.daily_data_file+'中证500成分股.feather').set_index('index').resample('M').last()).fillna(0)
            df3=(tr*pd.read_feather(homeplace.daily_data_file+'中证1000成分股.feather').set_index('index').resample('M').last()).fillna(0)
            df=(1-df1)*(1-df2)*(1-df3)
            df=df.replace(0,np.nan)*fac
            df=df.dropna(how='all')
        else:
            raise ValueError('总得指定一下是哪个成分股吧🤒')
    return df

def select_max(df1,df2):
    '''两个columns与index完全相同的df，每个值都挑出较大值'''
    return (df1+df2+np.abs(df1-df2))/2

def select_min(df1,df2):
    '''两个columns与index完全相同的df，每个值都挑出较小值'''
    return (df1+df2-np.abs(df1-df2))/2

@kk.desktop_sender(title='嘿，行业中性化做完啦～🛁')
def decap(df,daily=False,monthly=False):
    '''做市值中性化'''
    tqdm.tqdm.pandas()
    share=read_daily('AllStock_DailyAShareNum.mat')
    undi_close=read_daily('AllStock_DailyClose.mat')
    cap=(share*undi_close).stack().reset_index()
    cap.columns=['date','code','cap']
    cap.cap=ss.boxcox(cap.cap)[0]
    def single(x):
        x.cap=ss.boxcox(x.cap)[0]
        return x
    cap=cap.groupby(['date']).apply(single)
    cap=cap.set_index(['date','code']).unstack()
    cap.columns=[i[1] for i in list(cap.columns)]
    cap_monthly=cap.resample('M').last()
    last=df.resample('M').last()
    if df.shape[0]/last.shape[0]<2:
        monthly=True
    else:
        daily=True
    if daily:
        df=(pure_fallmount(df)-(pure_fallmount(cap),))()
    elif monthly:
        df=(pure_fallmount(df)-(pure_fallmount(cap_monthly),))()
    else:
        raise NotImplementedError('必须指定频率')
    return df

@kk.desktop_sender(title='嘿，行业市值中性化做完啦～🛁')
def decap_industry(df,daily=False,monthly=False):
    last=df.resample('M').last()
    homeplace=HomePlace()
    share=read_daily('AllStock_DailyAShareNum.mat')
    undi_close=read_daily('AllStock_DailyClose.mat')
    cap=(share*undi_close).stack().reset_index()
    cap.columns=['date','code','cap']
    cap.cap=ss.boxcox(cap.cap)[0]
    def single(x):
        x.cap=ss.boxcox(x.cap)[0]
        return x
    cap=cap.groupby(['date']).apply(single)
    df=df.stack().reset_index()
    df.columns=['date','code','fac']
    df=pd.merge(df,cap,on=['date','code'])
    if df.shape[0]/last.shape[0]<2:
        monthly=True
    else:
        daily=True
    def neutralize_factors(df):
        '''组内对因子进行市值中性化'''
        industry_codes=list(df.columns)
        industry_codes=[i for i in industry_codes if i.startswith('w')]
        industry_codes_str='+'.join(industry_codes)
        ols_result = smf.ols('fac~cap+'+industry_codes_str, data=df).fit()
        ols_w = ols_result.params['cap']
        ols_b = ols_result.params['Intercept']
        ols_bs={}
        for ind in industry_codes:
            ols_bs[ind]=ols_result.params[ind]
        df.fac = df.fac - ols_w * df.cap - ols_b
        for k,v in ols_bs.items():
            df.fac=df.fac-v*df[k]
        df = df[['fac']]
        return df
    if monthly:
        industry_dummy=pd.read_feather(homeplace.daily_data_file+'申万行业2021版哑变量.feather').set_index('date').groupby('code').resample('M').last()
        industry_dummy=industry_dummy.fillna(0).drop(columns=['code']).reset_index()
        industry_ws=[f'w{i}' for i in range(1,industry_dummy.shape[1]-1)]
        col=['code','date']+industry_ws
    elif daily:
        industry_dummy=pd.read_feather(homeplace.daily_data_file+'申万行业2021版哑变量.feather').fillna(0)
        industry_ws=[f'w{i}' for i in range(1,industry_dummy.shape[1]-1)]
        col=['date','code']+industry_ws
    industry_dummy.columns=col
    df=pd.merge(df,industry_dummy,on=['date','code'])
    df=df.set_index(['date','code'])
    tqdm.tqdm.pandas()
    df=df.groupby(['date']).progress_apply(neutralize_factors)
    df=df.unstack()
    df.columns=[i[1] for i in list(df.columns)]
    return df

def detect_nan(df):
    x=np.sum(df.to_numpy().flatten())
    if np.isnan(x):
        print(df)
        return np.nan
    else:
        return x

def deboth(df):
    shen=pure_moonnight(df,boxcox=1)
    return shen()

def boom_four(df,minus=None,backsee=20,daily=False,min_periods=None):
    '''使用20天均值和标准差，生成4个因子'''
    if min_periods is None:
        min_periods=int(backsee*0.5)
    if not daily:
        df_mean=df.rolling(backsee,min_periods=min_periods).mean().resample('M').last()
        df_std=df.rolling(backsee,min_periods=min_periods).std().resample('M').last()
        twins_add=(pure_fallmount(df_mean)+(pure_fallmount(df_std),))()
        rtwins_add=df_mean.rank(axis=1)+df_std.rank(axis=1)
        twins_minus=(pure_fallmount(df_mean)+(pure_fallmount(-df_std),))()
        rtwins_minus=df_mean.rank(axis=1)-df_std.rank(axis=1)
    else:
        df_mean=df.rolling(backsee,min_periods=min_periods).mean()
        df_std=df.rolling(backsee,min_periods=min_periods).std()
        twins_add=(pure_fallmount(df_mean)+(pure_fallmount(df_std),))()
        rtwins_add=df_mean.rank(axis=1)+df_std.rank(axis=1)
        twins_minus=(pure_fallmount(df_mean)+(pure_fallmount(-df_std),))()
        rtwins_minus=df_mean.rank(axis=1)-df_std.rank(axis=1)
    return df_mean,df_std,twins_add,rtwins_add,twins_minus,rtwins_minus

def get_abs(df,median=False,square=False):
    '''生产因子截面上距离均值的距离'''
    if not square:
        if median:
            return np.abs((df.T-df.T.median()).T)
        else:
            return np.abs((df.T-df.T.mean()).T)
    else:
        if median:
            return ((df.T-df.T.median()).T)**2
        else:
            return ((df.T-df.T.mean()).T)**2

def add_cross_standardlize(*args):
    '''将众多因子横截面标准化之后相加'''
    fms=[pure_fallmount(i) for i in args]
    one=fms[0]
    others=fms[1:]
    final=one+others
    return final()

def get_normal(df):
    '''将因子横截面正态化'''
    df=df.replace(0,np.nan)
    df=df.T.apply(lambda x:ss.boxcox(x)[0]).T
    return df


def read_index_three(day=False):
    '''读取三大指数的原始行情数据，返回并保存在本地'''
    def read(file):
        homeplace=HomePlace()
        file1=homeplace.daily_data_file+f'{file}日行情.xlsx'
        df=pd.read_excel(file1)
        df.columns=['code','name','date','open','high','low','close','amount','money']
        df=df[['date','close']]
        df.date=pd.to_datetime(df.date)
        df=df.set_index('date')
        df=df.resample('M').last()
        df.columns=[file]
        if not day:
            df=df[df.index>=pd.Timestamp('2013-04-01')]
        else:
            df=df[df.index>=pd.Timestamp(day)]
        df=df/df[file].iloc[0]
        return df
    hs300=read('沪深300')
    zz500=read('中证500')
    zz1000=read('中证1000')
    w=pd.ExcelWriter('3510原始行情.xlsx')
    hs300.to_excel(w,sheet_name='300')
    zz500.to_excel(w,sheet_name='500')
    zz1000.to_excel(w,sheet_name='1000')
    w.save()
    w.close()
    return hs300,zz500,zz1000

def make_relative_comments(ret_fac,hs300=0,zz500=0,zz1000=0,day=False):
    if hs300:
        net_index=read_index_three(day=day)[0].iloc[:,0]
    elif zz500:
        net_index=read_index_three(day=day)[1].iloc[:,0]
    elif zz1000:
        net_index=read_index_three(day=day)[2].iloc[:,0]
    else:
        raise IOError('你总得指定一个股票池吧？')
    ret_index=net_index.pct_change()
    ret=ret_fac-ret_index
    ret=ret.dropna()
    net=(1+ret).cumprod()
    net=net/net.iloc[0]
    com=comments_on_twins(net,ret)
    return com

def make_relative_comments_plot(ret_fac,hs300=0,zz500=0,zz1000=0,day=False):
    if hs300:
        net_index=read_index_three(day=day)[0].iloc[:,0]
    elif zz500:
        net_index=read_index_three(day=day)[1].iloc[:,0]
    elif zz1000:
        net_index=read_index_three(day=day)[2].iloc[:,0]
    else:
        raise IOError('你总得指定一个股票池吧？')
    ret_index=net_index.pct_change()
    ret=ret_fac-ret_index
    ret=ret.dropna()
    net=(1+ret).cumprod()
    net=net/net.iloc[0]
    com=comments_on_twins(net,ret)
    net.plot()
    plt.show()
    return net


def comments_ten(shen):
    rets_cols=list(shen.shen.group_rets.columns)
    rets_cols=rets_cols[:-1]
    coms=[]
    for i in rets_cols:
        ret=shen.shen.group_rets[i]
        net=shen.shen.group_net_values[i]
        com=comments_on_twins(net,ret)
        com=com.to_frame(i)
        coms.append(com)
    df=pd.concat(coms,axis=1)
    return df.T

def coin_reverse(ret20,vol20,mean=1,positive_negtive=0):
    '''根据vol20的大小，翻转一半ret20，把vol20较大的部分，给ret20添加负号'''
    if positive_negtive:
        if not mean:
            down20=np.sign(ret20)
            down20=down20.replace(1,np.nan)
            down20=down20.replace(-1,1)

            vol20_down=down20*vol20
            vol20_down=(vol20_down.T-vol20_down.T.median()).T
            vol20_down=np.sign(vol20_down)
            ret20_down=ret20[ret20<0]
            ret20_down=vol20_down*ret20_down

            up20=np.sign(ret20)
            up20=up20.replace(-1,np.nan)

            vol20_up=up20*vol20
            vol20_up=(vol20_up.T-vol20_up.T.median()).T
            vol20_up=np.sign(vol20_up)
            ret20_up=ret20[ret20>0]
            ret20_up=vol20_up*ret20_up

            ret20_up=ret20_up.replace(np.nan,0)
            ret20_down=ret20_down.replace(np.nan,0)
            new_ret20=ret20_up+ret20_down
            new_ret20_tr=new_ret20.replace(0,np.nan)
            return new_ret20_tr
        else:
            down20=np.sign(ret20)
            down20=down20.replace(1,np.nan)
            down20=down20.replace(-1,1)

            vol20_down=down20*vol20
            vol20_down=(vol20_down.T-vol20_down.T.mean()).T
            vol20_down=np.sign(vol20_down)
            ret20_down=ret20[ret20<0]
            ret20_down=vol20_down*ret20_down

            up20=np.sign(ret20)
            up20=up20.replace(-1,np.nan)

            vol20_up=up20*vol20
            vol20_up=(vol20_up.T-vol20_up.T.mean()).T
            vol20_up=np.sign(vol20_up)
            ret20_up=ret20[ret20>0]
            ret20_up=vol20_up*ret20_up

            ret20_up=ret20_up.replace(np.nan,0)
            ret20_down=ret20_down.replace(np.nan,0)
            new_ret20=ret20_up+ret20_down
            new_ret20_tr=new_ret20.replace(0,np.nan)
            return new_ret20_tr
    else:
        if not mean:
            vol20_dummy=np.sign((vol20.T-vol20.T.median()).T)
            ret20=ret20*vol20_dummy
            return ret20
        else:
            vol20_dummy=np.sign((vol20.T-vol20.T.mean()).T)
            ret20=ret20*vol20_dummy
            return ret20


def indus_name(df,col_name=None):
    '''将2021版申万行业的代码，转化为对应行业的名字'''
    names=pd.DataFrame({
        'indus_we_cant_same':
        ['801170.SI','801010.SI','801140.SI','801080.SI','801780.SI','801110.SI','801230.SI','801950.SI',
        '801180.SI','801040.SI','801740.SI','801890.SI','801770.SI','801960.SI','801200.SI','801120.SI','801710.SI',
        '801720.SI','801880.SI','801750.SI','801050.SI','801790.SI','801150.SI','801980.SI','801030.SI','801730.SI',
        '801160.SI','801130.SI','801210.SI','801970.SI','801760.SI'],
        '行业名称':
        ['交通运输','农林牧渔','轻工制造','电子','银行','家用电器','综合','煤炭','房地产','钢铁','国防军工','机械设备',
        '通信','石油石化','商贸零售','食品饮料','建筑材料','建筑装饰','汽车','计算机','有色金属','非银金融','医药生物','美容护理',
        '基础化工','电力设备','公用事业','纺织服饰','社会服务','环保','传媒']
    }).sort_values(['indus_we_cant_same'])
    if col_name:
        names=names.rename(columns={'indus_we_cant_same':col_name})
        df=pd.merge(df,names,on=[col_name])
    else:
        df=df.reset_index()
        df=df.rename(columns={list(df.columns)[0]:'indus_we_cant_same'})
        df=pd.merge(df,names,on=['indus_we_cant_same']).set_index('行业名称').drop(columns=['indus_we_cant_same'])
    return df

INDUS_DICT={k:v for k,v in zip(['801170.SI','801010.SI','801140.SI','801080.SI','801780.SI','801110.SI','801230.SI','801950.SI',
        '801180.SI','801040.SI','801740.SI','801890.SI','801770.SI','801960.SI','801200.SI','801120.SI','801710.SI',
        '801720.SI','801880.SI','801750.SI','801050.SI','801790.SI','801150.SI','801980.SI','801030.SI','801730.SI',
        '801160.SI','801130.SI','801210.SI','801970.SI','801760.SI'],['交通运输','农林牧渔','轻工制造','电子','银行','家用电器','综合','煤炭','房地产',
                                                                      '钢铁','国防军工','机械设备','通信','石油石化','商贸零售','食品饮料','建筑材料',
                                                                      '建筑装饰','汽车','计算机','有色金属','非银金融','医药生物','美容护理','基础化工',
                                                                      '电力设备','公用事业','纺织服饰','社会服务','环保','传媒'])}

def multidfs_to_one(*args):
    '''很多个df，各有一部分，其余位置都是空，
    想把各自df有值的部分保留，都没有值的部分继续设为空'''
    dfs=[i.fillna(0) for i in args]
    background=np.sign(np.abs(np.sign(sum(dfs)))+1).replace(1,0)
    dfs=[(i+background).fillna(0) for i in dfs]
    df_nans=[i.isna() for i in dfs]
    nan=reduce(lambda x,y:x*y,df_nans)
    nan=nan.replace(1,np.nan)
    nan=nan.replace(0,1)
    df_final=sum(dfs)*nan
    return df_final


def to_tradeends(df):
    '''将最后一个自然日改变为最后一个交易日'''
    trs=read_daily(tr=1)
    trs=trs.assign(tradeends=list(trs.index))
    trs=trs[['tradeends']]
    trs=trs.resample('M').last()
    df=pd.concat([trs,df],axis=1)
    df=df.set_index(['tradeends'])
    return df


def market_kind(df,zhuban=False,chuangye=False,kechuang=False,beijing=False):
    '''与宽基指数成分股的函数类似，限定股票在某个具体板块上'''
    trs=read_daily(tr=1)
    codes=list(trs.columns)
    dates=list(trs.index)
    if chuangye and kechuang:
        dummys=[1 if code[:2] in ['30','68'] else np.nan for code in codes]
    else:
        if zhuban:
            dummys=[1 if code[:2] in ['00','60'] else np.nan for code in codes]
        elif chuangye:
            dummys=[1 if code.startswith('3') else np.nan for code in codes]
        elif kechuang:
            dummys=[1 if code.startswith('68') else np.nan for code in codes]
        elif beijing:
            dummys=[1 if code.startswith('8') else np.nan for code in codes]
        else:
            raise ValueError('你总得选一个股票池吧？🤒')
    dummy_dict={k:v for k,v in zip(codes,dummys)}
    dummy_df=pd.DataFrame(dummy_dict,index=dates)
    df=df*dummy_df
    return df


def show_corr(fac1,fac2,method='spearman'):
    both1=fac1.stack().reset_index()
    befo1=fac2.stack().reset_index()
    both1.columns=['date','code','both']
    befo1.columns=['date','code','befo']
    twins=pd.merge(both1,befo1,on=['date','code']).set_index(['date','code'])
    corr=twins.groupby('date').apply(lambda x:x.corr(method=method).iloc[0,1])
    corr.plot()
    plt.show()
    return corr.mean()


# 半衰期序列
def calc_exp_list(window,half_life):
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)

#weighted_std
def calcWeightedStd(series, weights):
    '''
       加权平均std
    '''
    weights /= np.sum(weights)
    return np.sqrt(np.sum((series-np.mean(series)) ** 2 * weights))


def other_periods_comments_nets(fac,period=None,way=None,comments_writer=None,nets_writer=None,sheetname=None,group_num=10):
    '''不同频率下的评价指标，请输入行业市值中性化后的因子值'''
    closes=read_daily(open=1).shift(-1)
    fac1=fac.stack()
    df=al.utils.get_clean_factor_and_forward_returns(fac1,closes[closes.index.isin(fac.index)],quantiles=group_num,periods=(period,))
    df=df.reset_index()
    ics=df.groupby(['date'])[[f'{period}D','factor']].apply(lambda x:x.corr(method='spearman').iloc[0,1])
    ic=ics.mean()
    ir=ics.std()
    icir=ic/ir*(252**0.5)/(period**0.5)
    df=df.groupby(['date','factor_quantile'])[f'{period}D'].mean()/period
    df=df.unstack()
    df.columns=[f'分组{i}' for i in list(df.columns)]
    if way=='pos':
        df=df.assign(多空对冲=df[f'分组{group_num}']-df.分组1)
    elif way=='neg':
        df=df.assign(多空对冲=df.分组1-df[f'分组{group_num}'])
    nets=(df+1).cumprod()
    nets=nets.apply(lambda x:x/x.iloc[0])
    nets.plot()
    plt.show()
    comments=comments_on_twins_periods(nets.多空对冲,df.多空对冲,period)
    comments=pd.concat([pd.Series([ic,icir],index=['Rank IC','Rank ICIR']),comments])
    print(comments)
    if sheetname is None:
        ...
    else:
        if comments_writer is None:
            ...
        else:
            comments.to_excel(comments_writer,sheetname)
        if nets_writer is None:
            ...
        else:
            nets.to_excel(nets_writer,sheetname)
    return comments,nets


def get_list_std(delta_sts):
    '''同一天多个因子，计算这些因子在当天的标准差'''
    delta_sts_mean=sum(delta_sts)/len(delta_sts)
    delta_sts_std=[(i-delta_sts_mean)**2 for i in delta_sts]
    delta_sts_std=sum(delta_sts_std)
    delta_sts_std=delta_sts_std**0.5/len(delta_sts)
    return delta_sts_std


def get_industry_dummies(daily=False,monthly=False):
    '''生成31个行业的哑变量矩阵，返回一个字典'''
    homeplace=HomePlace()
    if monthly:
        industry_dummy=pd.read_feather(homeplace.daily_data_file+'申万行业2021版哑变量.feather')
        industry_dummy=industry_dummy.set_index('date').groupby('code').resample('M').last().fillna(0).drop(columns=['code']).reset_index()
    elif daily:
        industry_dummy=pd.read_feather(homeplace.daily_data_file+'申万行业2021版哑变量.feather').fillna(0)
    else:
        raise ValueError('您总得指定一个频率吧？🤒')
    ws=list(industry_dummy.columns)[2:]
    ress={}
    for w in ws:
        df=industry_dummy[['date', 'code', w]]
        df=df.pivot(index='date',columns='code',values=w)
        df=df.replace(0,np.nan)
        ress[w]=df
    return ress

class pure_moon():
    __slots__=[
        'homeplace'
        'path_prefix',
        'codes_path',
        'tradedays_path',
        'ages_path',
        'sts_path',
        'states_path',
        'opens_path',
        'closes_path',
        'highs_path',
        'lows_path',
        'pricloses_path',
        'flowshares_path',
        'amounts_path',
        'turnovers_path',
        'factors_file',
        'sts_monthly_file',
        'states_monthly_file',
        'sts_monthly_by10_file',
        'states_monthly_by10_file',
        'factors',
        'codes',
        'tradedays',
        'ages',
        'amounts',
        'closes',
        'flowshares',
        'highs',
        'lows',
        'opens',
        'pricloses',
        'states',
        'sts',
        'turnovers',
        'sts_monthly',
        'states_monthly',
        'ages_monthly',
        'tris_monthly',
        'opens_monthly',
        'closes_monthly',
        'rets_monthly',
        'opens_monthly_shift',
        'rets_monthly_begin',
        'limit_ups',
        'limit_downs',
        'data',
        'ic_icir_and_rank',
        'rets_monthly_limit_downs',
        'group_rets',
        'long_short_rets',
        'long_short_net_values',
        'group_net_values',
        'long_short_ret_yearly',
        'long_short_vol_yearly',
        'long_short_info_ratio',
        'long_short_win_times',
        'long_short_win_ratio',
        'retreats',
        'max_retreat',
        'long_short_comments',
        'total_comments',
        'square_rets',
        'cap',
        'cap_value',
        'industry_dummy',
        'industry_codes',
        'industry_codes_str',
        'industry_ws',
        'factors_out',
        'pricloses_copy',
        'flowshares_copy'
    ]

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(cls):
        now=datetime.datetime.now()
        now=datetime.datetime.strftime(now,format='%Y-%m-%d %H:%M:%S')
        cls.homeplace=HomePlace()
        # logger.add('pure_moon'+now+'.log')
        # 绝对路径前缀
        cls.path_prefix = cls.homeplace.daily_data_file
        # 股票代码文件
        cls.codes_path = 'AllStockCode.mat'
        # 交易日期文件
        cls.tradedays_path = 'TradingDate_Daily.mat'
        # 上市天数文件
        cls.ages_path = 'AllStock_DailyListedDate.mat'
        # st日子标志文件
        cls.sts_path = 'AllStock_DailyST.mat'
        # 交易状态文件
        cls.states_path = 'AllStock_DailyStatus.mat'
        # 复权开盘价数据文件
        cls.opens_path = 'AllStock_DailyOpen_dividend.mat'
        # 复权收盘价数据文件
        cls.closes_path = 'AllStock_DailyClose_dividend.mat'
        #复权最高价数据文件
        cls.highs_path = 'Allstock_DailyHigh_dividend.mat'
        #复权最低价数据文件
        cls.lows_path = 'Allstock_DailyLow_dividend.mat'
        # 不复权收盘价数据文件
        cls.pricloses_path = 'AllStock_DailyClose.mat'
        # 流通股本数据文件
        cls.flowshares_path = 'AllStock_DailyAShareNum.mat'
        # 成交量数据文件
        cls.amounts_path = 'AllStock_DailyVolume.mat'
        # 换手率数据文件
        cls.turnovers_path = 'AllStock_DailyTR.mat'
        # 因子数据文件
        cls.factors_file = ''
        # 已经算好的月度st状态文件
        cls.sts_monthly_file = 'sts_monthly.feather'
        # 已经算好的月度交易状态文件
        cls.states_monthly_file = 'states_monthly.feather'
        # 已经算好的月度st_by10状态文件
        cls.sts_monthly_by10_file = 'sts_monthly_by10.feather'
        # 已经算好的月度交易状态文件
        cls.states_monthly_by10_file = 'states_monthly_by10.feather'
        # 拼接绝对路径前缀和相对路径
        dirs = dir(cls)
        dirs.remove('new_path')
        dirs.remove('set_factor_file')
        dirs = [i for i in dirs if i.endswith('path')] + [i for i in dirs if i.endswith('file')]
        dirs_values = list(map(lambda x, y: getattr(x, y), [cls] * len(dirs), dirs))
        dirs_values = list(map(lambda x, y: x + y, [cls.path_prefix] * len(dirs), dirs_values))
        for attr, value in zip(dirs, dirs_values):
            setattr(cls, attr, value)

    def __call__(self, fallmount=0):
        '''调用对象则返回因子值'''
        df=self.factors_out.copy()
        # df=df.set_index(['date', 'code']).unstack()
        df.columns=list(map(lambda x:x[1],list(df.columns)))
        if fallmount == 0:
            return df
        else:
            return pure_fallmount(df)

    @params_setter(slogan=None)
    # @lru_cache(maxsize=None)
    def set_factor_file(self, factors_file):
        '''设置因子文件的路径，因子文件列名应为股票代码，索引为时间'''
        self.factors_file = factors_file
        self.factors = pd.read_feather(self.factors_file)
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors.reset_index()

    @params_setter(slogan=None)
    # @lru_cache(maxsize=None)
    def set_factor_df_date_as_index(self, df):
        '''设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间'''
        df = df.reset_index()
        df.columns = ['date'] + list(df.columns)[1:]
        # df.date=df.date.apply(self.next_month_end)
        # df=df.set_index('date')
        self.factors = df
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors.reset_index()

    @params_setter(slogan=None)
    # @lru_cache(maxsize=None)
    def set_factor_df_wide(self, df):
        '''从dataframe读入因子宽数据'''
        if isinstance(df,pure_fallmount):
            df=df()
        self.factors = df.copy()
        # self.factors.date=self.factors.date.apply(self.next_month_end)
        # self.factors=self.factors.set_index('date')
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors.reset_index()

    # def set_factor_df_long(self,df):
    #     '''从dataframe读入因子长数据'''
    #     self.factors=df
    #     self.factors.columns=['date','code','fac']

    @classmethod
    @lru_cache(maxsize=None)
    @history_remain(slogan=None)
    def new_path(cls, **kwargs):
        '''修改日频数据文件的路径，便于更新数据
        要修改的路径以字典形式传入，键为属性名，值为要设置的新路径'''
        for key, value in kwargs.items():
            setattr(cls, key, value)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def col_and_index(cls):
        '''读取股票代码，作为未来表格的行名
        读取交易日历，作为未来表格的索引'''
        cls.codes = list(scio.loadmat(cls.codes_path).values())[3]
        cls.tradedays = list(scio.loadmat(cls.tradedays_path).values())[3].astype(str)
        cls.codes = cls.codes.flatten().tolist()
        # cls.tradedays = cls.tradedays.flatten().tolist()
        cls.codes = list(map(lambda x: x[0], cls.codes))
        cls.tradedays=cls.tradedays[0].tolist()

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def loadmat(cls, path):
        '''重写一个加载mat文件的函数，以使代码更简洁'''
        return list(scio.loadmat(path).values())[3]

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def make_df(cls, data):
        '''将读入的数据，和股票代码与时间拼接，做成dataframe'''
        data = pd.DataFrame(data, columns=cls.codes, index=cls.tradedays)
        data.index = pd.to_datetime(data.index, format='%Y%m%d')
        return data

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def load_all_files(cls):
        '''加全部的mat文件'''
        attrs = dir(cls)
        attrs = [i for i in attrs if i.endswith('path')]
        attrs.remove('codes_path')
        attrs.remove('tradedays_path')
        attrs.remove('new_path')
        for attr in attrs:
            new_attr = attr[:-5]
            setattr(cls, new_attr, cls.make_df(cls.loadmat(getattr(cls, attr))))
        cls.opens = cls.opens.replace(0, np.nan)
        cls.closes = cls.closes.replace(0, np.nan)
        cls.pricloses_copy=cls.pricloses.copy()
        cls.flowshares_copy=cls.flowshares.copy()

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_st(cls, df):
        '''比较一个月内st的天数，如果st天数多，就删除本月，如果正常多，就保留本月'''
        st_count = len(df[df == 1])
        normal_count = len(df[df != 1])
        if st_count >= normal_count:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_st_by10(cls, df):
        '''比较一个月内正常交易的天数，如果少于10天，就删除本月'''
        normal_count = len(df[df != 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_state(cls, df):
        '''比较一个月内非正常交易的天数，如果非正常交易天数多，就删除本月，否则保留本月'''
        abnormal_count = len(df[df == 0])
        normal_count = len(df[df == 1])
        if abnormal_count >= normal_count:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def judge_month_state_by10(cls, df):
        '''比较一个月内正常交易天数，如果少于10天，就删除本月'''
        normal_count = len(df[df == 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def read_add(cls, pridf, df, func):
        '''由于数据更新，过去计算的月度状态可能需要追加'''
        if not NO_LOG:
            logger.info(f'this is max_index of pridf{pridf.index.max()}')
            logger.info(f'this is max_index of df{df.index.max()}')
        if pridf.index.max() > df.index.max():
            df_add = pridf[pridf.index > df.index.max()]
            df_add = df_add.resample('M').apply(func)
            df = pd.concat([df, df_add])
            return df
        else:
            return df

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def write_feather(cls, df, path):
        '''将算出来的数据存入本地，以免造成重复运算'''
        df1 = df.copy()
        df1 = df1.reset_index()
        df1.to_feather(path)

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def daily_to_monthly(cls, pridf, path, func):
        '''把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算'''
        try:
            if not NO_LOG:
                logger.info('try to read the prepared state file')
            month_df = pd.read_feather(path).set_index('index')
            if not NO_LOG:
                logger.info('state file load success')
            month_df = cls.read_add(pridf, month_df, func)
            if not NO_LOG:
                logger.info('adding after state file has finish')
            cls.write_feather(month_df, path)
            if not NO_LOG:
                logger.info('the feather is new now')
        except Exception as e:
            if not NO_LOG:
                logger.error('error occurs when read state files')
                logger.error(e)
            print('state file rewriting……')
            month_df = pridf.resample('M').apply(func)
            cls.write_feather(month_df, path)
        return month_df

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def daily_to_monthly_by10(cls, pridf, path, func):
        '''把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算'''
        try:
            month_df = pd.read_feather(path).set_index('date')
            month_df = cls.read_add(pridf, month_df, func)
            cls.write_feather(month_df, path)
        except Exception:
            print('rewriting')
            month_df = pridf.resample('M').apply(func)
            cls.write_feather(month_df, path)
        return month_df

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def judge_month(cls):
        '''生成一个月综合判断的表格'''
        cls.sts_monthly = cls.daily_to_monthly(cls.sts, cls.sts_monthly_file, cls.judge_month_st)
        cls.states_monthly = cls.daily_to_monthly(cls.states, cls.states_monthly_file, cls.judge_month_state)
        cls.ages_monthly = cls.ages.resample('M').last()
        cls.ages_monthly = np.sign(cls.ages_monthly.applymap(lambda x: x - 60)).replace(-1, 0)
        cls.tris_monthly = cls.sts_monthly * cls.states_monthly * cls.ages_monthly
        cls.tris_monthly = cls.tris_monthly.replace(0, np.nan)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def judge_month_by10(cls):
        '''生成一个月综合判断的表格'''
        cls.sts_monthly = cls.daily_to_monthly(cls.sts, cls.sts_monthly_by10_file, cls.judge_month_st_by10)
        cls.states_monthly = cls.daily_to_monthly(cls.states, cls.states_monthly_by10_file,
                                                    cls.judge_month_state_by10)
        cls.ages_monthly = cls.ages.resample('M').last()
        cls.ages_monthly = np.sign(cls.ages_monthly.applymap(lambda x: x - 60)).replace(-1, 0)
        cls.tris_monthly = cls.sts_monthly * cls.states_monthly * cls.ages_monthly
        cls.tris_monthly = cls.tris_monthly.replace(0, np.nan)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def get_rets_month(cls):
        '''计算每月的收益率，并根据每月做出交易状态，做出删减'''
        cls.opens_monthly = cls.opens.resample('M').first()
        cls.closes_monthly = cls.closes.resample('M').last()
        cls.rets_monthly = (cls.closes_monthly - cls.opens_monthly) / cls.opens_monthly
        cls.rets_monthly = cls.rets_monthly * cls.tris_monthly
        cls.rets_monthly = cls.rets_monthly.stack().reset_index()
        cls.rets_monthly.columns = ['date', 'code', 'ret']

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def neutralize_factors(cls, df):
        '''组内对因子进行市值中性化'''
        industry_codes=list(df.columns)
        industry_codes=[i for i in industry_codes if i.startswith('w')]
        industry_codes_str='+'.join(industry_codes)
        ols_result = smf.ols('fac~cap_size+'+industry_codes_str, data=df).fit()
        ols_w = ols_result.params['cap_size']
        ols_b = ols_result.params['Intercept']
        ols_bs={}
        for ind in industry_codes:
            ols_bs[ind]=ols_result.params[ind]
        df.fac = df.fac - ols_w * df.cap_size - ols_b
        for k,v in ols_bs.items():
            df.fac=df.fac-v*df[k]
        df = df[['fac']]
        return df

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def get_log_cap(cls,boxcox=False):
        '''获得对数市值'''
        try:
            cls.pricloses = cls.pricloses.replace(0, np.nan)
            cls.flowshares = cls.flowshares.replace(0, np.nan)
            cls.pricloses = cls.pricloses.resample('M').last()
        except Exception:
            cls.pricloses=cls.pricloses_copy.copy()
            cls.pricloses = cls.pricloses.replace(0, np.nan)
            cls.flowshares = cls.flowshares.replace(0, np.nan)
            cls.pricloses = cls.pricloses.resample('M').last()
        cls.pricloses = cls.pricloses.stack().reset_index()
        cls.pricloses.columns = ['date', 'code', 'priclose']
        try:
            cls.flowshares = cls.flowshares.resample('M').last()
        except Exception:
            cls.flowshares=cls.flowshares_copy.copy()
            cls.flowshares = cls.flowshares.resample('M').last()
        cls.flowshares = cls.flowshares.stack().reset_index()
        cls.flowshares.columns = ['date', 'code', 'flowshare']
        cls.flowshares = pd.merge(cls.flowshares, cls.pricloses, on=['date', 'code'])
        cls.cap = cls.flowshares.assign(cap_size=cls.flowshares.flowshare * cls.flowshares.priclose)
        if boxcox:
            def single(x):
                x.cap_size=ss.boxcox(x.cap_size)[0]
                return x
            cls.cap=cls.cap.groupby(['date']).apply(single)
        else:
            cls.cap['cap_size'] = np.log(cls.cap['cap_size'])

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_neutral_factors(self):
        '''对因子进行市值中性化'''
        self.factors = self.factors.set_index('date')
        self.factors.index=self.factors.index+pd.DateOffset(months=1)
        self.factors = self.factors.resample('M').last()
        last_date=self.tris_monthly.index.max()+pd.DateOffset(months=1)
        last_date=last_date+pd.tseries.offsets.MonthEnd()
        add_tail=pd.DataFrame(1,index=[last_date],columns=self.tris_monthly.columns)
        tris_monthly=pd.concat([self.tris_monthly,add_tail])
        self.factors = self.factors * tris_monthly
        self.factors.index=self.factors.index-pd.DateOffset(months=1)
        self.factors=self.factors.resample('M').last()
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ['date', 'code', 'fac']
        self.factors = pd.merge(self.factors, self.cap, how='inner', on=['date', 'code'])
        self.industry_dummy=pd.read_feather(self.homeplace.daily_data_file+'申万行业2021版哑变量.feather').set_index('date').groupby('code').resample('M').last()
        self.industry_dummy=self.industry_dummy.drop(columns=['code']).reset_index()
        self.industry_ws=[f'w{i}' for i in range(1,self.industry_dummy.shape[1]-1)]
        col=['code','date']+self.industry_ws
        self.industry_dummy.columns=col
        self.factors=pd.merge(self.factors,self.industry_dummy,on=['date','code'])
        self.factors = self.factors.set_index(['date', 'code'])
        self.factors = self.factors.groupby(['date']).apply(self.neutralize_factors)
        self.factors = self.factors.reset_index()

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def deal_with_factors(self):
        '''删除不符合交易条件的因子数据'''
        self.factors = self.factors.set_index('date')
        self.factors_out=self.factors.copy()
        self.factors.index=self.factors.index+pd.DateOffset(months=1)
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors * self.tris_monthly
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ['date', 'code', 'fac']

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def deal_with_factors_after_neutralize(self):
        '''中性化之后的因子处理方法'''
        self.factors = self.factors.set_index(['date', 'code'])
        self.factors = self.factors.unstack()
        self.factors_out=self.factors.copy()
        self.factors.index=self.factors.index+pd.DateOffset(months=1)
        self.factors = self.factors.resample('M').last()
        self.factors.columns = list(map(lambda x: x[1], list(self.factors.columns)))
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ['date', 'code', 'fac']

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def find_limit(cls, df, up=1):
        '''计算涨跌幅超过9.8%的股票，并将其存储进一个长列表里
        其中时间列，为某月的最后一天；涨停日虽然为下月初第一天，但这里标注的时间统一为上月最后一天'''
        limit_df = np.sign(df.applymap(lambda x: x - up * 0.098)).replace(-1 * up, np.nan)
        limit_df = limit_df.stack().reset_index()
        limit_df.columns = ['date', 'code', 'limit_up_signal']
        limit_df = limit_df[['date', 'code']]
        return limit_df

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def get_limit_ups_downs(cls):
        '''找月初第一天就涨停'''
        '''或者是月末跌停的股票'''
        cls.opens_monthly_shift = cls.opens_monthly.copy()
        cls.opens_monthly_shift = cls.opens_monthly_shift.shift(-1)
        cls.rets_monthly_begin = (cls.opens_monthly_shift - cls.closes_monthly) / cls.closes_monthly
        cls.closes2_monthly = cls.closes.shift(1).resample('M').last()
        cls.rets_monthly_last = (cls.closes_monthly - cls.closes2_monthly) / cls.closes2_monthly
        cls.limit_ups = cls.find_limit(cls.rets_monthly_begin, up=1)
        cls.limit_downs = cls.find_limit(cls.rets_monthly_last, up=-1)

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_ic_rankic(cls, df):
        '''计算IC和RankIC'''
        df1 = df[['ret', 'fac']]
        ic = df1.corr(method='pearson').iloc[0, 1]
        rankic = df1.corr(method='spearman').iloc[0, 1]
        df2 = pd.DataFrame({'ic': [ic], 'rankic': [rankic]})
        return df2

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_icir_rankicir(cls, df):
        '''计算ICIR和RankICIR'''
        ic = df.ic.mean()
        rankic = df.rankic.mean()
        icir = ic / np.std(df.ic) * (12 ** (0.5))
        rankicir = rankic / np.std(df.rankic) * (12 ** (0.5))
        return pd.DataFrame({'IC': [ic], 'ICIR': [icir], 'RankIC': [rankic], 'RankICIR': [rankicir]}, index=['评价指标'])

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_ic_icir_and_rank(cls, df):
        '''计算IC、ICIR、RankIC、RankICIR'''
        df1 = df.groupby('date').apply(cls.get_ic_rankic)
        df2 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        dura=(df.date.max()-df.date.min()).days/365
        t_value=df2.iloc[3,0]*(dura**(1/2))
        df3=pd.DataFrame({'评价指标':[t_value]},index=['RankIC均值t值'])
        df4=pd.concat([df2,df3])
        return df4

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def get_groups(cls, df, groups_num):
        '''依据因子值，判断是在第几组'''
        if 'group' in list(df.columns):
            df = df.drop(columns=['group'])
        df = df.sort_values(['fac'], ascending=True)
        each_group = round(df.shape[0] / groups_num)
        l = list(map(lambda x, y: [x] * y, list(range(1, groups_num + 1)), [each_group] * groups_num))
        l = reduce(lambda x, y: x + y, l)
        if len(l) < df.shape[0]:
            l = l + [groups_num] * (df.shape[0] - len(l))
        l = l[:df.shape[0]]
        df.insert(0, 'group', l)
        return df

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    # @history_remain(slogan='abandoned')
    def next_month_end(cls, x):
        '''找到下个月最后一天'''
        x1 = x = x + relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @classmethod
    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def limit_old_to_new(cls, limit, data):
        '''获取跌停股在旧月的组号，然后将日期调整到新月里
        涨停股则获得新月里涨停股的代码和时间，然后直接删去'''
        data1 = data.copy()
        data1 = data1.reset_index()
        data1.columns = ['data_index'] + list(data1.columns)[1:]
        old = pd.merge(limit, data1, how='inner', on=['date', 'code'])
        old = old.set_index('data_index')
        old = old[['group', 'date', 'code']]
        old.date = list(map(cls.next_month_end, list(old.date)))
        return old

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_data(self, groups_num):
        '''拼接因子数据和每月收益率数据，并对涨停和跌停股加以处理'''
        self.data = pd.merge(self.rets_monthly, self.factors, how='inner', on=['date', 'code'])
        self.ic_icir_and_rank = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby('date').apply(lambda x: self.get_groups(x, groups_num))
        self.data = self.data.reset_index(drop=True)
        limit_ups_object = self.limit_old_to_new(self.limit_ups, self.data)
        limit_downs_object = self.limit_old_to_new(self.limit_downs, self.data)
        self.data = self.data.drop(limit_ups_object.index)
        rets_monthly_limit_downs = pd.merge(self.rets_monthly, limit_downs_object, how='inner', on=['date', 'code'])
        self.data = pd.concat([self.data, rets_monthly_limit_downs])

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def select_data_time(self, time_start, time_end):
        '''筛选特定的时间段'''
        if time_start:
            self.data = self.data[self.data.date >= time_start]
        if time_end:
            self.data = self.data[self.data.date <= time_end]

    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def make_start_to_one(self, l):
        '''让净值序列的第一个数变成1'''
        min_date = self.factors.date.min()
        add_date = min_date - relativedelta(days=min_date.day)
        add_l = pd.Series([1], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    # @lru_cache(maxsize=None)
    @tool_box(slogan=None)
    def to_group_ret(self,l):
        '''每一组的年化收益率'''
        ret=l[-1]**(12/len(l))-1
        return ret

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_group_rets_net_values(self, groups_num=10,value_weighted=False):
        '''计算组内每一期的平均收益，生成每日收益率序列和净值序列'''
        if value_weighted:
            cap_value=self.pricloses_copy*self.flowshares_copy
            cap_value=cap_value.resample('M').last().shift(1)
            cap_value=cap_value*self.tris_monthly
            # cap_value=np.log(cap_value)
            cap_value=cap_value.stack().reset_index()
            cap_value.columns=['date','code','cap_value']
            self.data=pd.merge(self.data,cap_value,on=['date','code'])
            def in_g(df):
                df.cap_value=df.cap_value/df.cap_value.sum()
                df.ret=df.ret*df.cap_value
                return df.ret.sum()
            self.group_rets=self.data.groupby(['date','group']).apply(in_g)
        else:
            self.group_rets = self.data.groupby(['date', 'group']).apply(lambda x: x.ret.mean())
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[self.group_rets.index <= self.factors.date.max()]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix('group')
        self.long_short_rets = self.group_rets['group1'] - self.group_rets['group' + str(groups_num)]
        self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = self.group_rets['group' + str(groups_num)] - self.group_rets['group1']
            self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        self.long_short_net_values = self.make_start_to_one(self.long_short_net_values)
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.applymap(lambda x: x + 1)
        self.group_net_values = self.group_net_values.cumprod()
        self.group_net_values = self.group_net_values.apply(self.make_start_to_one)
        a=groups_num**(0.5)
        #判断是否要两个因子画表格
        if a==int(a):
            self.square_rets=self.group_net_values.iloc[:,:-1].apply(self.to_group_ret).to_numpy()
            self.square_rets=self.square_rets.reshape((int(a),int(a)))
            self.square_rets=pd.DataFrame(self.square_rets,columns=list(range(1,int(a)+1)),index=list(range(1,int(a)+1)))
            print('这是self.square_rets',self.square_rets)

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_long_short_comments(self,on_paper=False):
        '''计算多空对冲的相关评价指标
        包括年化收益率、年化波动率、信息比率、月度胜率、最大回撤率'''
        self.long_short_ret_yearly = self.long_short_net_values[-1] ** (12 / len(self.long_short_net_values)) - 1
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (12 ** 0.5)
        self.long_short_info_ratio = self.long_short_ret_yearly / self.long_short_vol_yearly
        self.long_short_win_times = len(self.long_short_rets[self.long_short_rets > 0])
        self.long_short_win_ratio = self.long_short_win_times / len(self.long_short_rets)
        self.max_retreat = -(self.long_short_net_values/self.long_short_net_values.expanding(1).max()-1).min()
        if on_paper:
            self.long_short_comments=pd.DataFrame({
                '评价指标':[
                    self.long_short_ret_yearly,
                    self.long_short_vol_yearly,
                    self.long_short_info_ratio,
                    self.long_short_win_ratio,
                    self.max_retreat
                ]
            },index=['年化收益率','年化波动率','收益波动比','月度胜率','最大回撤率'])
        else:
            self.long_short_comments = pd.DataFrame({
                '评价指标': [
                    self.long_short_ret_yearly,
                    self.long_short_vol_yearly,
                    self.long_short_info_ratio,
                    self.long_short_win_ratio,
                    self.max_retreat
                ]
            }, index=['年化收益率', '年化波动率', '信息比率', '月度胜率', '最大回撤率'])

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def get_total_comments(self):
        '''综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、月度胜率、最大回撤率'''
        self.total_comments = pd.concat([self.ic_icir_and_rank, self.long_short_comments])

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def plot_net_values(self, y2, filename):
        '''使用matplotlib来画图，y2为是否对多空组合采用双y轴'''
        self.group_net_values.plot(secondary_y=y2)
        filename_path = filename + '.png'
        if not NO_SAVE:
            plt.savefig(filename_path)

    # @lru_cache(maxsize=None)
    @main_process(slogan=None)
    def plotly_net_values(self, filename):
        '''使用plotly.express画图'''
        fig = pe.line(self.group_net_values)
        filename_path = filename + '.html'
        pio.write_html(fig, filename_path, auto_open=True)

    @classmethod
    @main_process(slogan=None)
    @lru_cache(maxsize=None)
    def prerpare(cls):
        '''通用数据准备'''
        cls.col_and_index()
        cls.load_all_files()
        cls.judge_month()
        cls.get_rets_month()

    # @lru_cache(maxsize=None)
    @kk.desktop_sender(title='嘿，回测结束啦～🗓')
    def run(self, groups_num=10, neutralize=False, boxcox=False, value_weighted=False,y2=False, plt_plot=True,
            plotly_plot=False, filename='分组净值图',
            time_start=None, time_end=None, print_comments=True,comments_writer=None,net_values_writer=None,rets_writer=None,
            comments_sheetname=None,net_values_sheetname=None,rets_sheetname=None,on_paper=False,sheetname=None):
        '''运行回测部分'''
        if comments_writer and not (comments_sheetname or sheetname):
            raise IOError('把total_comments输出到excel中时，必须指定sheetname🤒')
        if net_values_writer and not (net_values_sheetname or sheetname):
            raise IOError('把group_net_values输出到excel中时，必须指定sheetname🤒')
        if rets_writer and not (rets_sheetname or sheetname):
            raise IOError('把group_rets输出到excel中时，必须指定sheetname🤒')
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
        self.get_group_rets_net_values(groups_num=groups_num,value_weighted=value_weighted)
        self.get_long_short_comments(on_paper=on_paper)
        self.get_total_comments()
        if plt_plot:
            if not NO_PLOT:
                if filename:
                    self.plot_net_values(y2=y2, filename=filename)
                else:
                    self.plot_net_values(y2=y2,
                                         filename=self.factors_file.split('.')[-2].split('/')[-1] + str(groups_num) + '分组')
                plt.show()
        if plotly_plot:
            if not NO_PLOT:
                if filename:
                    self.plotly_net_values(filename=filename)
                else:
                    self.plotly_net_values(
                        filename=self.factors_file.split('.')[-2].split('/')[-1] + str(groups_num) + '分组')
        if print_comments:
            if not NO_COMMENT:
                print(self.total_comments)
        if sheetname:
            if comments_writer:
                total_comments=self.total_comments.copy()
                tc=list(total_comments.评价指标)
                tc[0]=str(round(tc[0]*100,2))+'%'
                tc[1]=str(round(tc[1],2))
                tc[2]=str(round(tc[2]*100,2))+'%'
                tc[3]=str(round(tc[3],2))
                tc[4]=str(round(tc[4],2))
                tc[5]=str(round(tc[5]*100,2))+'%'
                tc[6]=str(round(tc[6]*100,2))+'%'
                tc[7]=str(round(tc[7],2))
                tc[8]=str(round(tc[8]*100,2))+'%'
                tc[9]=str(round(tc[9]*100,2))+'%'
                new_total_comments=pd.DataFrame({sheetname:tc},index=total_comments.index)
                new_total_comments.T.to_excel(comments_writer,sheet_name=sheetname)
            if net_values_writer:
                groups_net_values=self.group_net_values.copy()
                groups_net_values.index=groups_net_values.index.strftime('%Y/%m/%d')
                groups_net_values.columns=[f'分组{i}' for i in range(1,len(list(groups_net_values.columns)))]+['多空对冲（右轴）']
                groups_net_values.to_excel(net_values_writer,sheet_name=sheetname)
            if rets_writer:
                group_rets=self.group_rets.copy()
                group_rets.index=group_rets.index.strftime('%Y/%m/%d')
                group_rets.columns=[f'分组{i}' for i in range(1,len(list(group_rets.columns)))]+['多空对冲（右轴）']
                group_rets.to_excel(rets_writer,sheet_name=sheetname)
        else:
            if comments_writer and comments_sheetname:
                total_comments=self.total_comments.copy()
                tc=list(total_comments.评价指标)
                tc[0]=str(round(tc[0]*100,2))+'%'
                tc[1]=str(round(tc[1],2))
                tc[2]=str(round(tc[2]*100,2))+'%'
                tc[3]=str(round(tc[3],2))
                tc[4]=str(round(tc[4],2))
                tc[5]=str(round(tc[5]*100,2))+'%'
                tc[6]=str(round(tc[6]*100,2))+'%'
                tc[7]=str(round(tc[7],2))
                tc[8]=str(round(tc[8]*100,2))+'%'
                tc[9]=str(round(tc[9]*100,2))+'%'
                new_total_comments=pd.DataFrame({comments_sheetname:tc},index=total_comments.index)
                new_total_comments.T.to_excel(comments_writer,sheet_name=comments_sheetname)
            if net_values_writer and net_values_sheetname:
                groups_net_values=self.group_net_values.copy()
                groups_net_values.index=groups_net_values.index.strftime('%Y/%m/%d')
                groups_net_values.columns=[f'分组{i}' for i in range(1,len(list(groups_net_values.columns)))]+['多空对冲（右轴）']
                groups_net_values.to_excel(net_values_writer,sheet_name=net_values_sheetname)
            if rets_writer and rets_sheetname:
                group_rets=self.group_rets.copy()
                group_rets.index=group_rets.index.strftime('%Y/%m/%d')
                group_rets.columns=[f'分组{i}' for i in range(1,len(list(group_rets.columns)))]+['多空对冲（右轴）']
                group_rets.to_excel(rets_writer,sheet_name=rets_sheetname)


class pure_fall():
    def __init__(
            self,
            minute_files_path=None,
            minute_columns=['date', 'open', 'high', 'low', 'close', 'amount', 'money'],
            daily_factors_path=None,
            monthly_factors_path=None
    ):
        self.homeplace=HomePlace()
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path = minute_files_path
        else:
            self.minute_files_path=self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files = os.listdir(self.minute_files_path)
        self.minute_files = [i for i in self.minute_files if i.endswith('.mat')]
        self.minute_files=sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns = minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list = []
        #更新数据用的列表
        self.daily_factors_list_update=[]
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        # 最终月度因子表格
        self.monthly_factors = None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path = daily_factors_path
        else:
            self.daily_factors_path=self.homeplace.factor_data_file+'日频_'
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path = monthly_factors_path
        else:
            self.monthly_factors_path=self.homeplace.factor_data_file+'月频_'

    def __call__(self,monthly=False):
        '''为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表'''
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        '''将几个因子截面标准化之后，因子值相加'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1+i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __truediv__(self, selfa):
        '''两个一正一副的因子，可以用此方法相减'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac=fac1-fac2
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __floordiv__(self, selfa):
        '''两个因子一正一负，可以用此方法相除'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1=fac1-fac1.min()
        fac2=fac2-fac2.min()
        fac=fac1/fac2
        fac=fac.replace(np.inf,np.nan)
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    @kk.desktop_sender(title='嘿，正交化结束啦～🐬')
    def __sub__(self, selfa):
        '''用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可'''
        tqdm.tqdm.pandas()
        if not isinstance(selfa,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfa} is changed into Iterable')
            selfa=(selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, 'fac')
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ['help' + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby('date').progress_apply(lambda x: self.de_in_group(x, help_names))
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def __rshift__(self, selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def wide_to_long(self, df, i):
        '''将宽数据转化为长数据，用于因子表转化和拼接'''
        df = df.stack().reset_index()
        df.columns = ['date', 'code', i]
        df = df.set_index(['date', 'code'])
        return df

    def de_in_group(self, df, help_names):
        '''对每个时间，分别做回归，剔除相关因子'''
        ols_order = 'fac~' + '+'.join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params['Intercept']
        df = df[['fac']]
        return df

    def mat_to_df(self, mat,use_datetime=True):
        '''将mat文件变成'''
        mat_path = '/'.join([self.minute_files_path, mat])
        df = list(scio.loadmat(mat_path).values())[3]
        df = pd.DataFrame(df, columns=self.minute_columns)
        if use_datetime:
            df.date = pd.to_datetime(df.date.apply(str), format='%Y%m%d')
            df = df.set_index('date')
        return df

    def add_suffix(self, code):
        '''给股票代码加上后缀'''
        if not isinstance(code, str):
            code = str(code)
        if len(code) < 6:
            code = '0' * (6 - len(code)) + code
        if code.startswith('0') or code.startswith('3'):
            code = '.'.join([code, 'SZ'])
        elif code.startswith('6'):
            code = '.'.join([code, 'SH'])
        elif code.startswith('8'):
            code = '.'.join([code, 'BJ'])
        return code

    def minute_to_daily(self, func,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        
        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files,desc='来日纵使千千阙歌，飘于远方我路上；来日纵使千千晚星，亮过今晚月亮。都不及今宵这刻美丽🌙'):
                try:
                    code = self.add_suffix(mat[-10:-4])
                    self.code=code
                    df = self.mat_to_df(mat,use_datetime=True)
                    if add_tr:
                        share=read_daily('AllStock_DailyAShareNum.mat')
                        share_this=share[code].to_frame('sharenum').reset_index()
                        share_this.columns=['date','sharenum']
                        df=df.reset_index()
                        df.columns=['date']+list(df.columns)[1:]
                        df=pd.merge(df,share_this,on=['date'],how='left')
                        df=df.assign(tr=df.amount/df.sharenum)
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df.date=df.date.dt.strftime('%Y%m%d')
                    df.date=df.date.astype(int)
                    df=df[(df.date>=start_date)&(df.date<=end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    priclose=df.groupby('date').last()
                    priclose=priclose.shift(1).reset_index()
                    df=pd.concat([priclose,df])
                    the_func = partial(func)
                    df = df.groupby('date').apply(the_func)
                    df = df.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(df)
                    else:
                        self.daily_factors_list_update.append(df)
                except Exception as e:
                    if not NO_LOG:
                        logger.warning(f'{code} 缺失')
                        logger.error(e)
        else:
            for mat in tqdm.tqdm(self.minute_files,desc='来日纵使千千阙歌，飘于远方我路上；来日纵使千千晚星，亮过今晚月亮。都不及今宵这刻美丽🌙'):
                try:
                    code = self.add_suffix(mat[-10:-4])
                    self.code=code
                    df = self.mat_to_df(mat,use_datetime=True)
                    if add_tr:
                        share=read_daily('AllStock_DailyAShareNum.mat')
                        share_this=share[code].to_frame('sharenum').reset_index()
                        share_this.columns=['date','sharenum']
                        df=df.reset_index()
                        df.columns=['date']+list(df.columns)[1:]
                        df=pd.merge(df,share_this,on=['date'],how='left')
                        df=df.assign(tr=df.amount/df.sharenum)
                    the_func = partial(func)
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df.date=df.date.dt.strftime('%Y%m%d')
                    df.date=df.date.astype(int)
                    df=df[(df.date>=start_date)&(df.date<=end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    df = df.groupby('date').apply(the_func)
                    df = df.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(df)
                    else:
                        self.daily_factors_list_update.append(df)
                except Exception as e:
                    if not NO_LOG:
                        logger.warning(f'{code} 缺失')
                        logger.error(e)
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors_update.index=pd.to_datetime(self.daily_factors_update.index.astype(int),format='%Y%m%d')
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
            self.daily_factors.index=pd.to_datetime(self.daily_factors.index.astype(int),format='%Y%m%d')
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def minute_to_daily_whole(self, func,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        for mat in tqdm.tqdm(self.minute_files):
            self.code=self.add_suffix(mat[-10:-4])
            df = self.mat_to_df(mat)
            df.date=df.date.astype(int)
            df=df[(df.date>=start_date)&(df.date<=end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func = partial(func)
            df = func(df)
            if isinstance(df,pd.DataFrame):
                df.columns = [self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df,pd.Series):
                df=df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not NO_LOG:
                    logger.warning(f'df is {df}')
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def standardlize_in_cross_section(self, df):
        '''
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        '''
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    @kk.desktop_sender(title='嘿，分钟数据处理完啦～🎈')
    def get_daily_factors(self, func, whole=False,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000):
        '''调用分钟到日度方法，算出日频数据'''
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index('date')
            now_minute_data=self.mat_to_df(self.minute_files[0])
            if self.daily_factors.index.max()<now_minute_data.index.max():
                if not NO_LOG:
                    logger.info(f'上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data.index.max()}，开始更新……')
                start_date_update=int(datetime.datetime.strftime(self.daily_factors.index.max()+pd.Timedelta('1 day'),'%Y%m%d'))
                end_date_update=int(datetime.datetime.strftime(now_minute_data.index.max(),'%Y%m%d'))
                if whole:
                    self.minute_to_daily_whole(func,start_date=start_date_update,end_date=end_date_update,update=1)
                else:
                    self.minute_to_daily(func,start_date=start_date_update,end_date=end_date_update,update=1)
        except Exception:
            if whole:
                self.minute_to_daily_whole(func,start_date=start_date,end_date=end_date)
            else:
                self.minute_to_daily(func,add_priclose=add_priclose,add_tr=add_tr,start_date=start_date,end_date=end_date)

    def get_neutral_monthly_factors(self, df,boxcox=False):
        '''对月度因子做市值中性化处理'''
        shen = pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5, boxcox=True, plt=False, print_comments=False)
        else:
            shen.run(5,neutralize=True,plt=False,print_comments=False)
        new_factors = shen.factors.copy()
        new_factors = new_factors.set_index(['date', 'code']).unstack()
        new_factors.columns = list(map(lambda x: x[1], list(new_factors.columns)))
        new_factors = new_factors.reset_index()
        add_start_point = new_factors.date.min()
        add_start_point = add_start_point - pd.Timedelta(days=add_start_point.day)
        new_factors.date = new_factors.date.shift(1)
        new_factors.date = new_factors.date.fillna(add_start_point)
        new_factors = new_factors.set_index('date')
        return new_factors

    def get_monthly_factors(self, func, neutralize,boxcox):
        '''将日频的因子转化为月频因子'''
        two_parts=self.monthly_factors_path.split('.')
        try:
            self.monthly_factors = pd.read_feather(self.monthly_factors_path)
            self.monthly_factors = self.monthly_factors.set_index('date')
        except Exception:
            the_func = partial(func)
            self.monthly_factors = the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors = self.get_neutral_monthly_factors(self.monthly_factors)
            elif boxcox:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors,boxcox=True)

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(self, whole=False,daily_func=None, monthly_func=None,
            neutralize=False,boxcox=False):
        '''执必要的函数，将分钟数据变成月度因子'''
        self.get_daily_factors(daily_func,whole)
        self.get_monthly_factors(monthly_func, neutralize,boxcox)


class pure_sunbath():
    def __init__(
            self,
            minute_files_path=None,
            minute_columns=['date','open','high','low','close','amount','money'],
            daily_factors_path=None,
            monthly_factors_path=None
    ):
        self.homeplace=HomePlace()
        print('在浴室的温暖里，连错误也不可怕；在阳光的照耀下，黑暗将无所遁形。')
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path=minute_files_path
        else:
            self.minute_files_path=self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files=os.listdir(self.minute_files_path)
        self.minute_files=[i for i in self.minute_files if i.endswith('.mat')]
        self.minute_files=sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns=minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list=[]
        #更新数据用的列表
        self.daily_factors_list_update=[]
        # 将分钟数据拼成一张日频因子表
        self.daily_factors=None
        # 最终月度因子表格
        self.monthly_factors=None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path=daily_factors_path
        else:
            self.daily_factors_path=self.homeplace.factor_data_file+'日频_'
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path=monthly_factors_path
        else:
            self.monthly_factors_path=self.homeplace.factor_data_file+'月频_'

    def __call__(self,monthly=False):
        '''为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表'''
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self,selfas):
        '''将几个因子截面标准化之后，因子值相加'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1+i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __truediv__(self,selfa):
        '''两个一正一副的因子，可以用此方法相减'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
        fac=fac1-fac2
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __floordiv__(self,selfa):
        '''两个因子一正一负，可以用此方法相除'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1=fac1-fac1.min()
        fac2=fac2-fac2.min()
        fac=fac1/fac2
        fac=fac.replace(np.inf,np.nan)
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __sub__(self,selfa):
        '''用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可'''
        tqdm.tqdm.pandas()
        if not isinstance(selfa,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfa} is changed into Iterable')
            selfa=(selfa,)
        fac_main=self.wide_to_long(self.monthly_factors,'fac')
        fac_helps=[i.monthly_factors for i in selfa]
        help_names=['help'+str(i) for i in range(1,(len(fac_helps)+1))]
        fac_helps=list(map(self.wide_to_long,fac_helps,help_names))
        fac_helps=pd.concat(fac_helps,axis=1)
        facs=pd.concat([fac_main,fac_helps],axis=1).dropna()
        facs=facs.groupby('date').progress_apply(lambda x:self.de_in_group(x,help_names))
        facs=facs.unstack()
        facs.columns=list(map(lambda x:x[1],list(facs.columns)))
        return facs

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def __rshift__(self,selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def wide_to_long(self,df,i):
        '''将宽数据转化为长数据，用于因子表转化和拼接'''
        df=df.stack().reset_index()
        df.columns=['date','code',i]
        df=df.set_index(['date','code'])
        return df

    def de_in_group(self,df,help_names):
        '''对每个时间，分别做回归，剔除相关因子'''
        ols_order='fac~'+'+'.join(help_names)
        ols_result=smf.ols(ols_order,data=df).fit()
        params={i:ols_result.params[i] for i in help_names}
        predict=[params[i]*df[i] for i in help_names]
        predict=reduce(lambda x,y:x+y,predict)
        df.fac=df.fac-predict-ols_result.params['Intercept']
        df=df[['fac']]
        return df

    def mat_to_df(self,mat,use_datetime=True):
        '''将mat文件变成'''
        mat_path='/'.join([self.minute_files_path,mat])
        df=list(scio.loadmat(mat_path).values())[3]
        df=pd.DataFrame(df,columns=self.minute_columns)
        if use_datetime:
            df.date=pd.to_datetime(df.date.apply(str),format='%Y%m%d')
            df=df.set_index('date')
        return df

    def add_suffix(self,code):
        '''给股票代码加上后缀'''
        if not isinstance(code,str):
            code=str(code)
        if len(code)<6:
            code='0'*(6-len(code))+code
        if code.startswith('0') or code.startswith('3'):
            code='.'.join([code,'SZ'])
        elif code.startswith('6'):
            code='.'.join([code,'SH'])
        elif code.startswith('8'):
            code='.'.join([code,'BJ'])
        return code

    def minute_to_daily(self,func,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''

        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files):
                # try:
                code=self.add_suffix(mat[-10:-4])
                self.code=code
                df=self.mat_to_df(mat,use_datetime=True)
                if add_tr:
                    share=read_daily('AllStock_DailyAShareNum.mat')
                    share_this=share[code].to_frame('sharenum').reset_index()
                    share_this.columns=['date','sharenum']
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df=pd.merge(df,share_this,on=['date'],how='left')
                    df=df.assign(tr=df.amount/df.sharenum)
                df=df.reset_index()
                df.columns=['date']+list(df.columns)[1:]
                df.date=df.date.dt.strftime('%Y%m%d')
                df.date=df.date.astype(int)
                df=df[(df.date>=start_date)&(df.date<=end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                priclose=df.groupby('date').last()
                priclose=priclose.shift(1).reset_index()
                df=pd.concat([priclose,df])
                the_func=partial(func)
                df=df.groupby('date').apply(the_func)
                df=df.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
                # except Exception as e:
                #     if not NO_LOG:
                #         logger.warning(f'{code} 缺失')
                #         logger.error(e)
        else:
            for mat in tqdm.tqdm(self.minute_files):
                # try:
                code=self.add_suffix(mat[-10:-4])
                self.code=code
                df=self.mat_to_df(mat,use_datetime=True)
                if add_tr:
                    share=read_daily('AllStock_DailyAShareNum.mat')
                    share_this=share[code].to_frame('sharenum').reset_index()
                    share_this.columns=['date','sharenum']
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df=pd.merge(df,share_this,on=['date'],how='left')
                    df=df.assign(tr=df.amount/df.sharenum)
                the_func=partial(func)
                df=df.reset_index()
                df.columns=['date']+list(df.columns)[1:]
                df.date=df.date.dt.strftime('%Y%m%d')
                df.date=df.date.astype(int)
                df=df[(df.date>=start_date)&(df.date<=end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                df=df.groupby('date').apply(the_func)
                df=df.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
                # except Exception as e:
                #     if not NO_LOG:
                #         logger.warning(f'{code} 缺失')
                #         logger.error(e)
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors_update.index=pd.to_datetime(self.daily_factors_update.index.astype(int),format='%Y%m%d')
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
            self.daily_factors.index=pd.to_datetime(self.daily_factors.index.astype(int),format='%Y%m%d')
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def minute_to_daily_whole(self,func,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        for mat in tqdm.tqdm(self.minute_files):
            self.code=self.add_suffix(mat[-10:-4])
            df=self.mat_to_df(mat)
            df.date=df.date.astype(int)
            df=df[(df.date>=start_date)&(df.date<=end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func=partial(func)
            df=func(df)
            if isinstance(df,pd.DataFrame):
                df.columns=[self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df,pd.Series):
                df=df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not NO_LOG:
                    logger.warning(f'df is {df}')
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def standardlize_in_cross_section(self,df):
        '''
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        '''
        df=df.T
        df=(df-df.mean())/df.std()
        df=df.T
        return df

    def get_daily_factors(self,func,whole=False,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000):
        '''调用分钟到日度方法，算出日频数据'''
        try:
            self.daily_factors=pd.read_feather(self.daily_factors_path)
            self.daily_factors=self.daily_factors.set_index('date')
            now_minute_data=self.mat_to_df(self.minute_files[0])
            if self.daily_factors.index.max()<now_minute_data.index.max():
                if not NO_LOG:
                    logger.info(
                        f'上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data.index.max()}，开始更新……')
                start_date_update=int(
                    datetime.datetime.strftime(self.daily_factors.index.max()+pd.Timedelta('1 day'),'%Y%m%d'))
                end_date_update=int(datetime.datetime.strftime(now_minute_data.index.max(),'%Y%m%d'))
                if whole:
                    self.minute_to_daily_whole(func,start_date=start_date_update,end_date=end_date_update,update=1)
                else:
                    self.minute_to_daily(func,start_date=start_date_update,end_date=end_date_update,update=1)
        except Exception:
            if whole:
                self.minute_to_daily_whole(func,start_date=start_date,end_date=end_date)
            else:
                self.minute_to_daily(func,add_priclose=add_priclose,add_tr=add_tr,start_date=start_date,
                                     end_date=end_date)

    def get_neutral_monthly_factors(self,df,boxcox=False):
        '''对月度因子做市值中性化处理'''
        shen=pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5,boxcox=True,plt=False,print_comments=False)
        else:
            shen.run(5,neutralize=True,plt=False,print_comments=False)
        new_factors=shen.factors.copy()
        new_factors=new_factors.set_index(['date','code']).unstack()
        new_factors.columns=list(map(lambda x:x[1],list(new_factors.columns)))
        new_factors=new_factors.reset_index()
        add_start_point=new_factors.date.min()
        add_start_point=add_start_point-pd.Timedelta(days=add_start_point.day)
        new_factors.date=new_factors.date.shift(1)
        new_factors.date=new_factors.date.fillna(add_start_point)
        new_factors=new_factors.set_index('date')
        return new_factors

    def get_monthly_factors(self,func,neutralize,boxcox):
        '''将日频的因子转化为月频因子'''
        two_parts=self.monthly_factors_path.split('.')
        try:
            self.monthly_factors=pd.read_feather(self.monthly_factors_path)
            self.monthly_factors=self.monthly_factors.set_index('date')
        except Exception:
            the_func=partial(func)
            self.monthly_factors=the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors)
            elif boxcox:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors,boxcox=True)

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(self,whole=False,daily_func=None,monthly_func=None,
            neutralize=False,boxcox=False):
        '''执必要的函数，将分钟数据变成月度因子'''
        self.get_daily_factors(daily_func,whole)
        self.get_monthly_factors(monthly_func,neutralize,boxcox)


class run_away_with_me():
    def __init__(
            self,
            minute_files_path=None,
            minute_columns=['date','open','high','low','close','amount','money'],
            daily_factors_path=None,
            monthly_factors_path=None
    ):
        self.homeplace=HomePlace()
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path=minute_files_path
        else:
            self.minute_files_path=self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files=os.listdir(self.minute_files_path)
        self.minute_files=[i for i in self.minute_files if i.endswith('.mat')]
        self.minute_files=sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns=minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list=[]
        #更新数据用的列表
        self.daily_factors_list_update=[]
        # 将分钟数据拼成一张日频因子表
        self.daily_factors=None
        # 最终月度因子表格
        self.monthly_factors=None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path=daily_factors_path
        else:
            self.daily_factors_path=self.homeplace.factor_data_file+'日频_'
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path=monthly_factors_path
        else:
            self.monthly_factors_path=self.homeplace.factor_data_file+'月频_'

    def __call__(self,monthly=False):
        '''为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表'''
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self,selfas):
        '''将几个因子截面标准化之后，因子值相加'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1+i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __truediv__(self,selfa):
        '''两个一正一副的因子，可以用此方法相减'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
        fac=fac1-fac2
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __floordiv__(self,selfa):
        '''两个因子一正一负，可以用此方法相除'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1=fac1-fac1.min()
        fac2=fac2-fac2.min()
        fac=fac1/fac2
        fac=fac.replace(np.inf,np.nan)
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __sub__(self,selfa):
        '''用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可'''
        tqdm.tqdm.pandas()
        if not isinstance(selfa,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfa} is changed into Iterable')
            selfa=(selfa,)
        fac_main=self.wide_to_long(self.monthly_factors,'fac')
        fac_helps=[i.monthly_factors for i in selfa]
        help_names=['help'+str(i) for i in range(1,(len(fac_helps)+1))]
        fac_helps=list(map(self.wide_to_long,fac_helps,help_names))
        fac_helps=pd.concat(fac_helps,axis=1)
        facs=pd.concat([fac_main,fac_helps],axis=1).dropna()
        facs=facs.groupby('date').progress_apply(lambda x:self.de_in_group(x,help_names))
        facs=facs.unstack()
        facs.columns=list(map(lambda x:x[1],list(facs.columns)))
        return facs

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def __rshift__(self,selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def wide_to_long(self,df,i):
        '''将宽数据转化为长数据，用于因子表转化和拼接'''
        df=df.stack().reset_index()
        df.columns=['date','code',i]
        df=df.set_index(['date','code'])
        return df

    def de_in_group(self,df,help_names):
        '''对每个时间，分别做回归，剔除相关因子'''
        ols_order='fac~'+'+'.join(help_names)
        ols_result=smf.ols(ols_order,data=df).fit()
        params={i:ols_result.params[i] for i in help_names}
        predict=[params[i]*df[i] for i in help_names]
        predict=reduce(lambda x,y:x+y,predict)
        df.fac=df.fac-predict-ols_result.params['Intercept']
        df=df[['fac']]
        return df

    def mat_to_df(self,mat,use_datetime=True):
        '''将mat文件变成'''
        mat_path='/'.join([self.minute_files_path,mat])
        df=list(scio.loadmat(mat_path).values())[3]
        df=pd.DataFrame(df,columns=self.minute_columns)
        if use_datetime:
            df.date=pd.to_datetime(df.date.apply(str),format='%Y%m%d')
            df=df.set_index('date')
        return df

    def add_suffix(self,code):
        '''给股票代码加上后缀'''
        if not isinstance(code,str):
            code=str(code)
        if len(code)<6:
            code='0'*(6-len(code))+code
        if code.startswith('0') or code.startswith('3'):
            code='.'.join([code,'SZ'])
        elif code.startswith('6'):
            code='.'.join([code,'SH'])
        elif code.startswith('8'):
            code='.'.join([code,'BJ'])
        return code

    def minute_to_daily(self,func,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''

        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files):
                try:
                    code=self.add_suffix(mat[-10:-4])
                    self.code=code
                    df=self.mat_to_df(mat,use_datetime=True)
                    if add_tr:
                        share=read_daily('AllStock_DailyAShareNum.mat')
                        share_this=share[code].to_frame('sharenum').reset_index()
                        share_this.columns=['date','sharenum']
                        df=df.reset_index()
                        df.columns=['date']+list(df.columns)[1:]
                        df=pd.merge(df,share_this,on=['date'],how='left')
                        df=df.assign(tr=df.amount/df.sharenum)
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df.date=df.date.dt.strftime('%Y%m%d')
                    df.date=df.date.astype(int)
                    df=df[(df.date>=start_date)&(df.date<=end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    priclose=df.groupby('date').last()
                    priclose=priclose.shift(1).reset_index()
                    df=pd.concat([priclose,df])
                    the_func=partial(func)
                    date_sets=sorted(list(set(df.date)))
                    ress=[]
                    for i in range(len(date_sets)-19):
                        res=df[(df.date>=date_sets[i])&(df.date<=date_sets[i+19])]
                        res=the_func(res)
                        ress.append(res)
                    ress=pd.concat(ress)
                    if isinstance(ress,pd.DataFrame):
                        if 'date' in list(ress.columns):
                            ress=ress.set_index('date').iloc[:,0]
                        else:
                            ress=ress.iloc[:,0]
                    else:
                        ress=ress.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(ress)
                    else:
                        self.daily_factors_list_update.append(ress)
                except Exception as e:
                    if not NO_LOG:
                        logger.warning(f'{code} 缺失')
                        logger.error(e)
        else:
            for mat in tqdm.tqdm(self.minute_files):
                try:
                    code=self.add_suffix(mat[-10:-4])
                    self.code=code
                    df=self.mat_to_df(mat,use_datetime=True)
                    if add_tr:
                        share=read_daily('AllStock_DailyAShareNum.mat')
                        share_this=share[code].to_frame('sharenum').reset_index()
                        share_this.columns=['date','sharenum']
                        df=df.reset_index()
                        df.columns=['date']+list(df.columns)[1:]
                        df=pd.merge(df,share_this,on=['date'],how='left')
                        df=df.assign(tr=df.amount/df.sharenum)
                    the_func=partial(func)
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df.date=df.date.dt.strftime('%Y%m%d')
                    df.date=df.date.astype(int)
                    df=df[(df.date>=start_date)&(df.date<=end_date)]
                    # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                    date_sets=sorted(list(set(df.date)))
                    ress=[]
                    for i in range(len(date_sets)-19):
                        res=df[(df.date>=date_sets[i])&(df.date<=date_sets[i+19])]
                        res=the_func(res)
                        ress.append(res)
                    ress=pd.concat(ress)
                    if isinstance(ress,pd.DataFrame):
                        if 'date' in list(ress.columns):
                            ress=ress.set_index('date').iloc[:,0]
                        else:
                            ress=ress.iloc[:,0]
                    else:
                        ress=ress.to_frame(name=code)
                    if not update:
                        self.daily_factors_list.append(ress)
                    else:
                        self.daily_factors_list_update.append(ress)
                except Exception as e:
                    if not NO_LOG:
                        logger.warning(f'{code} 缺失')
                        logger.error(e)
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors_update.index=pd.to_datetime(self.daily_factors_update.index.astype(int),format='%Y%m%d')
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
            self.daily_factors.index=pd.to_datetime(self.daily_factors.index.astype(int),format='%Y%m%d')
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def minute_to_daily_whole(self,func,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        for mat in tqdm.tqdm(self.minute_files):
            self.code=self.add_suffix(mat[-10:-4])
            df=self.mat_to_df(mat)
            df.date=df.date.astype(int)
            df=df[(df.date>=start_date)&(df.date<=end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func=partial(func)
            df=func(df)
            if isinstance(df,pd.DataFrame):
                df.columns=[self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df,pd.Series):
                df=df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not NO_LOG:
                    logger.warning(f'df is {df}')
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def standardlize_in_cross_section(self,df):
        '''
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        '''
        df=df.T
        df=(df-df.mean())/df.std()
        df=df.T
        return df

    @kk.desktop_sender(title='嘿，多日分钟数据处理完啦～🐬')
    def get_daily_factors(self,func,whole=False,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000):
        '''调用分钟到日度方法，算出日频数据'''
        try:
            self.daily_factors=pd.read_feather(self.daily_factors_path)
            self.daily_factors=self.daily_factors.set_index('date')
            now_minute_data=self.mat_to_df(self.minute_files[0])
            if self.daily_factors.index.max()<now_minute_data.index.max():
                if not NO_LOG:
                    logger.info(
                        f'上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data.index.max()}，开始更新……')
                start_date_update=int(
                    datetime.datetime.strftime(self.daily_factors.index.max()+pd.Timedelta('1 day'),'%Y%m%d'))
                end_date_update=int(datetime.datetime.strftime(now_minute_data.index.max(),'%Y%m%d'))
                if whole:
                    self.minute_to_daily_whole(func,start_date=start_date_update,end_date=end_date_update,update=1)
                else:
                    self.minute_to_daily(func,start_date=start_date_update,end_date=end_date_update,update=1)
        except Exception:
            if whole:
                self.minute_to_daily_whole(func,start_date=start_date,end_date=end_date)
            else:
                self.minute_to_daily(func,add_priclose=add_priclose,add_tr=add_tr,start_date=start_date,
                                     end_date=end_date)

    def get_neutral_monthly_factors(self,df,boxcox=False):
        '''对月度因子做市值中性化处理'''
        shen=pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5,boxcox=True,plt=False,print_comments=False)
        else:
            shen.run(5,neutralize=True,plt=False,print_comments=False)
        new_factors=shen.factors.copy()
        new_factors=new_factors.set_index(['date','code']).unstack()
        new_factors.columns=list(map(lambda x:x[1],list(new_factors.columns)))
        new_factors=new_factors.reset_index()
        add_start_point=new_factors.date.min()
        add_start_point=add_start_point-pd.Timedelta(days=add_start_point.day)
        new_factors.date=new_factors.date.shift(1)
        new_factors.date=new_factors.date.fillna(add_start_point)
        new_factors=new_factors.set_index('date')
        return new_factors

    def get_monthly_factors(self,func,neutralize,boxcox):
        '''将日频的因子转化为月频因子'''
        two_parts=self.monthly_factors_path.split('.')
        try:
            self.monthly_factors=pd.read_feather(self.monthly_factors_path)
            self.monthly_factors=self.monthly_factors.set_index('date')
        except Exception:
            the_func=partial(func)
            self.monthly_factors=the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors)
            elif boxcox:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors,boxcox=True)

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(self,whole=False,daily_func=None,monthly_func=None,
            neutralize=False,boxcox=False):
        '''执必要的函数，将分钟数据变成月度因子'''
        self.get_daily_factors(daily_func,whole)
        self.get_monthly_factors(monthly_func,neutralize,boxcox)



class carry_me_to_bathroom():
    def __init__(
            self,
            minute_files_path=None,
            minute_columns=['date','open','high','low','close','amount','money'],
            daily_factors_path=None,
            monthly_factors_path=None
    ):
        self.homeplace=HomePlace()
        if monthly_factors_path:
            # 分钟数据文件夹路径
            self.minute_files_path=minute_files_path
        else:
            self.minute_files_path=self.homeplace.minute_data_file[:-1]
        # 分钟数据文件夹
        self.minute_files=os.listdir(self.minute_files_path)
        self.minute_files=[i for i in self.minute_files if i.endswith('.mat')]
        self.minute_files=sorted(self.minute_files)
        # 分钟数据的表头
        self.minute_columns=minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list=[]
        #更新数据用的列表
        self.daily_factors_list_update=[]
        # 将分钟数据拼成一张日频因子表
        self.daily_factors=None
        # 最终月度因子表格
        self.monthly_factors=None
        if daily_factors_path:
            # 日频因子文件保存路径
            self.daily_factors_path=daily_factors_path
        else:
            self.daily_factors_path=self.homeplace.factor_data_file+'日频_'
        if monthly_factors_path:
            # 月频因子文件保存路径
            self.monthly_factors_path=monthly_factors_path
        else:
            self.monthly_factors_path=self.homeplace.factor_data_file+'月频_'

    def __call__(self,monthly=False):
        '''为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表'''
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self,selfas):
        '''将几个因子截面标准化之后，因子值相加'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1+i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __truediv__(self,selfa):
        '''两个一正一副的因子，可以用此方法相减'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
        fac=fac1-fac2
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __floordiv__(self,selfa):
        '''两个因子一正一负，可以用此方法相除'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1=fac1-fac1.min()
        fac2=fac2-fac2.min()
        fac=fac1/fac2
        fac=fac.replace(np.inf,np.nan)
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __sub__(self,selfa):
        '''用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可'''
        tqdm.tqdm.pandas()
        if not isinstance(selfa,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfa} is changed into Iterable')
            selfa=(selfa,)
        fac_main=self.wide_to_long(self.monthly_factors,'fac')
        fac_helps=[i.monthly_factors for i in selfa]
        help_names=['help'+str(i) for i in range(1,(len(fac_helps)+1))]
        fac_helps=list(map(self.wide_to_long,fac_helps,help_names))
        fac_helps=pd.concat(fac_helps,axis=1)
        facs=pd.concat([fac_main,fac_helps],axis=1).dropna()
        facs=facs.groupby('date').progress_apply(lambda x:self.de_in_group(x,help_names))
        facs=facs.unstack()
        facs.columns=list(map(lambda x:x[1],list(facs.columns)))
        return facs

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def __rshift__(self,selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def wide_to_long(self,df,i):
        '''将宽数据转化为长数据，用于因子表转化和拼接'''
        df=df.stack().reset_index()
        df.columns=['date','code',i]
        df=df.set_index(['date','code'])
        return df

    def de_in_group(self,df,help_names):
        '''对每个时间，分别做回归，剔除相关因子'''
        ols_order='fac~'+'+'.join(help_names)
        ols_result=smf.ols(ols_order,data=df).fit()
        params={i:ols_result.params[i] for i in help_names}
        predict=[params[i]*df[i] for i in help_names]
        predict=reduce(lambda x,y:x+y,predict)
        df.fac=df.fac-predict-ols_result.params['Intercept']
        df=df[['fac']]
        return df

    def mat_to_df(self,mat,use_datetime=True):
        '''将mat文件变成'''
        mat_path='/'.join([self.minute_files_path,mat])
        df=list(scio.loadmat(mat_path).values())[3]
        df=pd.DataFrame(df,columns=self.minute_columns)
        if use_datetime:
            df.date=pd.to_datetime(df.date.apply(str),format='%Y%m%d')
            df=df.set_index('date')
        return df

    def add_suffix(self,code):
        '''给股票代码加上后缀'''
        if not isinstance(code,str):
            code=str(code)
        if len(code)<6:
            code='0'*(6-len(code))+code
        if code.startswith('0') or code.startswith('3'):
            code='.'.join([code,'SZ'])
        elif code.startswith('6'):
            code='.'.join([code,'SH'])
        elif code.startswith('8'):
            code='.'.join([code,'BJ'])
        return code

    def minute_to_daily(self,func,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''

        if add_priclose:
            for mat in tqdm.tqdm(self.minute_files):
                code=self.add_suffix(mat[-10:-4])
                self.code=code
                df=self.mat_to_df(mat,use_datetime=True)
                if add_tr:
                    share=read_daily('AllStock_DailyAShareNum.mat')
                    share_this=share[code].to_frame('sharenum').reset_index()
                    share_this.columns=['date','sharenum']
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df=pd.merge(df,share_this,on=['date'],how='left')
                    df=df.assign(tr=df.amount/df.sharenum)
                df=df.reset_index()
                df.columns=['date']+list(df.columns)[1:]
                df.date=df.date.dt.strftime('%Y%m%d')
                df.date=df.date.astype(int)
                df=df[(df.date>=start_date)&(df.date<=end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                priclose=df.groupby('date').last()
                priclose=priclose.shift(1).reset_index()
                df=pd.concat([priclose,df])
                the_func=partial(func)
                date_sets=sorted(list(set(df.date)))
                ress=[]
                for i in range(len(date_sets)-19):
                    res=df[(df.date>=date_sets[i])&(df.date<=date_sets[i+19])]
                    res=the_func(res)
                    ress.append(res)
                ress=pd.concat(ress)
                if isinstance(ress,pd.DataFrame):
                    if 'date' in list(ress.columns):
                        ress=ress.set_index('date').iloc[:,0]
                    else:
                        ress=ress.iloc[:,0]
                else:
                    ress=ress.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(ress)
                else:
                    self.daily_factors_list_update.append(ress)
        else:
            for mat in tqdm.tqdm(self.minute_files):
                code=self.add_suffix(mat[-10:-4])
                self.code=code
                df=self.mat_to_df(mat,use_datetime=True)
                if add_tr:
                    share=read_daily('AllStock_DailyAShareNum.mat')
                    share_this=share[code].to_frame('sharenum').reset_index()
                    share_this.columns=['date','sharenum']
                    df=df.reset_index()
                    df.columns=['date']+list(df.columns)[1:]
                    df=pd.merge(df,share_this,on=['date'],how='left')
                    df=df.assign(tr=df.amount/df.sharenum)
                the_func=partial(func)
                df=df.reset_index()
                df.columns=['date']+list(df.columns)[1:]
                df.date=df.date.dt.strftime('%Y%m%d')
                df.date=df.date.astype(int)
                df=df[(df.date>=start_date)&(df.date<=end_date)]
                # df.date=pd.to_datetime(df.date,format='%Y%m%d')
                date_sets=sorted(list(set(df.date)))
                ress=[]
                for i in range(len(date_sets)-19):
                    res=df[(df.date>=date_sets[i])&(df.date<=date_sets[i+19])]
                    res=the_func(res)
                    ress.append(res)
                ress=pd.concat(ress)
                if isinstance(ress,pd.DataFrame):
                    if 'date' in list(ress.columns):
                        ress=ress.set_index('date').iloc[:,0]
                    else:
                        ress=ress.iloc[:,0]
                else:
                    ress=ress.to_frame(name=code)
                if not update:
                    self.daily_factors_list.append(ress)
                else:
                    self.daily_factors_list_update.append(ress)
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors_update.index=pd.to_datetime(self.daily_factors_update.index.astype(int),format='%Y%m%d')
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
            self.daily_factors.index=pd.to_datetime(self.daily_factors.index.astype(int),format='%Y%m%d')
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def minute_to_daily_whole(self,func,start_date=10000000,end_date=30000000,update=0):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        for mat in tqdm.tqdm(self.minute_files):
            self.code=self.add_suffix(mat[-10:-4])
            df=self.mat_to_df(mat)
            df.date=df.date.astype(int)
            df=df[(df.date>=start_date)&(df.date<=end_date)]
            # df.date=pd.to_datetime(df.date,format='%Y%m%d')
            the_func=partial(func)
            df=func(df)
            if isinstance(df,pd.DataFrame):
                df.columns=[self.code]
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            elif isinstance(df,pd.Series):
                df=df.to_frame(name=self.code)
                if not update:
                    self.daily_factors_list.append(df)
                else:
                    self.daily_factors_list_update.append(df)
            else:
                if not NO_LOG:
                    logger.warning(f'df is {df}')
        if update:
            self.daily_factors_update=pd.concat(self.daily_factors_list_update,axis=1)
            self.daily_factors=pd.concat([self.daily_factors,self.daily_factors_update])
        else:
            self.daily_factors=pd.concat(self.daily_factors_list,axis=1)
        self.daily_factors=self.daily_factors.dropna(how='all')
        self.daily_factors=self.daily_factors[self.daily_factors.index>=pd.Timestamp('2013-03-26')]
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)
        if not NO_LOG:
            logger.success('更新已完成')

    def standardlize_in_cross_section(self,df):
        '''
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        '''
        df=df.T
        df=(df-df.mean())/df.std()
        df=df.T
        return df

    def get_daily_factors(self,func,whole=False,add_priclose=False,add_tr=False,start_date=10000000,end_date=30000000):
        '''调用分钟到日度方法，算出日频数据'''
        try:
            self.daily_factors=pd.read_feather(self.daily_factors_path)
            self.daily_factors=self.daily_factors.set_index('date')
            now_minute_data=self.mat_to_df(self.minute_files[0])
            if self.daily_factors.index.max()<now_minute_data.index.max():
                if not NO_LOG:
                    logger.info(
                        f'上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data.index.max()}，开始更新……')
                start_date_update=int(
                    datetime.datetime.strftime(self.daily_factors.index.max()+pd.Timedelta('1 day'),'%Y%m%d'))
                end_date_update=int(datetime.datetime.strftime(now_minute_data.index.max(),'%Y%m%d'))
                if whole:
                    self.minute_to_daily_whole(func,start_date=start_date_update,end_date=end_date_update,update=1)
                else:
                    self.minute_to_daily(func,start_date=start_date_update,end_date=end_date_update,update=1)
        except Exception:
            if whole:
                self.minute_to_daily_whole(func,start_date=start_date,end_date=end_date)
            else:
                self.minute_to_daily(func,add_priclose=add_priclose,add_tr=add_tr,start_date=start_date,
                                     end_date=end_date)

    def get_neutral_monthly_factors(self,df,boxcox=False):
        '''对月度因子做市值中性化处理'''
        shen=pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5,boxcox=True,plt=False,print_comments=False)
        else:
            shen.run(5,neutralize=True,plt=False,print_comments=False)
        new_factors=shen.factors.copy()
        new_factors=new_factors.set_index(['date','code']).unstack()
        new_factors.columns=list(map(lambda x:x[1],list(new_factors.columns)))
        new_factors=new_factors.reset_index()
        add_start_point=new_factors.date.min()
        add_start_point=add_start_point-pd.Timedelta(days=add_start_point.day)
        new_factors.date=new_factors.date.shift(1)
        new_factors.date=new_factors.date.fillna(add_start_point)
        new_factors=new_factors.set_index('date')
        return new_factors

    def get_monthly_factors(self,func,neutralize,boxcox):
        '''将日频的因子转化为月频因子'''
        two_parts=self.monthly_factors_path.split('.')
        try:
            self.monthly_factors=pd.read_feather(self.monthly_factors_path)
            self.monthly_factors=self.monthly_factors.set_index('date')
        except Exception:
            the_func=partial(func)
            self.monthly_factors=the_func(self.daily_factors)
            if neutralize:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors)
            elif boxcox:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors,boxcox=True)

            self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(self,whole=False,daily_func=None,monthly_func=None,
            neutralize=False,boxcox=False):
        '''执必要的函数，将分钟数据变成月度因子'''
        self.get_daily_factors(daily_func,whole)
        self.get_monthly_factors(monthly_func,neutralize,boxcox)


class pure_fallmount(pure_fall):
    '''继承自父类，专为做因子截面标准化之后相加和因子剔除其他辅助因子的作用'''
    def __init__(self, monthly_factors):
        '''输入月度因子值，以设定新的对象'''
        super(pure_fall, self).__init__()
        self.monthly_factors = monthly_factors

    def __call__(self,monthly=False):
        '''为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表'''
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        '''返回一个对象，而非一个表格，如需表格请调用对象'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fallmount(fac1)
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        if not isinstance(selfas,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfas} is changed into Iterable')
            selfas=(selfas,)
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __sub__(self, selfa):
        '''返回对象，如需表格，请调用对象'''
        tqdm.tqdm.pandas()
        if not isinstance(selfa,Iterable):
            if not NO_LOG:
                logger.warning(f'{selfa} is changed into Iterable')
            selfa=(selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, 'fac')
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ['help' + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby('date').progress_apply(lambda x: self.de_in_group(x, help_names))
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        new_pure = pure_fallmount(facs)
        return new_pure

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fallmount(xy)
        return new_pure

    def __rshift__(self, selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fallmount(xy)
        return new_pure

class pure_winter():
    def __init__(self):
        self.homeplace=HomePlace()
        # barra因子数据
        self.barras = self.read_h5(self.homeplace.barra_data_file+'FactorLoading_Style.h5')

    def __call__(self,fallmount=0):
        '''返回纯净因子值'''
        if fallmount==0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)

    def read_h5(self,path):
        '''读入h5文件'''
        res={}
        a=h5py.File(path)
        for k,v in tqdm.tqdm(list(a.items()),desc='数据加载中……'):
            value=list(v.values())[-1]
            col=[i.decode('utf-8') for i in list(list(v.values())[0])]
            ind=[i.decode('utf-8') for i in list(list(v.values())[1])]
            res[k]=pd.DataFrame(value,columns=col,index=ind)
        return res

    @history_remain(slogan='abandoned')
    def last_month_end(self, x):
        '''找到下个月最后一天'''
        x1 = x = x - relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @history_remain(slogan='abandoned')
    def set_factors_df(self, df):
        '''传入因子dataframe，应为三列，第一列是时间，第二列是股票代码，第三列是因子值'''
        df1=df.copy()
        df1.columns = ['date', 'code', 'fac']
        df1 = df1.set_index(['date', 'code'])
        df1 = df1.unstack().reset_index()
        df1.date = df1.date.apply(self.last_month_end)
        df1 = df1.set_index(['date']).stack()
        self.factors = df1.copy()

    def set_factors_df_wide(self, df):
        '''传入因子数据，时间为索引，代码为列名'''
        df1=df.copy()
        # df1.index=df1.index-pd.DateOffset(months=1)
        df1=df1.resample('M').last()
        df1=df1.stack().reset_index()
        df1.columns = ['date', 'code','fac']
        self.factors=df1.copy()

    def daily_to_monthly(self, df):
        '''将日度的barra因子月度化'''
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df = df.resample('M').last()
        return df

    def get_monthly_barras_industrys(self):
        '''将barra因子和行业哑变量变成月度数据'''
        for key, value in self.barras.items():
            self.barras[key] = self.daily_to_monthly(value)

    def wide_to_long(self, df, name):
        '''将宽数据变成长数据，便于后续拼接'''
        df = df.stack().reset_index()
        df.columns = ['date', 'code', name]
        df = df.set_index(['date', 'code'])
        return df

    def get_wide_barras_industrys(self):
        '''将barra因子和行业哑变量都变成长数据'''
        for key, value in self.barras.items():
            self.barras[key] = self.wide_to_long(value, key)

    def get_corr_pri_ols_pri(self):
        '''拼接barra因子和行业哑变量，生成用于求相关系数和纯净因子的数据表'''
        if self.factors.shape[0]>1:
            self.factors=self.factors.set_index(['date', 'code'])
        self.corr_pri = pd.concat([self.factors] + list(self.barras.values()), axis=1).dropna()

    def get_corr(self):
        '''计算每一期的相关系数，再求平均值'''
        self.corr_by_step = self.corr_pri.groupby(['date']).apply(lambda x: x.corr().head(1))
        self.__corr = self.corr_by_step.mean()
        self.__corr.index=['因子自身','贝塔','估值','杠杆',
                         '盈利','成长','流动性','反转','波动率',
                         '市值','非线性市值']

    @property
    def corr(self):
        return self.__corr.copy()

    def ols_in_group(self, df):
        '''对每个时间段进行回归，并计算残差'''
        xs = list(df.columns)
        xs = [i for i in xs if i != 'fac']
        xs_join = '+'.join(xs)
        ols_formula = 'fac~' + xs_join
        ols_result = smf.ols(ols_formula, data=df).fit()
        ols_ws = {i: ols_result.params[i] for i in xs}
        ols_b = ols_result.params['Intercept']
        to_minus = [ols_ws[i] * df[i] for i in xs]
        to_minus = reduce(lambda x, y: x + y, to_minus)
        df = df.assign(snow_fac=df.fac - to_minus - ols_b)
        df = df[['snow_fac']]
        df = df.rename(columns={'snow_fac': 'fac'})
        return df

    def get_snow_fac(self):
        '''获得纯净因子'''
        self.snow_fac = self.corr_pri.groupby(['date']).apply(self.ols_in_group)
        self.snow_fac = self.snow_fac.unstack()
        self.snow_fac.columns = list(map(lambda x: x[1], list(self.snow_fac.columns)))

    def run(self):
        '''运行一些必要的函数'''
        self.get_monthly_barras_industrys()
        self.get_wide_barras_industrys()
        self.get_corr_pri_ols_pri()
        self.get_corr()
        self.get_snow_fac()


class pure_snowtrain(pure_winter):
    '''直接返回纯净因子'''

    def __init__(self, factors):
        '''直接输入原始因子数据'''
        super(pure_snowtrain, self).__init__()
        self.set_factors_df_wide(factors.copy())
        self.run()

    def __call__(self, fallmount=0):
        '''可以直接返回pure_fallmount对象，或纯净因子矩阵'''
        if fallmount == 0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)


class pure_moonlight(pure_moon):
    '''继承自pure_moon回测框架，使用其中的日频复权收盘价数据，以及换手率数据'''

    def __init__(self):
        '''加载全部数据'''
        super(pure_moonlight, self).__init__()
        self.homeplace=HomePlace()
        self.col_and_index()
        self.load_all_files()
        self.judge_month()
        self.get_log_cap()
        # 对数市值
        self.cap_as_factor = self.cap[['date', 'code', 'cap_size']].set_index(['date', 'code']).unstack()
        self.cap_as_factor.columns = list(map(lambda x: x[1], list(self.cap_as_factor.columns)))
        # 传统反转因子ret20
        self.ret20_database = self.homeplace.factor_data_file+'月频_反转因子ret20.feather'
        # 传统换手率因子turn20
        self.turn20_database = self.homeplace.factor_data_file+'月频_换手率因子turn20.feather'
        # 传统波动率因子vol20
        self.vol20_database = self.homeplace.factor_data_file+'月频_波动率因子vol20.feather'
        # #自动更新
        self.get_updated_factors()

    def __call__(self, name):
        '''可以通过call方式，直接获取对应因子数据'''
        value = getattr(self, name)
        return value

    def get_ret20(self, pri):
        '''计算20日涨跌幅因子'''
        past = pri.iloc[:-20, :]
        future = pri.iloc[20:, :]
        ret20 = (future.to_numpy() - past.to_numpy()) / past.to_numpy()
        df = pd.DataFrame(ret20, columns=pri.columns, index=future.index)
        df = df.resample('M').last()
        return df

    def get_turn20(self, pri):
        '''计算20换手率因子'''
        turns = pri.rolling(20).mean()
        turns = turns.resample('M').last().reset_index()
        turns.columns = ['date'] + list(turns.columns)[1:]
        self.factors = turns
        self.get_neutral_factors()
        df = self.factors.copy()
        df = df.set_index(['date', 'code'])
        df = df.unstack()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        return df

    def get_vol20(self, pri):
        '''计算20日波动率因子'''
        rets = pri.pct_change()
        vol = rets.rolling(20).apply(np.std)
        df = vol.resample('M').last()
        return df

    def update_single_factor_in_database(self, path, pri, func):
        '''
        用基础数据库更新因子数据库
        执行顺序为，先读取文件，如果没有，就直接全部计算，然后存储
        如果有，就读出来看看。数据是不是最新的
        如果不是最新的，就将原始数据在上次因子存储的日期处截断
        计算出新的一段时间的因子值，然后追加写入因子文件中
        '''
        the_func = partial(func)
        try:
            df = pd.read_feather(path)
            df.columns = ['date'] + list(df.columns)[1:]
            df = df.set_index('date')
            if df.index.max() < pri.index.max():
                to_add = pri[(pri.index > df.index.max()) & (pri.index <= pri.index.max())]
                to_add = the_func(to_add)
                df = pd.concat([df, to_add])
                print('之前数据有点旧了，已为您完成更新')
            else:
                print('数据很新，无需更新')
            df1 = df.reset_index()
            df1.columns = ['date'] + list(df1.columns)[1:]
            df1.to_feather(path)
            return df
        except Exception:
            df = the_func(pri)
            df1 = df.reset_index()
            df1.columns = ['date'] + list(df1.columns)[1:]
            df1.to_feather(path)
            print('新因子建库完成')
            return df

    def get_updated_factors(self):
        '''更新因子数据'''
        self.ret20 = self.update_single_factor_in_database(self.ret20_database, self.closes, self.get_ret20)
        self.turn20 = self.update_single_factor_in_database(self.turn20_database, self.turnovers, self.get_turn20)
        self.vol20 = self.update_single_factor_in_database(self.vol20_database, self.closes, self.get_vol20)


class pure_moonnight():
    '''封装选股框架'''
    __slots__ = ['shen']
    def __init__(self, factors, groups_num=10, neutralize=False, boxcox=False,by10=False, value_weighted=False,y2=False, plt_plot=True, plotly_plot=False,
                 filename='分组净值图', time_start=None, time_end=None, print_comments=True,comments_writer=None,net_values_writer=None,rets_writer=None,
            comments_sheetname=None,net_values_sheetname=None,rets_sheetname=None,on_paper=False,sheetname=None):
        '''直接输入因子数据'''
        if isinstance(factors,pure_fallmount):
            factors=factors().copy()
        self.shen=pure_moon()
        self.shen.set_factor_df_date_as_index(factors)
        self.shen.prerpare()
        self.shen.run(groups_num=groups_num,neutralize=neutralize,boxcox=boxcox,value_weighted=value_weighted,y2=y2,plt_plot=plt_plot,
                      plotly_plot=plotly_plot,filename=filename,time_start=time_start,time_end=time_end,print_comments=print_comments,
                      comments_writer=comments_writer,net_values_writer=net_values_writer,rets_writer=rets_writer,
                      comments_sheetname=comments_sheetname,net_values_sheetname=net_values_sheetname,
                      rets_sheetname=rets_sheetname,on_paper=on_paper,sheetname=sheetname)

    def __call__(self, fallmount=0):
        '''调用则返回因子数据'''
        if fallmount == 0:
            df=self.shen.factors_out.copy()
            # df=df.set_index(['date', 'code']).unstack()
            df.columns=list(map(lambda x:x[1],list(df.columns)))
            return df
        else:
            return pure_fallmount(df)

class pure_newyear():
    '''转为生成25分组和百分组的收益矩阵而封装'''

    def __init__(self,facx,facy,group_num_single,namex='主',namey='次'):
        '''初始化时即进行回测'''
        homex=pure_fallmount(facx)
        homey=pure_fallmount(facy)
        if group_num_single==5:
            homexy=homex>homey
        elif group_num_single==10:
            homexy=homex>>homey
        shen=pure_moonnight(homexy(),group_num_single**2,plt_plot=False,print_comments=False)
        sq=shen.shen.square_rets.copy()
        sq.index=[namex+str(i) for i in list(sq.index)]
        sq.columns=[namey+str(i) for i in list(sq.columns)]
        self.square_rets=sq

    def __call__(self):
        '''调用对象时，返回最终结果，正方形的分组年化收益率表'''
        return self.square_rets

class pure_dawn():
    '''
    因子切割论的母框架，可以对两个因子进行类似于因子切割的操作
    可用于派生任何"以两个因子生成一个因子"的子类
    使用举例
    cut函数里，必须带有输入变量df,df有两个columns，一个名为'fac1'，一个名为'fac2'，df是最近一个回看期内的数据
    class Cut(pure_dawn):
        def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(Cut,self).__init__(fac1=fac1,fac2=fac2)

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
    '''

    def __init__(self,fac1,fac2,*args):
        self.fac1=fac1
        self.fac1=self.fac1.stack().reset_index()
        self.fac1.columns=['date','code','fac1']
        self.fac2=fac2
        self.fac2=self.fac2.stack().reset_index()
        self.fac2.columns=['date','code','fac2']
        fac_all=pd.merge(self.fac1,self.fac2,on=['date','code'])
        for i,fac in enumerate(args):
            fac=fac.stack().reset_index()
            fac.columns=['date','code',f'fac{i+3}']
            fac_all=pd.merge(fac_all,fac,on=['date','code'])
        fac_all=fac_all.sort_values(['date','code'])
        self.fac=fac_all.copy()

    def __call__(self):
        '''返回最终月度因子值'''
        return self.fac

    def get_fac_long_and_tradedays(self):
        '''将两个因子的矩阵转化为长列表'''
        self.tradedays=sorted(list(set(self.fac.date)))

    def get_month_starts_and_ends(self,backsee=20):
        '''计算出每个月回看期间的起点日和终点日'''
        self.month_ends=[i for i,j in zip(self.tradedays[:-1],self.tradedays[1:]) if i.month!=j.month]
        self.month_ends.append(self.tradedays[-1])
        self.month_starts=[self.find_begin(self.tradedays,i,backsee=backsee) for i in self.month_ends]
        self.month_starts[0]=self.tradedays[0]

    def find_begin(self,tradedays,end_day,backsee=20):
        '''找出回看若干天的开始日，默认为20'''
        end_day_index=tradedays.index(end_day)
        start_day_index=end_day_index-backsee+1
        start_day=tradedays[start_day_index]
        return start_day

    def make_monthly_factors_single_code(self,df,func):
        '''
        对单一股票来计算月度因子
        func为单月执行的函数，返回值应为月度因子，如一个float或一个list
        df为一个股票的四列表，包含时间、代码、因子1和因子2
        '''
        res={}
        for start,end in zip(self.month_starts,self.month_ends):
            this_month=df[(df.date>=start)&(df.date<=end)]
            res[end]=func(this_month)
        dates=list(res.keys())
        corrs=list(res.values())
        part=pd.DataFrame({'date':dates,'corr':corrs})
        return part

    def get_monthly_factor(self,func):
        '''运行自己写的函数，获得月度因子'''
        tqdm.tqdm.pandas(desc='when the dawn comes, tonight will be a memory too.')
        self.fac=self.fac.groupby(['code']).progress_apply(lambda x:self.make_monthly_factors_single_code(x,func))
        self.fac=self.fac.reset_index(level=1,drop=True).reset_index().set_index(['date','code']).unstack()
        self.fac.columns=[i[1] for i in list(self.fac.columns)]
        self.fac=self.fac.resample('M').last()

    @kk.desktop_sender(title='嘿，切割完成啦🛁')
    def run(self,func,backsee=20):
        '''运行必要的函数'''
        self.get_fac_long_and_tradedays()
        self.get_month_starts_and_ends(backsee=backsee)
        self.get_monthly_factor(func)


class pure_cloud(object):
    '''
    为了测试其他不同的频率而设计的类，仅考虑了上市满60天这一要素
    这一回测采取的方案是，对于回测频率n天，将初始资金等分成n笔，每天以1/n的资金调仓
    每笔资金之间相互独立，最终汇聚成一个收益率序列
    '''
    def __init__(self,fac,freq,group=10,boxcox=1,trade_cost=0,print_comments=1,plt_plot=1,plotly_plot=0,
                 filename='净值走势图',comments_writer=None,nets_writer=None,sheet_name=None):
        '''n是回测的频率，等分成n份，group是回测的组数，boxcox是是否做行业市值中性化'''
        self.fac=fac
        self.freq=freq
        self.group=group
        self.boxcox=boxcox
        self.trade_cost=trade_cost
        moon=pure_moon()
        moon.prerpare()
        ages=moon.ages.copy()
        ages=(ages>=60)+0
        self.ages=ages.replace(0,np.nan)
        self.closes=read_daily(close=1)
        self.rets=((self.closes.shift(-self.freq)/self.closes-1)*self.ages)/self.freq
        self.run(print_comments=print_comments,plt_plot=plt_plot,plotly_plot=plotly_plot,filename=filename)
        if comments_writer:
            if sheet_name:
                self.long_short_comments.to_excel(comments_writer,sheet_name=sheet_name)
            else:
                raise AttributeError('必须制定sheet_name参数🤒')
        if nets_writer:
            if sheet_name:
                self.group_nets.to_excel(nets_writer,sheet_name=sheet_name)
            else:
                raise AttributeError('必须制定sheet_name参数🤒')

    def comments(self,series,series1):
        '''对twins中的结果给出评价
        评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率'''
        ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
        duration=(series.index[-1]-series.index[0]).days
        year=duration/365
        ret_yearly=(series.iloc[-1]/series.iloc[0])**(1/year)-1
        max_draw=-(series/series.expanding(1).max()-1).min()
        vol=np.std(series1)*(250**0.5)
        sharpe=ret_yearly/vol
        wins=series1[series1>0]
        win_rate=len(wins)/len(series1)
        return pd.Series([ret,ret_yearly,vol,sharpe,max_draw,win_rate],
                         index=['总收益率','年化收益率','年化波动率','信息比率','最大回撤率','胜率'])

    @kk.desktop_sender(title='嘿，变频回测结束啦～🗓')
    def run(self,print_comments,plt_plot,plotly_plot,filename):
        '''对因子值分组并匹配'''
        if self.boxcox:
            self.fac=decap_industry(self.fac)
        self.fac=self.fac.T.apply(lambda x:pd.qcut(x,self.group,labels=False,duplicates='drop')).T
        self.fac=self.fac.shift(1)
        self.vs=[(((self.fac==i)+0).replace(0,np.nan)*self.rets).mean(axis=1) for i in range(self.group)]
        self.group_rets=pd.DataFrame({f'group{k}':list(v) for k,v in zip(range(1,self.group+1),self.vs)},index=self.vs[0].index)
        self.group_rets=self.group_rets.dropna(how='all')
        self.group_rets=self.group_rets
        self.group_nets=(self.group_rets+1).cumprod()
        self.group_nets=self.group_nets.apply(lambda x:x/x.iloc[0])
        self.one=self.group_nets['group1']
        self.end=self.group_nets[f'group{self.group}']
        if self.one.iloc[-1]>self.end.iloc[-1]:
            self.long_name='group1'
            self.short_name=f'group{self.group}'
        else:
            self.long_name=f'group{self.group}'
            self.short_name='group1'
        self.long_short_ret=self.group_rets[self.long_name]-self.group_rets[self.short_name]
        self.long_short_net=(self.long_short_ret+1).cumprod()
        self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
        if self.long_short_net.iloc[-1]<1:
            self.long_short_ret=self.group_rets[self.short_name]-self.group_rets[self.long_name]
            self.long_short_net=(self.long_short_ret+1).cumprod()
            self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
            self.long_short_ret=self.group_rets[self.short_name]-self.group_rets[self.long_name]-2*self.trade_cost/self.freq
            self.long_short_net=(self.long_short_ret+1).cumprod()
            self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
        else:
            self.long_short_ret=self.group_rets[self.long_name]-self.group_rets[self.short_name]-2*self.trade_cost/self.freq
            self.long_short_net=(self.long_short_ret+1).cumprod()
            self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
        self.group_rets=pd.concat([self.group_rets,self.long_short_ret.to_frame('long_short')],axis=1)
        self.group_nets=pd.concat([self.group_nets,self.long_short_net.to_frame('long_short')],axis=1)
        self.long_short_comments=self.comments(self.long_short_net,self.long_short_ret)
        if print_comments:
            print(self.long_short_comments)
        if plt_plot:
            self.group_nets.plot()
            plt.savefig(filename+'.png')
            plt.show()
        if plotly_plot:
            fig=pe.line(self.group_nets)
            filename_path=filename+'.html'
            pio.write_html(fig,filename_path,auto_open=True)
#
#
# class pure_cloud(object):
#     '''
#     为了测试其他不同的频率而设计的类，仅考虑了上市满60天这一要素
#     这一回测采取的方案是，对于回测频率n天，将初始资金等分成n笔，每天以1/n的资金调仓
#     每笔资金之间相互独立，最终汇聚成一个收益率序列
#     '''
#     def __init__(self,fac,freq,group=10,boxcox=1,trade_cost=0,print_comments=1,plt_plot=1,plotly_plot=0,
#                  filename='净值走势图',comments_writer=None,nets_writer=None,sheet_name=None):
#         '''n是回测的频率，等分成n份，group是回测的组数，boxcox是是否做行业市值中性化'''
#         self.fac=fac
#         self.freq=freq
#         self.group=group
#         self.boxcox=boxcox
#         self.trade_cost=trade_cost
#         moon=pure_moon()
#         moon.prerpare()
#         ages=moon.ages.copy()
#         ages=(ages>=60)+0
#         self.ages=ages.replace(0,np.nan)
#         self.closes=read_daily(close=1)
#         self.rets=((self.closes.shift(-self.freq)/self.closes-1)*self.ages-trade_cost)/self.freq
#         self.run(print_comments=print_comments,plt_plot=plt_plot,plotly_plot=plotly_plot,filename=filename)
#         if comments_writer:
#             if sheet_name:
#                 self.long_short_comments.to_excel(comments_writer,sheet_name=sheet_name)
#             else:
#                 raise AttributeError('必须制定sheet_name参数🤒')
#         if nets_writer:
#             if sheet_name:
#                 self.group_nets.to_excel(nets_writer,sheet_name=sheet_name)
#             else:
#                 raise AttributeError('必须制定sheet_name参数🤒')
#
#     def comments(self,series,series1):
#         '''对twins中的结果给出评价
#         评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率'''
#         ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
#         duration=(series.index[-1]-series.index[0]).days
#         year=duration/365
#         ret_yearly=(series.iloc[-1]/series.iloc[0])**(1/year)-1
#         max_draw=-(series/series.expanding(1).max()-1).min()
#         vol=np.std(series1)*(250**0.5)
#         sharpe=ret_yearly/vol
#         wins=series1[series1>0]
#         win_rate=len(wins)/len(series1)
#         return pd.Series([ret,ret_yearly,vol,sharpe,max_draw,win_rate],
#                          index=['总收益率','年化收益率','年化波动率','信息比率','最大回撤率','胜率'])
#
#     def run(self,print_comments,plt_plot,plotly_plot,filename):
#         '''对因子值分组并匹配'''
#         if self.boxcox:
#             self.fac=decap_industry(self.fac)
#         self.fac=self.fac.T.apply(lambda x:pd.qcut(x,self.group,labels=False,duplicates='drop')).T
#         self.fac=self.fac.shift(1)
#         self.vs=[(((self.fac==i)+0).replace(0,np.nan)*self.rets).mean(axis=1) for i in range(self.group)]
#         self.group_rets=pd.DataFrame({f'group{k}':list(v) for k,v in zip(range(1,self.group+1),self.vs)},index=self.vs[0].index)
#         self.group_rets=self.group_rets.dropna(how='all')
#         self.group_nets=(self.group_rets+1).cumprod()
#         self.group_nets=self.group_nets.apply(lambda x:x/x.iloc[0])
#         self.one=self.group_nets['group1']
#         self.end=self.group_nets[f'group{self.group}']
#         if self.one.iloc[-1]>self.end.iloc[-1]:
#             self.long_name='group1'
#             self.short_name=f'group{self.group}'
#         else:
#             self.long_name=f'group{self.group}'
#             self.short_name='group1'
#         self.long_short_ret=self.group_rets[self.long_name]-self.group_rets[self.short_name]
#         self.long_short_net=(self.long_short_ret+1).cumprod()
#         self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
#         if self.long_short_net.iloc[-1]<1:
#             self.long_short_ret=self.group_rets[self.short_name]-self.group_rets[self.long_name]
#             self.long_short_net=(self.long_short_ret+1).cumprod()
#             self.long_short_net=self.long_short_net/self.long_short_net.iloc[0]
#         self.group_rets=pd.concat([self.group_rets,self.long_short_ret.to_frame('long_short')],axis=1)
#         self.group_nets=pd.concat([self.group_nets,self.long_short_net.to_frame('long_short')],axis=1)
#         self.long_short_comments=self.comments(self.long_short_net,self.long_short_ret)
#         if print_comments:
#             print(self.long_short_comments)
#         if plt_plot:
#             self.group_nets.plot()
#             plt.savefig(filename+'.png')
#             plt.show()
#         if plotly_plot:
#             fig=pe.line(self.group_nets)
#             filename_path=filename+'.html'
#             pio.write_html(fig,filename_path,auto_open=True)


class pure_wood(object):
    '''一种因子合成的方法，灵感来源于adaboost算法
    adaboost算法的精神是，找到几个分类效果较差的弱学习器，通过改变不同分类期训练时的样本的权重，
    计算每个学习器的错误概率，通过线性加权组合的方式，让弱分类期进行投票，决定最终分类结果，
    这里取通过计算各个因子多头或空头的错误概率，进而通过错误概率对因子进行加权组合，
    此处方式将分为两种，一种是主副因子的方式，另一种是全等价的方式，
    主副因子即先指定一个主因子（通常为效果更好的那个因子），然后指定若干个副因子，先计算主因子的错误概率，
    找到主因子多头里分类错误的部分，然后通过提高期加权，依次计算后续副因子的错误概率（通常按照因子效果从好到坏排序），
    最终对依次得到的错误概率做运算，然后加权
    全等价方式即不区分主副因子，分别独立计算每个因子的错误概率，然后进行加权'''
    def __init__(self,domain_fac: pd.DataFrame,subdomain_facs: list,group_num=10):
        '''声明主副因子和分组数'''
        self.domain_fac=domain_fac
        self.subdomain_facs=subdomain_facs
        self.group_num=group_num
        opens=read_daily(open=1).resample('M').first()
        closes=read_daily(close=1).resample('M').last()
        self.ret_next=closes/opens-1
        self.ret_next=self.ret_next.shift(-1)
        self.domain_fac=self.domain_fac.T.apply(lambda x:pd.qcut(x,group_num,labels=False,duplicates='drop')).T+1
        self.ret_next=self.ret_next.T.apply(lambda x:pd.qcut(x,group_num,labels=False,duplicates='drop')).T+1
        self.subdomain_facs=[i.T.apply(lambda x:pd.qcut(x,group_num,labels=False,duplicates='drop')).T+1 for i in self.subdomain_facs]
        self.get_all_a()
        self.get_three_new_facs()

    def __call__(self, *args, **kwargs):
        return copy.copy(self.new_facs)

    def get_a_and_new_weight(self,n,fac,weight=None):
        '''计算主因子的权重和权重矩阵'''
        fac_at_n=(fac==n)+0
        ret_at_n=(self.ret_next==n)+0
        not_nan=fac_at_n+ret_at_n
        not_nan=not_nan[not_nan.index.isin(fac_at_n.index)]
        not_nan=(not_nan>0)+0
        wrong=((ret_at_n-fac_at_n)>0)+0
        wrong=wrong[wrong.index.isin(fac_at_n.index)]
        right=((ret_at_n-fac_at_n)==0)+0
        right=right[right.index.isin(fac_at_n.index)]
        wrong=wrong*not_nan
        right=right*not_nan
        wrong=wrong.dropna(how='all')
        right=right.dropna(how='all')
        if isinstance(weight,pd.DataFrame):
            e_rate=(wrong*weight).sum(axis=1)
            a_rate=0.5*np.log((1-e_rate)/e_rate)
            wrong_here=-wrong
            g_df=multidfs_to_one(right,wrong_here)
            on_exp=(g_df.T*a_rate.to_numpy()).T
            with_exp=np.exp(on_exp)
            new_weight=weight*with_exp
            new_weight=(new_weight.T/new_weight.sum(axis=1).to_numpy()).T
        else:
            e_rate=(right.sum(axis=1))/(right.sum(axis=1)+wrong.sum(axis=1))
            a_rate=0.5*np.log((1-e_rate)/e_rate)
            wrong_here=-wrong
            g_df=multidfs_to_one(right,wrong_here)
            on_exp=(g_df.T*a_rate.to_numpy()).T
            with_exp=np.exp(on_exp)
            new_weight=with_exp.copy()
            new_weight=(new_weight.T/new_weight.sum(axis=1).to_numpy()).T
        return a_rate,new_weight

    def get_all_a(self):
        '''计算每个因子的a值'''
        #第一组部分
        one_a_domain,one_weight=self.get_a_and_new_weight(1,self.domain_fac)
        one_a_list=[one_a_domain]
        for fac in self.subdomain_facs:
            one_new_a,one_weight=self.get_a_and_new_weight(1,fac,one_weight)
            one_a_list.append(one_new_a)
        self.a_list_one=one_a_list
        #最后一组部分
        end_a_domain,end_weight=self.get_a_and_new_weight(self.group_num,self.domain_fac)
        end_a_list=[end_a_domain]
        for fac in self.subdomain_facs:
            end_new_a,end_weight=self.get_a_and_new_weight(self.group_num,fac,end_weight)
            end_a_list.append(end_new_a)
        self.a_list_end=end_a_list

    def get_three_new_facs(self):
        '''分别使用第一组加强、最后一组加强、两组平均的方式结合'''
        one_fac=sum([(i.iloc[1:,:].T*j.iloc[:-1].to_numpy()).T for i,j in zip([self.domain_fac]+self.subdomain_facs,self.a_list_one)])
        end_fac=sum([(i.iloc[1:,:].T*j.iloc[:-1].to_numpy()).T for i,j in zip([self.domain_fac]+self.subdomain_facs,self.a_list_end)])
        both_fac=one_fac+end_fac
        self.new_facs=[one_fac,end_fac,both_fac]


class pure_fire(object):
    '''一种因子合成的方法，灵感来源于adaboost算法
    adaboost算法的精神是，找到几个分类效果较差的弱学习器，通过改变不同分类期训练时的样本的权重，
    计算每个学习器的错误概率，通过线性加权组合的方式，让弱分类期进行投票，决定最终分类结果，
    这里取通过计算各个因子多头或空头的错误概率，进而通过错误概率对因子进行加权组合，
    此处方式将分为两种，一种是主副因子的方式，另一种是全等价的方式，
    主副因子即先指定一个主因子（通常为效果更好的那个因子），然后指定若干个副因子，先计算主因子的错误概率，
    找到主因子多头里分类错误的部分，然后通过提高期加权，依次计算后续副因子的错误概率（通常按照因子效果从好到坏排序），
    最终对依次得到的错误概率做运算，然后加权
    全等价方式即不区分主副因子，分别独立计算每个因子的错误概率，然后进行加权'''
    def __init__(self,facs: list,group_num=10):
        '''声明主副因子和分组数'''
        self.facs=facs
        self.group_num=group_num
        opens=read_daily(open=1).resample('M').first()
        closes=read_daily(close=1).resample('M').last()
        self.ret_next=closes/opens-1
        self.ret_next=self.ret_next.shift(-1)
        self.ret_next=self.ret_next.T.apply(lambda x:pd.qcut(x,group_num,labels=False,duplicates='drop')).T+1
        self.facs=[i.T.apply(lambda x:pd.qcut(x,group_num,labels=False,duplicates='drop')).T+1 for i in self.facs]
        self.get_all_a()
        self.get_three_new_facs()

    def __call__(self, *args, **kwargs):
        return copy.copy(self.new_facs)

    def get_a(self,n,fac):
        '''计算主因子的权重和权重矩阵'''
        fac_at_n=(fac==n)+0
        ret_at_n=(self.ret_next==n)+0
        not_nan=fac_at_n+ret_at_n
        not_nan=not_nan[not_nan.index.isin(fac_at_n.index)]
        not_nan=(not_nan>0)+0
        wrong=((ret_at_n-fac_at_n)>0)+0
        wrong=wrong[wrong.index.isin(fac_at_n.index)]
        right=((ret_at_n-fac_at_n)==0)+0
        right=right[right.index.isin(fac_at_n.index)]
        wrong=wrong*not_nan
        right=right*not_nan
        wrong=wrong.dropna(how='all')
        right=right.dropna(how='all')
        e_rate=(right.sum(axis=1))/(right.sum(axis=1)+wrong.sum(axis=1))
        a_rate=0.5*np.log((1-e_rate)/e_rate)
        return a_rate

    def get_all_a(self):
        '''计算每个因子的a值'''
        #第一组部分
        self.a_list_one=[self.get_a(1,i) for i in self.facs]
        #最后一组部分
        self.a_list_end=[self.get_a(self.group_num,i) for i in self.facs]

    def get_three_new_facs(self):
        '''分别使用第一组加强、最后一组加强、两组平均的方式结合'''
        one_fac=sum([(i.iloc[1:,:].T*j.iloc[:-1].to_numpy()).T for i,j in zip(self.facs,self.a_list_one)])
        end_fac=sum([(i.iloc[1:,:].T*j.iloc[:-1].to_numpy()).T for i,j in zip(self.facs,self.a_list_end)])
        both_fac=one_fac+end_fac
        self.new_facs=[one_fac,end_fac,both_fac]



