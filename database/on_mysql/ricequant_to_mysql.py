from pure_ocean_breeze.pure_ocean_breeze import *

'''执行本代码之前，请先删除之前的两个数据minute_data和minute_data_alter'''

#分钟数据文件所在的路径
mipath='/Users/chenzongwei/pythoncode/数据库/Ricequant-minbar/equities'

#新建数据库
sql=sqlConfig()
#每只股票一张表
sql.add_new_database('minute_data_stock')
#每天所有股票一张表
sql.add_new_database('minute_data_stock_alter')
#每个指数一张表
sql.add_new_database('minute_data_index')
#每天所有指数一张表
sql.add_new_database('minute_data_index_alter')

#连接4个数据库
sqls=sqlConfig('minute_data_stock')
sqlsa=sqlConfig('minute_data_stock_alter')
sqli=sqlConfig('minute_data_index')
sqlia=sqlConfig('minute_data_index_alter')

#遍历所有分钟数据文件
files=os.listdir(mipath)
files=[mipath+'/'+i for i in files]
fails=[]

for path in tqdm.tqdm(files):
    #识别代码属于股票还是指数
    code,kind=convert_code(path)
    #一些数据预处理
    df=read_h5_new(path)
    df=df.rename(columns={'datetime':'date','volume':'amount','total_turnover':'money'})
    df=df[['date','open','high','low','close','amount','money']].sort_values('date')
    df.date=df.date.astype(str).str.slice(stop=8).astype(int)
    df=df.groupby('date').apply(lambda x:x.assign(num=list(range(1,x.shape[0]+1))))
    df=(np.around(df,2)*100).ffill().dropna().astype(int).assign(code=code)
    #所有的日期
    dates=list(set(df.date))
    try:
        #股票
        if kind=='stock':
            #写入每只股票一张表
            df.drop(columns=['code']).to_sql(
                name=code,con=sqls.engine,if_exists='replace',index=False,
                dtype={'date':INT,'open':INT,'high':INT,'low':INT,'close':INT,'amount':BIGINT,'money':BIGINT,'num':INT})
            #把每天写入每天所有股票一张表
            for date in dates:
                dfi=df[df.date==date]
                dfi.drop(columns=['date']).to_sql(
                    name=str(date),con=sqlsa.engine,if_exists='append',index=False,
                    dtype={'code':VARCHAR(9),'open':INT,'high':INT,'low':INT,'close':INT,'amount':BIGINT,'money':BIGINT,'num':INT})
        #指数
        else:
            #写入每个指数一张表
            df.drop(columns=['code']).to_sql(
                name=code,con=sqli.engine,if_exists='replace',index=False,
                dtype={'date':INT,'open':INT,'high':INT,'low':INT,'close':INT,'amount':BIGINT,'money':BIGINT,'num':INT})
            #把每天写入每天所有指数一张表
            for date in dates:
                dfi=df[df.date==date]
                dfi.drop(columns=['date']).to_sql(
                    name=str(date),con=sqlia.engine,if_exists='append',index=False,
                    dtype={'code':VARCHAR(9),'open':INT,'high':INT,'low':INT,'close':INT,'amount':BIGINT,'money':BIGINT,'num':INT})
    except Exception:
        fails.append(code)
        logger.warning(f'{code}失败了，请检查')
logger.success('全部搬运到mysql里啦')