from pure_ocean_breeze.pure_ocean_breeze import *

mipath='/Users/chenzongwei/pythoncode/数据库/Ricequant-minbar/equities'
sql=sqlConfig('minute_data')
chc=ClickHouseClient('minute_data')
files=sorted(os.listdir(mipath))
files=[mipath+'/'+i for i in files]
fails=[]

for path in tqdm.tqdm(files):
    code,kind=convert_code(path)
    df=read_h5_new(path)
    df=df.rename(columns={'datetime':'date','volume':'amount','total_turnover':'money'})
    df=df[['date','open','high','low','close','amount','money']].sort_values('date')
    df.date=df.date.astype(str).str.slice(stop=8).astype(int)
    df=df.groupby('date').apply(lambda x:x.assign(num=list(range(1,x.shape[0]+1))))
    df=(np.around(df,2)*100).ffill().dropna().astype(int).assign(code=code)
    try:
        if kind=='stock':
            df.to_sql('minute_data_stock',chc.engine,if_exists='append',index=False)
        else:
            df.to_sql('minute_data_index',chc.engine,if_exists='append',index=False)
    except Exception:
        fails.append(code)
        logger.warning(f'{code}失败了，请检查')
logger.success('存储完啦')

# again=[i for i in files if fails[0][:6] in i][0]
# again=[i for i in files if '300671' in i][0]
#
# code,kind=convert_code(again)
# df=read_h5_new(again)
# df=df.rename(columns={'datetime':'date','volume':'amount','total_turnover':'money'})
# df=df[['date','open','high','low','close','amount','money']].sort_values('date')
# df.date=df.date.astype(str).str.slice(stop=8).astype(int)
# df=df.groupby('date').apply(lambda x:x.assign(num=list(range(1,x.shape[0]+1))))
# df=(np.around(df,2)*100).ffill().dropna().astype(int).assign(code=code)
# df.to_sql('minute_data_stock',chc.engine,if_exists='append',index=False)
#
# chc.get_data("select * from minute_data.minute_data_stock where code='300671.SZ'")