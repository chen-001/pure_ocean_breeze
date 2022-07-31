from pure_ocean_breeze.pure_ocean_breeze import *

'''创建分钟数据的数据库'''
#db_user为用户名，db_password为密码
sql=sqlConfig(db_user='root',db_host='127.0.0.1',db_port=3306,db_password='Kingwila98')
sql.add_new_database('minute_data')
sql.add_new_database('minute_data_alter')

'''将分钟数据写入sql，每个股票一张表'''
s=sqlConfig(db_name='minute_data')
fs=sorted(os.listdir(homeplace.minute_data_file))
fs=[i for i in fs if i.endswith('.mat')]
for f in tqdm.tqdm(fs):
    k,v=read_minute_mat(f)
    v=v.where(v<1e38,np.nan).where(v>-1e38,np.nan)
    v.to_sql(name=k,con=s.engine,if_exists='replace',index=False,
             dtype={'date':INT,'open':FLOAT(2),'high':FLOAT(2),'low':FLOAT(2),'close':FLOAT(2),'amount':INT,'money':FLOAT(2),'num':INT})
logger.success('minute_data数据库，即每个股票一张表已经写入完成')


'''将分钟数据写入sql，每天一张表'''
front=list(range(2013,2023))
behind=list(range(2013,2024))

def single_year(f,b):
    sa=sqlConfig('minute_data_alter')
    f,b=f*10000,b*10000
    #读入全部分钟数据
    dfs=[]
    for file in tqdm.tqdm(fs,desc='文件读取中'):
        code,df=read_minute_mat(file)
        df=df[(df.date>f)&(df.date<b)]
        df=df.where(df<1e38,np.nan).where(df>-1e38,np.nan)
        df=df.assign(code=code)
        dfs.append(df)
    dfs=pd.concat(dfs)
    #获取期间的交易日历
    f1,f2=str(f+101),str(b+1231-10000)
    dates=list(map(int,sorted(list(set(pro.a_calendar(start_date=f1,end_date=f2).trade_date)))))
    #拆分并逐个写入
    for day in tqdm.tqdm(dates,desc='逐日写入中'):
        df=dfs[dfs.date==day]
        df=df.drop(columns='date')
        if df.shape[0]>0:
            df.to_sql(name=str(day),con=sa.engine,if_exists='replace',index=False,
                      dtype={'open':FLOAT(2),'high':FLOAT(2),'low':FLOAT(2),'close':FLOAT(2),'amount':INT,'money':FLOAT(2),'num':INT,'code':VARCHAR(9)})

for f,b in zip(front,behind):
    single_year(f,b)
logger.success('minute_data_alter数据库，即每天一张表已经写入完成')

