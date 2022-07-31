# pure_ocean_breeze 
#### **芷琦哥的回测框架**
##### 我们的口号是：量价因子才是最牛的！
***

>### 全新大版本📢
>* v2.0.0 — 2022.07.12
>>回测框架2.0版本来啦！数据库自动更新&最终因子库功能上线啦！

>### 更新日志🗓
>* v2.6.3 — 2022.07.31 
>>1. 修复了分钟数据计算因子类pure_fall_frequent和pure_fall_flexible的更新bug
>>2. 修复了更新3510指数行情数据的bug，以及读取bug
>>3. 将STATES中的默认参数startdate从20100101修改为20130101

>### 安装&使用指南🎯
>>1. 安装
>>使用`pip install pure_ocean_breeze`命令进行安装
>>2. 初始化
>>>* 在初次安装框架时，请进行初始化，以将路径设置到自己的文件里
>>>* 使用如下语句进行初始化
>>>```
>>>import pure_ocean_breeze.initialize
>>>pure_ocean_breeze.initialize.initialize()
>>>```
>>>* 然后根据提示进行操作即可
>>>* 请注意路径不要写反斜杠\，而要写成/
>>>* 经过初始化后，以后就可以直接使用，不论重启电脑或者版本升级，都不用再初始化
>>>* 如果更换了数据库路径，请重新初始化
>>3. 日常调用
>>>* **导入框架** 
>>>>```
>>>>import pure_ocean_breeze.pure_ocean_breeze as pp
>>>>```
>>>* **一键回测** 
>>>>```
>>>>shen=pp.pure_moonnight(fac,boxcox=1)
>>>>```
>>>>`fac`为因子矩阵，`boxcox`表示是否做行业市值中性化
>>>* **一键读入日频数据**
>>>>```
>>>>pp.read_daily(path=None,open=0,close=0,high=0,low=0,tr=0,sharenum=0,volume=0,unadjust=0,>>>>start=STATES['START'])
>>>>```
>>>>将其中任何一个为0的参数改为1，可以读取对应的复权价，或者换手率，`unadjust`可以修改为不复权，`start`可以指定起始日期。
>>>* **因子合成运算** 使用pp.pure_fallmount()类，具体方法正在补充中……
>>>* **其余内容敬请期待**

>#### 作者😉
>>* 量价选股因子卖方矿工💁‍♂️
>>* 挖因子兼新技术爱好者💁‍♂️
>>* 芷琦哥的小迷弟&感谢芷琦哥教我做因子💐
>>* 欢迎交流技术优化&因子灵感&工作信息：winterwinter999@163.com

>#### 相关链接🔗
>* [pypi链接](https://pypi.org/project/pure-ocean-breeze/)
>* [因子合集1](https://mp.weixin.qq.com/s/5oM5qyfQL_zxEGT4HYxqEQ)
>* [因子合集2](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2MDMyMjU0Ng==&action=getalbum&album_id=2441294745860882432&scene=173&from_msgid=2247497619&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
