# pure_ocean_breeze 
#### **芷琦哥的回测框架**
##### 我们的口号是：量价因子才是最牛的！
***

>### 全新大版本📢
>* v3.0.0 — 2022.08.16
>>回测框架3.0版本来啦！ 模块拆分&说明文档来啦！[pure_ocean_breeze说明文档](https://chen-001.github.io/pure_ocean_breeze/)
>* v2.0.0 — 2022.07.12
>>回测框架2.0版本来啦！数据库&自动更新&最终因子库功能上线啦！

>### 安装&使用指南🎯
>>1. 安装
>>使用`pip install pure_ocean_breeze`命令进行安装
>>2. 初始化
>>>* 在初次安装框架时，请进行初始化，以将路径设置到自己的文件里
>>>* 使用如下语句进行初始化
>>>```python
>>>import pure_ocean_breeze.initialize
>>>pure_ocean_breeze.initialize.initialize()
>>>```
>>>* 然后根据提示进行操作即可
>>>* 请注意路径不要写反斜杠\，而要写成/
>>>* 经过初始化后，以后就可以直接使用，不论重启电脑或者版本升级，都不用再初始化
>>>* 如果更换了数据库路径，请重新初始化
>>3. 日常调用
>>>* **导入框架** 
>>>>```python
>>>>import pure_ocean_breeze.pure_ocean_breeze as pp
>>>>```
>>>* **一键回测** 
>>>>```python
>>>>shen=pp.pure_moonnight(fac,boxcox=1)
>>>>```
>>>>`fac`为因子矩阵，`boxcox`表示是否做行业市值中性化
>>>* **一键读入日频数据**
>>>>```python
>>>>pp.read_daily(path=None,open=0,close=0,high=0,low=0,tr=0,sharenum=0,volume=0,unadjust=0,>>>>start=STATES['START'])
>>>>```
>>>>将其中任何一个为0的参数改为1，可以读取对应的复权价，或者换手率，`unadjust`可以修改为不复权，`start`可以指定起始日期。
>>>* **因子合成运算** 使用`pp.pure_fallmount()`类，具体方法正在补充中……
>>>* **其余内容敬请期待**

>#### 作者😉
>>* 量价选股因子黄金矿工💁‍♂️
>>* 挖因子兼新技术爱好者💁‍♂️
>>* 芷琦哥的小迷弟&感谢芷琦哥教我做因子💐
>>* 欢迎交流技术优化&因子灵感&工作信息：<winterwinter999@163.com>

>#### 相关链接🔗
>* [PyPi](https://pypi.org/project/pure-ocean-breeze/)
>* [pure_ocean_breeze说明文档](https://chen-001.github.io/pure_ocean_breeze/)
>* [Github同步到Pypi操作手册](https://github.com/chen-001/pure_ocean_breeze/blob/master/Github同步Pypi操作手册/Github同步Pypi操作手册.md)
>* [更新日志version3](更新日志/version3.md)
>* [更新日志version2](更新日志/version2.md)



