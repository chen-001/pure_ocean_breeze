# pure_ocean_breeze
芷琦哥的回测框架

在初次安装框架时，请进行初始化，以将路径设置到自己的文件里。

请将分钟数据单独放在一个文件夹里。

使用如下语句进行初始化

import pure_ocean_breeze.initialize

pure_ocean_breeze.initialize.initialize()

然后根据提示进行操作即可

例如windows系统会输入

D:/日频数据/

D:/分钟数据/

D:/因子数据/

D:/barra/

mac系统会输入

/Users/meme/日频数据/

/Users/meme/分钟数据/

/Users/meme/因子数据/

/Users/meme/barra数据/

以上路径仅为举例，请依照自己电脑实际情况而定。注意不要写反斜杠\，而要写成/，并且注意请以/结尾

经过初始化后，以后就可以直接使用，不论重启电脑或者版本升级，都不用再初始化。

如果更换了数据库路径，请重新初始化。

日常调用时，请使用import pure_ocean_breeze.pure_ocean_breeze as pp

一键回测 a=pp.pure_moonnight(fac,10,boxcox=True)
fac为因子矩阵，10为分成10组，boxcox表示是否做市值中性化。

首次回测时，会话费大概30-60分钟左右的时间，来生成交易状态的数据，并存储在本地，以便以后回测时直接调用。

pp.read_daily(path=None,close=0,open=0,high=0,low=0,tr=0)将其中任何一个为0的参数改为1，可以读取对应的复权价，或者换手率。

数据库文件更新后，请调用pp.read_daily.clear_cache()来清除之前的缓存。

因子合成运算，使用pp.pure_fallmount()
具体方法可以联系作者，懒得写说明了

