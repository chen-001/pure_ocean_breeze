from setuptools import setup
import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setup(
    name='pure_ocean_breeze',
    version='2.6.7',
    description='芷琦哥的回测框架',
    long_description='详见homepage\nhttps://github.com/chen-001/pure_ocean_breeze.git',
    author='chenzongwei',
    author_email='17695480342@163.com',
    py_modules=['pure_ocean_breeze','pure_ocean_breeze.pure_ocean_breeze','pure_ocean_breeze.initialize'],
    url='https://github.com/chen-001/pure_ocean_breeze.git',
    # project_urls={'Documentation':'https://www.craft.do/s/xazRpMa29CO895','Say Thanks!':'https://www.craft.do/s/jqbL7e1mBuzbtB'},
    install_requires=['numpy','pandas','scipy','statsmodels','plotly','matplotlib','feather','loguru','h5py','cachier','knockknock','dcube','tenacity','alphalens','pickledb','pymysql','sqlalchemy','pretty_errors'],
    python_requires='>=3',
    license='MIT',
    packages=setuptools.find_packages(),
    requires=[]
)

