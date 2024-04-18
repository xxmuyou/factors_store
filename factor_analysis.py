import pandas as pd
import numpy as np
import alphalens
import matplotlib.pyplot as plt
from typing import Tuple

'''
                                            1D15m	    2D30m	    5D1h15m	    10D2h30m	factor	factor_quantile
date	            asset						
2022-01-01 00:15:00	BLZUSDT.BINANCE-UFUT	0.002930	0.010902	0.016784	0.011947	0.001763	1
                    ARPAUSDT.BINANCE-UFUT	0.009460	0.007364	0.008693	0.011557	0.005861	3
                    C98USDT.BINANCE-UFUT	0.005001	0.009377	0.018170	0.021421	0.007461	4
                    CELOUSDT.BINANCE-UFUT	-0.004648	0.004749	0.028291	0.029706	0.017176	5
                    CELRUSDT.BINANCE-UFUT	0.002986	0.011943	0.030128	0.017100	0.007666	4
...	...	...	...	...	...	...	...
2022-02-28 00:00:00	OMGUSDT.BINANCE-UFUT	0.000000	0.000000	0.000000	0.000000	0.002633	5
                    DUSKUSDT.BINANCE-UFUT	0.000000	0.000000	0.000000	0.000000	-0.000012	2
                    SKLUSDT.BINANCE-UFUT	0.000000	0.000000	0.000000	0.000000	-0.001198	1
                    NEARUSDT.BINANCE-UFUT	0.000000	0.000000	0.000000	0.000000	0.003551	5
                    SANDUSDT.BINANCE-UFUT	0.000000	0.000000	0.000000	0.000000	0.000263	3
'''
# 输入的dataframe格式

class Factor_Analysis:
    def __init__(self,
                 factor:pd.DataFrame,
                 frequnce_name:str = '1D15m',
                 suptitle:str='factor_sample'
                 ):
        self.factor =  factor   
        self.suptitle = suptitle
        self.frequnce_name = frequnce_name
    def performance(self):
        self.ic = calculate_ic(self.factor)
        ic_means = self.ic.mean()
        if ic_means[0]>0: # 确定因子方向
            factor_side = True
        else:
            factor_side = False
        long_short = long_short_returns(self.factor, factor_side)
        self.group_return = group_returns(self.factor,self.frequnce_name, factor_side)
        self.long_short_net_value = long_short[0]
        self.long_short_return = long_short[1]
        self.indicator = indicator(self.ic, self.long_short_net_value, self.long_short_return)
        return self.indicator
    def show(self):
        fig = plt.figure(figsize=(10, 12))  # 调整图片大小
        plt.suptitle(self.suptitle)  # 大标题
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        # 对ic进行作图
        ax1 = plt.subplot2grid((5, 2), (0, 0), rowspan=1, colspan=2)
        plot_subplot(ax1, self.ic.cumsum(), 'cal IC', self.ic.columns)

        # 多空净值图
        ax2 = plt.subplot2grid((5, 2), (1, 0), rowspan=1, colspan=2)
        plot_subplot(ax2, self.long_short_net_value, 'long short Net_values', self.long_short_net_value.columns)

        # 分组收益率
        ax3 = plt.subplot2grid((5, 2), (2, 0), rowspan=1, colspan=2)
        plot_subplot(ax3, self.group_return.cumsum(), f'group Returns', self.group_return.columns)
        
        ax4 = plt.subplot2grid((5, 2), (3, 0), rowspan=1, colspan=1)
        year_return = (1 + self.group_return).prod() - 1
        ax4.bar(year_return.index,year_return.values,color='royalblue',width=0.6)
        ax4.tick_params(axis='y', labelsize=6)
        ax4.tick_params(axis='x', labelrotation=45,labelsize=6)
        ax4.set_title('group_return compare',fontsize=8)
        



# 计算IC的dataframe函数
def calculate_ic(df) -> pd.DataFrame:
    IC = alphalens.performance.factor_information_coefficient(df)
    return IC

# 计算组合分组的收益率函数的dataframe
def group_returns(df:pd.DataFrame, frequence:str, factor_side:bool) -> pd.DataFrame:
    returns = df.loc[:,frequence].unstack()
    groups = df.iloc[:,-1].unstack()
    group_dataframe = pd.DataFrame()
    for i,group_num in enumerate(sorted(set(df['factor_quantile'].values),reverse=factor_side)):
        group_dataframe[f'group{i+1}'] = returns[groups==group_num].mean(axis=1)
    return group_dataframe

# 计算组合的多空净值
def long_short_returns(df:pd.DataFrame, factor_side:bool) -> Tuple[pd.DataFrame,pd.DataFrame]:
    frequnce_list = df.columns.tolist()
    frequnce_list.remove('factor')
    frequnce_list.remove('factor_quantile')
    return_dict = {} # 存多空收益率
    net_value_dict = {} # 存多空净值
    for fre in frequnce_list:
        freq = int(fre.split('D')[0]) # 频率更正
        frequnce_df = group_returns(df, fre, factor_side) / freq
        top = frequnce_df.iloc[:,0]
        bottom = frequnce_df.iloc[:,-1]
        diff_return = top - bottom
        return_dict[fre] = diff_return
        net_value_dict[fre] = (1 + (diff_return)).cumprod()
    return pd.DataFrame(net_value_dict) ,pd.DataFrame(return_dict)

# 用于指标计算
def indicator(IC:pd.DataFrame, long_short_net_value:pd.DataFrame, long_short_return:pd.DataFrame):
    result_dict = {} # 结果存储
    IC_means = IC.mean()
    ICIR = IC_means / IC.std()
    result_dict['IC_MEAN'] = IC_means
    result_dict['ICIR'] = ICIR
    
    daliy_num = pd.Timedelta(days=1)/long_short_net_value.index.to_series().diff().unique()[1] # 周期
    t = len(long_short_net_value)
    daliy_return = long_short_net_value.iloc[-1,:]**(daliy_num/t) - 1 # 日度收益率
    daliy_sharpe = (long_short_return.mean() / long_short_return.std()) * np.sqrt(daliy_num) # 日度夏普
    result_dict['long_short_daliy_return'] = daliy_return
    result_dict['long_short_daliy_sharpe'] = daliy_sharpe
    min_ = long_short_net_value[::-1].cummin()[::-1] # 每个时间后的最小值
    result_dict['long_short_max_drawn'] = ((long_short_net_value - min_) / long_short_net_value).max()
    return pd.DataFrame(result_dict)

# 画图
def plot_subplot(ax, data, title, legend_labels):
    ax.plot(data, linewidth=0.8)
    ax.set_title(title, fontsize=8)
    ax.legend(legend_labels, loc='lower left', bbox_to_anchor=(0.0, 0.0), prop={'size': 6})
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelrotation=0, labelsize=8)
    ax.grid(True)
    


if __name__ == '__main__()':
    result = Factor_Analysis(factor,suptitle='test')
    results_dict = result.performance()
    result.show()