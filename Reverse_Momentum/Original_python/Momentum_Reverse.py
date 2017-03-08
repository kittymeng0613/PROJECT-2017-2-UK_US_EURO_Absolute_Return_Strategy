# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:03:28 2017

@author: Kitty
"""
import pandas as pd
import numpy as np
from datetime import *
import datetime
from pylab import *
from WindPy import *
import statsmodels.api as sm
from sklearn import linear_model

import os 
os.chdir('C:\Users\Kitty\Desktop\Guo\python')
save_direction='C:\Users\Kitty\Desktop\Guo\python'
from Momentum_Reverse_data import *

w.start()

firstday='2006-12-31'
lastday='2016-12-31'

# 获得标的：沪港通，深港通wind ticker
str_hk_ticker=hk_list
list_hk_ticker=str_hk_ticker.split(',')
pd_hk_ticker=pd.DataFrame(str_hk_ticker.split(','),columns=['wind_code'])

'''
# 获得所有样本股的日度closeprice
daily_close=data_fetch(list_hk_ticker,pd_hk_ticker,'close',0.1,firstday,lastday,'D')
daily_close.to_csv(u'%s\\hk_daily_close.csv'%(save_direction),encoding= 'gb2312')

# 获得最高价，最低价和成交量
daily_high=data_fetch(list_hk_ticker,pd_hk_ticker,'high',0.1,firstday,lastday,'D')
daily_high.to_csv(u'%s\\hk_daily_high.csv'%(save_direction),encoding= 'gb2312')

daily_low=data_fetch(list_hk_ticker,pd_hk_ticker,'low',0.1,firstday,lastday,'D')
daily_low.to_csv(u'%s\\hk_daily_low.csv'%(save_direction),encoding= 'gb2312')

daily_volume=data_fetch(list_hk_ticker,pd_hk_ticker,'volume',0.1,firstday,lastday,'D')
daily_volume.to_csv(u'%s\\hk_daily_volume.csv'%(save_direction),encoding= 'gb2312')

daily_turn=data_fetch(list_hk_ticker,pd_hk_ticker,'turn',0.1,firstday,lastday,'D')
daily_turn.to_csv(u'%s\\hk_daily_turn.csv'%(save_direction),encoding= 'gb2312')

daily_open=data_fetch(list_hk_ticker,pd_hk_ticker,'open',0.1,firstday,lastday,'D')
daily_open.to_csv(u'%s\\hk_daily_open.csv'%(save_direction),encoding= 'gb2312')

'''
daily_close=pd.read_csv(u'%s\\hk_daily_close.csv'%(save_direction),encoding= 'gb2312',index_col=0)

# 因为接下来的收益排名都是基于前10个交易日的累计收益率，因此计算：
'''
window1=10
cum_return=pd.DataFrame(index=daily_close.index[window1:len(daily_close)],columns=list_hk_ticker)
for i in range(window1,len(daily_close)):
    # i=10
    close_1=daily_close.loc[daily_close.index[i-window1]]
    close_2=daily_close.loc[daily_close.index[i]]
    cum_return.loc[daily_close.index[i]]=(close_2-close_1)/close_1
    print (daily_close.index[i])
cum_return.to_csv(u'%s\\hk_cum_return.csv'%(save_direction),encoding= 'gb2312')
'''
cum_return=pd.read_csv(u'%s\\hk_cum_return.csv'%(save_direction),encoding= 'gb2312',index_col=0)   

# 按照申万一级行业分类得到所属行业code
'''
data_fetched=w.wsd(str_hk_ticker,"industry_gicscode","2017-02-14","2017-02-14","industryType=1")
ind_data =pd.DataFrame(data_fetched.Data[0],index=list_hk_ticker,columns=['industry_code'])
ind_data.to_csv(u'%s\\hk_industry_data.csv'%(save_direction),encoding= 'gb2312')
'''
ind_data=pd.read_csv(u'%s\\hk_industry_data.csv'%(save_direction),encoding= 'gb2312',index_col=0)   
industry_code_list=ind_data['industry_code'].drop_duplicates()

# 计算同属于一个行业的股票前10个交易日的累计收益率进行排名并标准化，作为个股的行业间动量
ind_momentum=pd.DataFrame(index=cum_return.index,columns=industry_code_list)
for i in industry_code_list:
    # i=industry_code_list[0]
    sub_ind_ticker=ind_data[ind_data['industry_code']==i].index
    sub_ind_cum_return=cum_return[sub_ind_ticker]
    ind_momentum[i]=sub_ind_cum_return.T.mean().values
    print i
    
score_ind_momentum=Get_score(ind_momentum,False,'ind')
   
# 个股所属于的行业的score作为该股票的行业间动量
stock_score_ind_momentum=pd.DataFrame(index=cum_return.index,columns=cum_return.columns)
for i in stock_score_ind_momentum.index:
    for j in stock_score_ind_momentum.columns:
        ind_code=ind_data.at[j,'industry_code']
        if daily_close.loc[i].fillna(0)[j]!=0:
            stock_score_ind_momentum.at[i,j]=score_ind_momentum.at[i,ind_code]
    print i
stock_score_ind_momentum.to_csv(u'%s\\score\\stock_score_ind_momentum.csv'%(save_direction),encoding= 'gb2312')

# 用个股原始收益-行业平均收益，得到经行业调整后的收益，进行排名并标准化

relative_ret=pd.DataFrame(index=cum_return.index,columns=cum_return.columns)
for i in cum_return.columns:
    ind_code=ind_data.at[i,'industry_code']
    relative_ret[i]=cum_return[i]-ind_momentum[ind_code]
    print i

stock_score_ind_reverse=Get_score(relative_ret,False,'stock')  
stock_score_ind_reverse.to_csv(u'%s\\score\\stock_score_ind_reverse.csv'%(save_direction),encoding= 'gb2312')

# 计算个股前Amihud指标，进行排名并标准化，作为流动性反转指标
daily_high=pd.read_csv(u'%s\\hk_daily_high.csv'%(save_direction),encoding= 'gb2312',index_col=0)
daily_low=pd.read_csv(u'%s\\hk_daily_low.csv'%(save_direction),encoding= 'gb2312',index_col=0)
daily_turn=pd.read_csv(u'%s\\hk_daily_turn.csv'%(save_direction),encoding= 'gb2312',index_col=0)
daily_open=pd.read_csv(u'%s\\hk_daily_open.csv'%(save_direction),encoding= 'gb2312',index_col=0)

window2=20
amihud=pd.DataFrame(index=daily_open.index[window2:len(daily_open)],columns=list_hk_ticker)
for i in range(window2,len(daily_open)):
    # i=20
    temp_period=daily_open.index[i-window2:i]
    temp_amihud=(daily_high.loc[temp_period]-daily_low.loc[temp_period])/daily_open.loc[temp_period]/daily_turn.loc[temp_period]

    amihud.loc[daily_open.index[i]]=temp_amihud.mean()
    print (daily_open.index[i])
    
amihud.to_csv(u'%s\\hk_20day_amihud.csv'%(save_direction),encoding= 'gb2312')

amihud=pd.read_csv(u'%s\\hk_20day_amihud.csv'%(save_direction),encoding= 'gb2312',index_col=0)
stock_score_illiquidity_reverse=Get_score(amihud,False,'stock')  
stock_score_illiquidity_reverse.to_csv(u'%s\\score\\stock_score_illiquidity_reverse.csv'%(save_direction),encoding= 'gb2312')

# 个股对wind A股进行回归，残差的标准差作为非系统性风险

# 获得wind港股通指数/MSCI香港市场指数 十天累计收益率
'''
ggt_index_data_fetched=w.wsd("881005.WI","close",firstday,lastday)
ggt_index =pd.DataFrame(ggt_index_data_fetched.Data[0],index=ggt_index_data_fetched.Times,columns=['ggt'])

hk_index_data_fetched=w.wsd("934400.MI","close",firstday,lastday)
hk_index =pd.DataFrame(hk_index_data_fetched.Data[0],index=hk_index_data_fetched.Times,columns=['hk_index'])
hk_index.to_csv(u'%s\\hk_market_index.csv'%(save_direction),encoding= 'gb2312')
'''
hk_index=pd.read_csv(u'%s\\hk_market_index.csv'%(save_direction),encoding= 'gb2312',index_col=0)

window1=10
index_cum_return=pd.DataFrame(index=hk_index.index[window1:len(daily_close)],columns=['index_cum_ret'])
for i in range(window1,len(hk_index)):
    # i=10
    close_1=hk_index.at[hk_index.index[i-window1],'hk_index']
    close_2=hk_index.at[hk_index.index[i],'hk_index']
    index_cum_return.at[hk_index.index[i],'index_cum_ret']=(close_2-close_1)/close_1
    print (hk_index.index[i])
index_cum_return.to_csv(u'%s\\hk_cum_return.csv'%(save_direction),encoding= 'gb2312')


# 用每只个股的十天累计收益和index_cum_return做回归，滚动20个交易日计算残差的标准差
    
window2=20
unsystematic_risk=pd.DataFrame(index=cum_return.index[window2:len(cum_return)],columns=list_hk_ticker)
for i in range(window2,len(cum_return)):
    # i=20
    temp_period=cum_return.index[i-window2:i]
    
    regression_Y=cum_return.loc[temp_period].fillna(0)
    regression_X=sm.add_constant(index_cum_return.loc[temp_period]).fillna(0) 
    est=sm.OLS(regression_Y.values,regression_X.values)
    est=est.fit() 
    unsystematic_risk.loc[cum_return.index[i]]=pd.DataFrame(est.resid).std().values   
    print cum_return.index[i]
    
unsystematic_risk.to_csv(u'%s\\unsystematic_risk.csv'%(save_direction),encoding= 'gb2312')

unsystematic_risk=pd.read_csv(u'%s\\unsystematic_risk.csv'%(save_direction),encoding= 'gb2312',index_col=0)
stock_score_unsys_reverse=Get_score(unsystematic_risk,False,'stock')   
stock_score_unsys_reverse.to_csv(u'%s\\score\\stock_score_unsys_reverse.csv'%(save_direction),encoding= 'gb2312')

# 用一致目标价-当前价格，差越大，排名越靠前，得分越高
target_price_fetched=w.wsd(str_hk_ticker,"targetprice_avg",firstday,lastday,"unit=1;westPeriod=30")
target_price =pd.DataFrame(index=target_price_fetched.Times)
for z in range(len(pd_hk_ticker)):
    # z=0
    temp=pd.DataFrame(target_price_fetched.Data[z],index=target_price_fetched.Times,columns=['target_price'])
    # 取得开始公开交易以后的数据
    trade_data=temp[temp['target_price'].isnull()==False]         
    if len(trade_data)!=0:
        if len(trade_data[trade_data['target_price'].isnull()==True])<=len(trade_data)*0.1:
            target_price[pd_hk_ticker['wind_code'].loc[z]]=target_price_fetched.Data[z]
            
target_price.to_csv(u'%s\\hk_target_price.csv'%(save_direction),encoding= 'gb2312')

target_price=pd.read_csv(u'%s\\hk_target_price.csv'%(save_direction),encoding= 'gb2312',index_col=0)
price_gap=target_price-daily_close
stock_score_target_momentum=Get_score(price_gap,False,'stock')  
stock_score_target_momentum.to_csv(u'%s\\score\\stock_score_target_momentum.csv'%(save_direction),encoding= 'gb2312')

######################################################################################

# 每日的涨跌停情况, 港股没有涨跌停
'''
maxupdown=data_fetch(list_hk_ticker,pd_hk_ticker,"maxupordown",0.1,firstday,lastday,'D')
maxupdown.to_csv(u'%s\\hk_maxupdown.csv'%(save_direction),encoding= 'gb2312')
'''


# 计算每只个股样本区间内每个交易日的总分数
ind_momentum=pd.read_csv(u'%s\\score\\stock_score_ind_momentum.csv'%(save_direction),encoding= 'gb2312',index_col=0)
ind_reverse=pd.read_csv(u'%s\\score\\stock_score_ind_reverse.csv'%(save_direction),encoding= 'gb2312',index_col=0)
illiquidity_reverse=pd.read_csv(u'%s\\score\\stock_score_illiquidity_reverse.csv'%(save_direction),encoding= 'gb2312',index_col=0)
unsys_reverse=pd.read_csv(u'%s\\score\\stock_score_unsys_reverse.csv'%(save_direction),encoding= 'gb2312',index_col=0)
target_momentum=pd.read_csv(u'%s\\score\\stock_score_target_momentum.csv'%(save_direction),encoding= 'gb2312',index_col=0)

final_daily_score=1.5*ind_momentum+1.5*ind_reverse+3*target_momentum+2*illiquidity_reverse+unsys_reverse

# 对冲的指数
hk_index=pd.read_csv(u'%s\\hk_market_index.csv'%(save_direction),encoding= 'gb2312',index_col=0)

# 组合仓位
port_member_num=25
# 每次调仓数量
change_position=2
# 单边个股交易费用
fee=0.002
# 建仓变量
formation=False
# 初始资金
capital_stock=1.0

# 计算股指期货的净值




# pool
port_NV=pd.DataFrame(capital_stock,index=final_daily_score.index,columns=['NV'])
weight=pd.DataFrame(index=final_daily_score.index,columns=final_daily_score.columns)
portfolio=pd.DataFrame(index=final_daily_score.index,columns=['port_members'])
formation_price=pd.DataFrame(index=final_daily_score.columns,columns=['formation_price'])


for i in range(len(final_daily_score)):
    #print '%s次循环开始'%(i)
    # i=30
    sort_score=final_daily_score.loc[final_daily_score.index[i]].sort_values()
    test=members_sort(sort_score,port_member_num)
 
    if len(test.dropna())==port_member_num and not formation:
        formation=True
        temp_port=test.index
        portfolio.at[portfolio.index[i],'port_members']=list(temp_port)
		
        # 建仓当天的组合成分股的开盘价，收盘价
        foramtion_open=daily_open.loc[daily_open.index[i]][temp_port]
        formation_close=daily_close.loc[daily_close.index[i]][temp_port]
        # 保存建仓价格
        for m in temp_port:
            formation_price.at[m,'formation_price']=daily_open.at[daily_open.index[i],m]
		
        # 减去建仓成本后的净值
        port_NV.at[port_NV.index[i],'NV']=pd.DataFrame(capital_stock*(1-fee)*formation_close/foramtion_open/port_member_num).sum().values[0]
        # 建仓初始的组合成分股权重            
        weight.loc[weight.index[i]][temp_port]=capital_stock*(1-fee)/port_member_num
        continue
         
    if formation:#调仓
        temp_port=portfolio.at[portfolio.index[i-1],'port_members']#list
        port_score=sort_score.loc[temp_port]
        stock_out=members_sort(port_score,2,False).index.tolist()
        pool_score=sort_score.drop(temp_port)
        stock_in=members_sort(pool_score,2,True).index.tolist()
        # 加入的stock的formation price
        for m in stock_in:
            # 如果没有开盘价，用前一天的收盘价
            if not daily_open.at[daily_open.index[i],m]>0:
                temp_formation_open=daily_close.at[daily_close.index[i-1],m]
            else:    
                formation_price.at[m,'formation_price']=daily_open.at[daily_open.index[i],m]       
        # 新的portfolio
        portfolio.at[portfolio.index[i],'port_members']=position_change(temp_port,stock_in,stock_out)

        # 计算卖出股票的闲钱（减去单边交易成本）

        if not daily_open.loc[daily_open.index[i]][stock_out].any()>0:
            out_open_2=daily_close.loc[daily_close.index[i-1]][stock_out]
        else:
            out_open_2=daily_open.loc[daily_open.index[i]][stock_out]
        out_formation_price=formation_price.loc[stock_out]['formation_price']
        out_weight=weight.loc[weight.index[i-1]][stock_out]
        cash=pd.DataFrame(out_weight*out_open_2*(1-fee)/out_formation_price).sum()
		
        #用cash各买入调入股票（减去单边交易成本），计算买入股票的净值
        in_formation_price=formation_price.loc[stock_in]['formation_price']
        in_close_2=daily_close.loc[daily_close.index[i]][stock_in]
        NV_in=cash.values*in_close_2.values/2/in_formation_price.values.sum()
        NV_in=sum(NV_in)
		
        
        # 计算调入股票的weight
        weight.loc[weight.index[i]][stock_in]=float(cash/2)
        # 仍在组合中的memebers的formation close
        stay_port=position_change(temp_port,[],stock_out)
        # 仍在组合中的成分股净值
        new_close=daily_close[stay_port].loc[daily_close.index[i]]
        # 仍在组和中的成分股formation_close
        stay_formation_price=formation_price.loc[stay_port]
        # 仍在组合中的净值
        NV_stay=(weight[stay_port].loc[weight.index[i-1]]*new_close/stay_formation_price['formation_price']).sum()
        # 其他成分股weight不变
        weight.loc[weight.index[i]][stay_port]= weight.loc[weight.index[i-1]][stay_port]
        #assert float(NV_stay)+float(NV_in)!=None

        port_NV.at[port_NV.index[i],'NV']=float(NV_stay)+float(NV_in)
        '''
        print ('当i=%s'%(i))
        print ('新组合成分股个数：%s'%(len(portfolio.at[portfolio.index[i],'port_members'])))
        if len(pd.DataFrame(portfolio.at[portfolio.index[i],'port_members']).drop_duplicates())!=25:
            print ('有重复')
        print ('调入个股数：%s'%(len(stock_in)))
        print ('调出个股数：%s'%(len(stock_out)))
        print '------------------------------------------------------------------'
        '''
        if final_daily_score.index[i].encode('utf-8')[0:7]!=final_daily_score.index[i-1].encode('utf-8')[0:7]:
            print final_daily_score.index[i].encode('utf-8')[0:7]

    
    
portfolio.to_csv(u'%s\\result\\long_only\\portforlio_members.csv'%(save_direction),encoding= 'gb2312')
port_NV.to_csv(u'%s\\result\\long_only\\port_nv.csv'%(save_direction),encoding= 'gb2312')
    
port_NV.plot()

'''
######################################################################################
'''
# datatype="close","pct_chg","volume"
def data_fetch(ticker_list,ticker_frame,datatype,n,firstday,lastday,fre):
    data_fetched=w.wsd(ticker_list, datatype,firstday,lastday,"unit=1;Period=%s"%(fre))
    data =pd.DataFrame(index=data_fetched.Times)
    for z in range(len(ticker_frame)):
        # z=0
        temp=pd.DataFrame(data_fetched.Data[z],index=data_fetched.Times,columns=[datatype])
        # 取得开始公开交易以后的数据
        trade_data=temp[temp[datatype].isnull()==False]         
        # 次日期后交易日 没有数据的比例在可接受范围内的ETF ticker 
        if len(trade_data)!=0:
            if len(trade_data[trade_data[datatype].isnull()==True])<=len(trade_data)*n:
                data[ticker_frame['wind_code'].loc[z]]=data_fetched.Data[z]
    print ('%s Got!'%(datatype))
    return data
# 因为排名是从小到大sort，因此input_data reverse=True,数值小的排名靠前，得分小
# reverse=False, 数值大的得分小

def Get_score(input_data,reverse,datatype):   
    score_output=pd.DataFrame(index=input_data.index,columns=input_data.columns)
    for i in input_data.index:
        # i=input_data.index[0]
        if datatype=='stock':
            # 当天没有成交价的去掉
            trading_stock=daily_close.loc[i][daily_close.loc[i].fillna(0)!=0].index
            input_data_2=input_data[trading_stock]
            sort_data=pd.DataFrame(input_data_2.loc[i].values,index=input_data_2.columns,columns=['sort_type'])
        else:
            sort_data=pd.DataFrame(input_data.loc[i].values,index=input_data.columns,columns=['sort_type'])
        score=sort_data.sort(columns='sort_type')
        ll=len(sort_data)
        # score标准化
        if reverse==True:
            score['score']=range(1,ll+1)
        if reverse==False:
            score['score']=sorted(range(1,ll+1),cmp=None, key=None, reverse=True)
        score['score']=score['score']/ll
        for j in score.index:
            score_output.at[i,j]=score.at[j,'score']
        print i
    return score_output

# 排序函数，取seriers的前head个数据
def members_sort(series,head,asc=True):
	ans=series.sort_values(ascending=asc).head(head)
	return ans
 
# 调仓函数
def position_change(lst,inn,out):
	temp=lst+inn
	for a in out:
		temp.remove(a)
	return temp   
    
    
    
    
    
    
    
    