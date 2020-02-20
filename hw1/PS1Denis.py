# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wrds, psycopg2
#db = wrds.Connection(wrds_username='denisste')

aapl=db.raw_sql("select date, ret from crsp.dsf where permco in (7) and date>='2001-01-01' and date<='2018-12-31'")
gs=db.raw_sql("select date, ret from crsp.dsf where permco in (35048) and date>='2001-01-01' and date<='2018-12-31'")
msft=db.raw_sql("select date, ret from crsp.dsf where permco in (8048) and date>='2001-01-01' and date<='2018-12-31'")
pg=db.raw_sql("select date, ret from crsp.dsf where permco in (21446) and date>='2001-01-01' and date<='2018-12-31'")
ge=db.raw_sql("select date, ret from crsp.dsf where permco in (20792) and date>='2001-01-01' and date<='2018-12-31'")
aapl.to_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/AAPL.csv")
gs.to_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/GS.csv")
msft.to_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/MSFT.csv")
pg.to_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/PG.csv")
ge.to_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/GE.csv")

AAPL = pd.read_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/AAPL.csv")
GS = pd.read_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/GS.csv")
MSFT = pd.read_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/MSFT.csv")
PG = pd.read_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/PG.csv")
GE = pd.read_csv("/Users/denis.steffen/Documents/Etudes supérieures/Master/Investments/GE.csv")

def generate_Geom_BrownianMotion(n,S0,mu,delta,sigma):
    S = np.zeros(n)
    S[0]=S0
    for i in np.arange(1,n):
        S[i] = S[i-1]*(1 + mu*delta + np.sqrt(delta)*sigma * np.random.randn())
    return(S)
    

mu = 0.06
sigma = 0.2
n = 3650
delta = 10/n
np.random.seed(100)

S = np.zeros(n)
S[0]=1
for i in np.arange(1,n):
    S[i] = S[i-1]*(1 + mu*delta + sigma * np.sqrt(delta)* np.random.randn())
    
plt.plot(S)
plt.title("Share Price")
plt.show()

R = (S[1:]-S[:-1])/S[:-1]
r = np.log(1+R)

plt.plot(r)
plt.title("Daily log-returns")
plt.show()

annual_mean = r.mean()/(delta)
print("Annualized mean = ")
print(annual_mean)

variance = ((r-annual_mean*delta)**2).sum()/((n-1)*delta)

sd = np.sqrt(variance)
print("Standard deviation = ")
print(sd)


n = 25202
delta = 68/n
prices = pd.Series(generate_Geom_BrownianMotion(n,1,mu,delta,sigma),pd.period_range(start = '1950-01-01',end = '2018-12-31'))
prices.plot()

S_monthly= prices.resample('M').mean()
S_monthly.plot()

S_weekly = prices.resample('W').mean()
S_weekly.plot()
plt.show()

r = np.log(1+(prices.values[1:]-prices.values[:-1])/prices.values[:-1])
R_monthly = (S_monthly.values[1:]-S_monthly.values[:-1])/S_monthly[:-1]
R_weekly = (S_weekly.values[1:]-S_weekly.values[:-1])/S_weekly[:-1]
r_monthly = pd.Series(np.log(1+R_monthly.values),pd.period_range(start = '1950-02-01',end = '2018-12-31',freq = 'M'))
r_weekly = pd.Series(np.log(1+R_weekly),pd.period_range(start = '1950-01-01',end = '2018-12-31',freq = 'W'))

r_monthly.plot()
r_weekly.plot()
plt.show()

r_monthly.describe()
r_weekly.describe()

annual_mean_total = r.mean()/delta
annual_mean_monthly = r_monthly.mean()*12 # I am not sure about which n divides the sum, n is the number of observations so here the number of months
annual_mean_weekly = r_weekly.mean()*52
print("Annualized mean of weekly data = ")
print(annual_mean_weekly)
print("Annualized mean of monthly data = ")
print(annual_mean_monthly)


variance_monthly = ((r_monthly-annual_mean_monthly*delta)**2).sum()/((n-1)*delta)
sd_monthly = np.sqrt(variance_monthly)

variance_weekly = ((r_weekly-annual_mean_weekly*delta)**2).sum()/((n-1)*delta)
sd_weekly = np.sqrt(variance_weekly)

print("Standard Deviation of weekly data = ")
print(sd_weekly)
print("Standard Deviation of monthly data = ")
print(sd_monthly)

# rolling mean:

rolling_monthly = r_monthly.rolling(12).mean().copy()
rolling_weekly = r_weekly.rolling(52).mean().copy()

annual_mean_rolling_monthly = rolling_monthly.resample('Y').mean()*12
annual_mean_rolling_monthly.plot()

rolling_weekly.index = rolling_weekly.index.astype('datetime64[ns]') # change index type because of frequency problems between weeks and years
annual_mean_rolling_weekly = rolling_weekly.resample('Y').mean()*52 
annual_mean_rolling_weekly.plot()

r_annual = np.log(1+(prices.values[1:]-prices.values[:-1])/prices[:-1])
annual_mean = r_annual.resample('Y').mean()
annual_mean.plot()
plt.show()

annual_sd = np.sqrt((r_annual**2).resample('Y').mean()) #no need to use delta, since it is annual
annual_sd.plot()
plt.show()

bins_mean = annual_mean.mean()
bins_mean_sd = np.sqrt(annual_mean.var())
bins_sd = annual_sd.mean()
bins_sd_sd = np.sqrt(annual_sd.var())

print(bins_mean,bins_mean_sd)
print(bins_sd,bins_sd_sd)


############## Ex 4

AAPL.index = pd.to_datetime(AAPL['date'])
AAPL = AAPL['ret']
MSFT.index = pd.to_datetime(MSFT['date'])
MSFT = MSFT['ret']
GS.index = pd.to_datetime(GS['date'])
GS = GS['ret']
GE.index = pd.to_datetime(GE['date'])
GE = GE['ret']
PG.index = pd.to_datetime(PG['date'])
PG = PG['ret']

def Problem3analysis(Data):
    temp = np.log(1+Data)
    n = np.shape(Data)[0]
    r_monthly= temp.resample('M').mean()
    r_weekly = temp.resample('W').mean()
    temp_summary = temp.describe()
    monthly_summary = r_monthly.describe()
    weekly_summary = r_weekly.describe()
    print(temp_summary)
    print(monthly_summary)
    print(weekly_summary)
    
    delta = 18/4527
    
    annual_mean = temp.mean()/delta
    annual_mean_monthly = r_monthly.mean()/delta
    annual_mean_weekly = r_weekly.mean()/delta
    print(annual_mean,annual_mean_monthly,annual_mean_weekly)
    
    sd_monthly = np.sqrt(((r_monthly-annual_mean_monthly*delta/10)**2).sum()/((n-1)*delta)*10)
    sd_weekly = np.sqrt(((r_weekly-annual_mean_weekly*delta/50)**2).sum()/((n-1)*delta)*50)
    #sd_weekly = np.sqrt(r_weekly.var()*52)
    #sd_monthly = np.sqrt(r_monthly.var()*12)
    sd = np.sqrt(((temp-annual_mean*delta)**2).sum()/((n-1)*delta))
    print(sd,sd_monthly,sd_weekly)
    
    rolling_monthly = r_monthly.rolling(12).mean().copy()
    rolling_weekly = r_weekly.rolling(52).mean().copy()
    
    annual_mean_rolling_monthly = rolling_monthly.resample('Y').sum()/(18) 
    annual_mean_rolling_monthly.plot()
    
    rolling_weekly.index = rolling_weekly.index.astype('datetime64[ns]') # change index type because of frequency problems between weeks and years
    annual_mean_rolling_weekly = rolling_weekly.resample('Y').sum()/18 
    annual_mean_rolling_weekly.plot()
    







