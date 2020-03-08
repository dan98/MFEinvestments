# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wrds, psycopg2
db = wrds.Connection(wrds_username='denisste')

data = db.raw_sql("select date,vwretd from crsp.msi where date>='1960-01-01'and date<='2019-12-31'", date_cols=['date'])
bonds= db.raw_sql("select caldt, b2ret from crsp.mcti where caldt>='1960-01-01' and caldt<='2019-12-31'", date_cols=['caldt'])

bill = db.raw_sql("select mcaldt, tmytm from crsp.tfz_mth_rf where kytreasnox = 2000001 and mcaldt>='1960-01-01' and mcaldt<='2019-12-31'")

data.index = pd.to_datetime(data['date'])
data = data['vwretd']
bonds.index = pd.to_datetime(bonds['caldt'])
bonds = bonds['b2ret']
bill.index = pd.to_datetime(bill['mcaldt'])
bill = np.exp(bill['tmytm']/(12*100))-1
print("CRSP mean and std")
print(data.mean())
print(data.std())
print("Weighted bond index mean and std")
print(bonds.mean())
print(bonds.std())
print("1-month treasury bill mean and std")
print(bill.mean())
print(bill.std())

R0 = bill.mean()
excess_data = data - R0
excess_bonds = bonds - R0

mu = np.array([data.mean(),bonds.mean(),R0])
Cov = np.cov([data.values,bonds.values,bill.values]) #covariance matrix between the 3 

Corr = np.linalg.inv(np.diag([data.std(),bonds.std(),bill.std()]))@Cov@np.linalg.inv(np.diag([data.std(),bonds.std(),bill.std()]))
A = (np.ones((1,3))@np.linalg.inv(Corr)@np.ones((3,1)))[0][0]
B = (np.ones((1,3))@np.linalg.inv(Corr)@mu.transpose())[0]
w_TAN = (1/(B-A*R0))*np.linalg.inv(Corr)@(mu.transpose()-R0*np.ones((3,)))

TAN_mean = np.dot(w_TAN,mu)
TAN_std = np.sqrt(w_TAN.transpose()@Cov@w_TAN)
TAN_SR = (TAN_mean - R0)/TAN_std #Shape Ratio

w_60_40 = np.array([0.6,0.4,0])
mean_60_40 = np.dot(w_60_40,mu)
std_60_40 = np.sqrt(w_60_40.transpose()@Cov@w_60_40)
SR_60_40 = (mean_60_40 - R0)/std_60_40 #Shape Ratio

w_RP_unlevered = np.array([1./data.std(),1./bonds.std(),0])
w_RP_unlevered /= np.linalg.norm(w_RP_unlevered,1)

RP_unlevered_mean = np.dot(w_RP_unlevered,mu)
RP_unlevered_std = np.sqrt(w_RP_unlevered.transpose()@Cov@w_RP_unlevered)
RP_unlevered_SR = (RP_unlevered_mean-R0)/RP_unlevered_std


w_RP = np.array([1./data.std(),1./bonds.std(),1./bill.std()])
w_RP /= np.linalg.norm(w_RP,1)
temp = np.sqrt(w_RP.transpose()@Cov@w_RP)
w_RP /= temp
w_RP *= std_60_40

RP_mean = np.dot(w_RP,mu)
RP_std = np.sqrt(w_RP.transpose()@Cov@w_RP)
RP_SR = (RP_mean-R0)/RP_std

plot_mean = np.array([TAN_mean,mean_60_40,RP_mean,RP_unlevered_mean])
plot_std = np.array([TAN_std,std_60_40,RP_std,RP_unlevered_std])
plt.plot(plot_std,plot_mean,'x')



