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


