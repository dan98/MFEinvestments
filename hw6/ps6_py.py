#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 06:34:11 2020

@author: daniel
"""

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import wrds
from IPython.display import display


start = '2000-01-01'
end = '2019-12-31'

DOWNLOAD = False
if DOWNLOAD:
        
    # connect to databse and download csv files (run once)
    db = wrds.Connection(wrds_username = 'kuriyoki')
    db.create_pgpass_file() # run once

    # download brk data  
    vwret = db.raw_sql("select date, vwretd "
                     " from crsp.msi "
                     "where date>='{}' and date<='{}'".format(start, end))
    
    stonks = db.raw_sql("select a.permno, a.date, a.ret, b.shrcd, b.exchcd "
                     "from crsp.msf as a left join crsp.msenames as b "
                     "on a.permno=b.permno "
                     "and b.namedt <= a.date "
                     "and a.date <= b.nameendt "
                     "where b.exchcd in (1, 2) "
                     "and b.shrcd in (10, 11) "
                     "and date>='{}' and date<='{}'".format(start, end))
    
    riskfree = db.raw_sql("select mcaldt, tmytm "
                        "from crsp.tfz_mth_rf "
                        "where kytreasnox = 2000001 and "
                        "mcaldt>='{}' and mcaldt<='{}'".format(start, end))

    # index to datetime
    vwret = vwret.set_index('date')
    stonks = stonks.set_index('date')
    riskfree = riskfree.set_index('mcaldt')

    # write to csv
    vwret.to_csv('vwret.csv')
    stonks.to_csv('stonks.csv')
    riskfree.to_csv('riskfree.csv')
else:
    vwret = pd.read_csv('vwret.csv', index_col = 'date')
    stonks = pd.read_csv('stonks.csv', index_col = 'date')
    riskfree = pd.read_csv('riskfree.csv', index_col = 'mcaldt')
    

# Take only stocks with sufficient observations
nr_observations = stonks.groupby(['permno']).count()
sufficient_stocks = nr_observations[nr_observations.ret >= 240]
stonks = stonks[stonks['permno'].isin(sufficient_stocks.index)]

# Transform to excess returns


riskfree.columns = ['rf']
riskfree['rf'] = riskfree['rf']/100

# De-annualize
riskfree['rf'] = riskfree['rf']/12

riskfree.index = riskfree.index.rename('date')

stonks = stonks.merge(riskfree, left_on='date', right_on='date')
stonks = stonks.merge(vwret, left_on='date', right_on='date')

stonks['ret-rf'] = stonks['ret'] - stonks['rf']


def compute_beta(eret, mkt):
    #print(eret)
    #print(mkt)
    # Instead of the regression, can use the cov/var formula
    # regr = LinearRegression().fit(pd.DataFrame(mkt), eret)
    # print(regr.coef_[0])
    return eret.cov(mkt) / mkt.var()
    


stonks = stonks.sort_index()


full_sample_beta = stonks.groupby('permno')["ret-rf", "vwretd"].apply(lambda x:
                           compute_beta(x["ret-rf"], x["vwretd"]))
    
full_sample_beta.name = 'mkt_beta'

stonks = stonks.reset_index().merge(full_sample_beta, how='left', left_on=['permno'], right_on=['permno']).set_index('date')

#Compute beta based on first half

first_sample_beta = stonks.groupby('permno')["ret-rf", "vwretd"].apply(lambda x:
                           compute_beta(x["ret-rf"].loc['2000-01-31':'2009-12-31'], x["vwretd"].loc['2000-01-31':'2009-12-31']))
first_sample_beta.name = 'mkt_beta_first'
stonks = stonks.reset_index().merge(first_sample_beta, how='left', left_on=['permno'], right_on=['permno']).set_index('date')

#Compute beta based on second half

second_sample_beta = stonks.groupby('permno')["ret-rf", "vwretd"].apply(lambda x:
                           compute_beta(x["ret-rf"].loc['2010-01-31':'2019-12-31'], x["vwretd"].loc['2010-01-31':'2019-12-31']))
    
second_sample_beta.name = 'mkt_beta_second'
stonks = stonks.reset_index().merge(second_sample_beta, how='left', left_on=['permno'], right_on=['permno']).set_index('date')



#compute_beta(test_ret, vwret, riskfree)

#stonks.index = stonks.index.rename('date')

# cut in 10 decile bins
bins = stonks.groupby('date')['mkt_beta'].transform(lambda x: pd.qcut(x, 10, labels=False))
bins.name = 'bin'
stonks['bin'] = bins

# split in 10 deciles by first beta
bins_first = stonks.groupby('date')['mkt_beta_first'].transform(lambda x: pd.qcut(x, 10, labels=False))
bins_first.name = 'bin_first'
stonks['bin_first'] = bins_first

bin_means = stonks.groupby(['date', 'bin'])['ret-rf'].mean()
bin_means.name = 'bin_mean'


# Reset index first so that we don't lose it, put it back afterwards.
stonks = (stonks.reset_index()
.merge(bin_means, how='left', left_on=['date', 'bin'], right_on=['date', 'bin'])
.set_index('date'))

bin_means_first = stonks.loc['2010-01-31':'2019-12-31'].groupby(['date', 'bin_first'])['ret-rf'].mean()
bin_means_first.name = 'bin_mean_first'

stonks = (stonks.reset_index()
.merge(bin_means_first, how='left', left_on=['date', 'bin_first'], right_on=['date', 'bin_first'])
.set_index('date'))


bin_betas = stonks.groupby(['bin'])['mkt_beta'].mean()
stonks['bin_beta'] = bin_betas

bin_betas_first = stonks.groupby(['bin_first'])['mkt_beta_first'].mean()
stonks['bin_beta_first'] = bin_betas_first
bin_betas_second = stonks.groupby(['bin_first'])['mkt_beta_second'].mean()
stonks['bin_beta_second'] = bin_betas_second


bin_means_sample = bin_means.groupby('bin').mean()
bin_means_first_sample = bin_means_first.groupby('bin_first').mean()
m, b = np.polyfit(bin_betas, bin_means_sample, 1)

with plt.style.context('Solarize_Light2'):
    plt.plot(bin_betas, bin_means_sample, 'o')
    plt.plot(bin_betas, m*bin_betas + b)
    plt.title("Excess return vs. beta")
    plt.xlabel(r'$\beta_M$')
    plt.ylabel(r'$R_P - R_f$')
    plt.tight_layout()
    plt.savefig('plot1.eps', format='eps', transparent=True, pad_inches=0.4)
    plt.show()


print(m)
#Test CAPM
print(b)
print(stonks['vwretd'].mean() - stonks['rf'].mean())

m, b = np.polyfit(bin_betas_first, bin_means_first_sample, 1)

with plt.style.context('Solarize_Light2'):
    plt.plot(bin_betas_first, bin_means_first_sample, 'o')
    plt.plot(bin_betas_first, m*bin_betas_first + b)
    plt.title("Excess return vs. beta (first vs. second sample)")
    plt.xlabel(r'$\beta_M$' ', 2000-2010')
    plt.ylabel(r'$R_P - R_f$' + ', 2010-2019')
    plt.savefig('plot2.eps', format='eps', transparent=True, pad_inches=0.4)
    plt.show()

m, b = np.polyfit(bin_betas_first, bin_betas_second, 1)

with plt.style.context('Solarize_Light2'):
    print(m)
    plt.plot(bin_betas_first, bin_betas_second, 'o')
    plt.plot(bin_betas_first, m*bin_betas_second + b)
    plt.title("Excess return vs. beta (first vs. second sample)")
    plt.xlabel(r'$\beta_M$' ', 2000-2010')
    plt.ylabel(r'$\beta_M$' + ', 2010-2019')
    plt.savefig('plot3.eps', format='eps', transparent=True, pad_inches=0.4)
    plt.show()


#display(vwret)
#display(exchodes)
#display(exchodes)
















