#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#PREPROCESSING OF DATASETS TO ENABLE SUPERVISED LEARNING ANALYSIS OF CORPORATE DISCLOSURES (INPUT DATA) TO STOCK PRICES (LABELS)

import pandas as pd
import numpy as np
import re
from datetime import datetime
from datetime import timedelta


def read_documents(path):
    #Documents (input data)
    
    documents = pd.read_json(path)
    documents.drop('language', axis=1)
    documents.drop('doc_id', axis=1)
    
    return documents


def read_metadata(path):
    #Documents metadata, including company names. Company names are needed for matching with RIC numbers.

    metadata = pd.read_json(path)

    #Changing names of columns from Swedish to English to facilitate for readers to follow the code
    metadata['date_time'] = metadata['Datum (CET)']
    metadata['company'] = metadata['Bolag']
    metadata['headline'] = metadata['Ämne']
    metadata['category'] = metadata['Kategori']
    metadata.drop(['Datum (CET)', 'Bolag', 'Ämne', 'Kategori'], axis=1, inplace=True)

    #Changing column data type to datetime
    metadata['date_time'] = pd.to_datetime(metadata['date_time'])

    #Some disclosures are represented as multiple rows in the metadata scraped from Nasdaq. 
    #They have unique row ID's but identical document ID's. This code drops the columns containing row ID's, 
    #thereby making the duplicate rows completely identical,thus enabling "pd.drop_duplicates()".
    metadata.drop('web-scraper-start-url', axis=1, inplace=True)
    metadata.drop('web-scraper-order', axis=1, inplace=True)
    metadata = metadata.drop_duplicates()
    
    return metadata


def read_ric(path):
    
    ric = pd.read_csv(path, sep=';')

    #Cleaning the ric document company names (ric['company']) to enable partial match with doc_meta['company'].
    new = ric['company'].str.rsplit(expand=True, n=1)
    new[1].replace(to_replace=['A', 'B', 'C', 'D'], value='', inplace=True)
    new[1].fillna('', inplace=True)
    ric['company'] = new[0] + " " + new[1]
    ric['company'] = ric['company'].str.rstrip('.')
    ric['company'] = ric['company'].str.rstrip()

    #Enabling match of the company names "Byggmax Group First", "Sv. Handelsbanken", "Fast. Balder", "Samhällsbyggnadsbo. i Norden"
    ric['company'].replace(to_replace=[' First','Group', 'Sv. ', 'Fast. ', '. i Norden'], value='', inplace=True, regex=True)
    ric['company'].replace(to_replace='SEB ', value='Skandinaviska Enskilda Banken', inplace=True)

    #Converting all company names to lowercase.
    ric['company'] = ric['company'].str.lower()
    
    return ric


def read_stocks(path):
    
    stocks = pd.read_excel(path, sheet_name='Daily')
    stocks = stocks.drop([0,1,2])
    
    return stocks


def read_market_index(path):
    market_index = pd.read_csv(path, sep=';', decimal=',')
    market_index['Date'] = pd.to_datetime(market_index['Date'])
    
    return market_index


def read_risk_free_rate_data(path):

    risk_free_rate_data = pd.read_csv(path, sep=';', decimal=',')
    risk_free_rate_data.Period = pd.to_datetime(risk_free_rate_data.Period, dayfirst=True)
    risk_free_rate_data = risk_free_rate_data.sort_values(by='Period', ascending=False)
    risk_free_rate_data.set_index('Period', inplace=True)
    
    return risk_free_rate_data


def merge_docs_with_meta(documents, metadata):

    #Documents and Metadata merged/joined on headlines and dates
    doc_meta = documents.merge(metadata, left_on=['headline', 'date_time'], right_on=['headline', 'date_time'], how='inner')

    #Resetting the index to reduce the risk for incorrect iterations/filtrations.
    doc_meta=doc_meta.reset_index()
    doc_meta.drop('index', axis=1, inplace=True)

    #Changing all company names to lowercase. Needed for improving match percentage with ric['company']
    doc_meta['company']=doc_meta['company'].str.lower()
    
    return doc_meta


def add_ric(doc_meta_ric, ric):
    
    #Processing and matching a RIC number with each doc_meta row using company name as key.
    
    #For each row in the RIC document,
    #filter on rows in doc_meta where the company name of the RIC document equals to, 
    #or is partially equal to, the company name in doc_meta.
    for row in range(len(ric.company)):
        doc_meta_ric.loc[doc_meta_ric[doc_meta_ric.company.str.contains(ric.company.loc[row])].index,'RIC'] = ric.loc[row,'RIC']
    doc_meta_ric = doc_meta_ric.dropna()
    doc_meta_ric = doc_meta_ric.reset_index()
    doc_meta_ric.drop('index', axis=1, inplace=True)
    
    return doc_meta_ric


def add_stock_prices(doc_meta_ric_stocks, stocks):
    #Append stock prices to doc_meta_ric; dates + prices before and after document disclosure. Takes about 4 min

    time_series = pd.Index(stocks['Timestamp'])
    time_series = time_series.dropna()

    for doc_row in range(len(doc_meta_ric_stocks)):
        #The loop crashes when a variable (before or after) = *non-existent*. 
        #This happens for disclosures that are released on a particular date, 
        #but where there is no corresponding stock price on that date. 
        #This happens for stocks that are currently listed on OMX Stockholm, 
        #but have previously been listed on another stock exchange owned by Nasdaq. 
        #Press releases from dates when they were listed on another exchange are included in the documents dataset, 
        #but we have only managed to export stock prices from the period since they became listed on OMX Stockholm.
        
        #It also occurs for "before" dates, if a document was disclosed on the first day of its timeseries
        #at or before 17:30:00, since there is no "before" date on such occasions. The same goes for "after",
        #if a disclosure is published on 2020-03-11.
        
        #To solve this problem, "try" - when encountering non-existency, 
        #skip and instead continue with the next doc_row in the loop. 
        try:
            ric_label = doc_meta_ric_stocks.loc[doc_row]['RIC']
            timestamp = doc_meta_ric_stocks.loc[doc_row]['date_time']

            if timestamp >= datetime(timestamp.year, timestamp.month, timestamp.day, 0, 0, 0) and timestamp < datetime(timestamp.year, timestamp.month, timestamp.day, 17, 30, 0):
                before = stocks.iloc[time_series.get_loc(timestamp, method='nearest')+1]['Timestamp']
                after  = stocks.iloc[time_series.get_loc(timestamp, method='nearest')]['Timestamp']

            elif timestamp >= datetime(timestamp.year, timestamp.month, timestamp.day, 17, 30, 0) and timestamp <= datetime(timestamp.year, timestamp.month, timestamp.day, 23, 59, 59):
                before = stocks.iloc[time_series.get_loc(timestamp, method='nearest')]['Timestamp']
                after = stocks.iloc[time_series.get_loc(timestamp, method='nearest')-1]['Timestamp']
            else:
                print('error')

            before_price = stocks.iloc[time_series.get_loc(before), stocks.columns.get_loc(ric_label)]
            after_price = stocks.iloc[time_series.get_loc(after), stocks.columns.get_loc(ric_label)]

            doc_meta_ric_stocks.loc[doc_row,'before'] = before
            doc_meta_ric_stocks.loc[doc_row,'before_price'] = before_price
            doc_meta_ric_stocks.loc[doc_row,'after'] = after
            doc_meta_ric_stocks.loc[doc_row,'after_price'] = after_price

        except:
            continue
    
    doc_meta_ric_stocks = doc_meta_ric_stocks.dropna()
    doc_meta_ric_stocks.reset_index(inplace=True)
    doc_meta_ric_stocks.drop('index', axis=1, inplace=True)
    
    return doc_meta_ric_stocks


def add_market_data(dataset, market_index):

    #Append market data to the dataset - OMX GI value for before and after dates

    time_series = pd.Index(market_index.Date)
    time_series = time_series.dropna()

    for doc_row in range(len(dataset)):

        date_before = dataset.loc[doc_row, 'before']
        index_before = market_index.loc[time_series.get_loc(date_before), 'Trade_Close_daily']

        date_after = dataset.loc[doc_row, 'after']
        index_after = market_index.loc[time_series.get_loc(date_after), 'Trade_Close_daily']

        dataset.loc[doc_row, 'index_before'] = index_before
        dataset.loc[doc_row, 'index_after'] = index_after
    
    return dataset


def price_movement(dataset):
    price_mov = (dataset['after_price']/dataset['before_price'])-1
    dataset['PM'] = price_mov
    return dataset

def price_movement_index_adjusted(dataset):
    #Price movement equals stock movement adjusted for market index movement
    price_mov = (dataset['after_price']/dataset['before_price'])
    index_mov = (dataset['index_after']/dataset['index_before'])
    dataset['PM_index_adjusted'] = price_mov-index_mov
    return dataset


def jensens_alpha(dataset, stocks, market_index, risk_free_rate_data):
    count=0
    for row in range(len(dataset)):

        ric_label = dataset.loc[row]['RIC']
        timestamp = dataset.loc[row]['before']

        time_series = pd.Index(stocks['Timestamp'])
        time_series = time_series.dropna()

        calculation_period = stocks.iloc[time_series.get_loc(timestamp, method='nearest'):time_series.get_loc(timestamp, method='nearest')+60]['Timestamp'] 
        stock_prices = pd.Series(stocks.loc[calculation_period.index][ric_label].values, index=calculation_period, name='stock_prices')
        market_prices = pd.Series(market_index[market_index['Date'].isin(calculation_period)]['Trade_Close_daily'].values, index=calculation_period, name='market_prices')

        period_prices = pd.concat([stock_prices, market_prices], axis=1)
        period_returns = period_prices.iloc[::-1].pct_change()
        covariance_matrix = period_returns.cov(min_periods=30)

        try:
            cov = covariance_matrix.loc['stock_prices','market_prices']
            market_var = covariance_matrix.loc['market_prices','market_prices']
            beta = cov/market_var

        except:
            pass
        
        #The dataset containing the risk-free return as exported from Riksbanken, expresses
        #the annual rate of return as percent. Before calculating the risk-free rate
        #like it is depicted in the Method section of the thesis, the value is transformed into decimal value.
        risk_free_rate = (((float(risk_free_rate_data.loc[str(dataset.iloc[row].after).split()[0],'Value'])/100)+1)**(1/360))-1
        market_return = (dataset.loc[row]['index_after']/dataset.loc[row]['index_before'])-1

        expected_return = (risk_free_rate + beta*(market_return-risk_free_rate))
        daily_return = (dataset.loc[row].after_price/dataset.loc[row].before_price)-1

        alpha = daily_return-expected_return
        
        if str(alpha) == 'nan':
            dataset.loc[row, 'alpha'] = dataset.loc[row, 'PM_index_adjusted']
        else:
            dataset.loc[row, 'beta'] = beta
            dataset.loc[row, 'rfr'] = risk_free_rate
            dataset.loc[row, 'mr'] = market_return
            dataset.loc[row, 'er'] = expected_return
            dataset.loc[row, 'alpha'] = alpha
    
    return dataset


def PM_label_dataset(dataset, threshold):
    dataset.loc[dataset['PM']>threshold, 'PM_label'] = 'Up'
    dataset.loc[dataset['PM']<-threshold, 'PM_label'] = 'Down'
    dataset.loc[(dataset['PM']<=threshold) & (dataset['PM']>=-threshold), 'PM_label'] = 'Stable'
    return dataset

def PM_index_adjusted_label_dataset(dataset, threshold):
    dataset.loc[dataset['PM_index_adjusted']>threshold, 'PM_index_adjusted_label'] = 'Up'
    dataset.loc[dataset['PM_index_adjusted']<-threshold, 'PM_index_adjusted_label'] = 'Down'
    dataset.loc[(dataset['PM_index_adjusted']<=threshold) & (dataset['PM_index_adjusted']>=-threshold), 'PM_index_adjusted_label'] = 'Stable'
    return dataset

def PM_alpha_label_dataset(dataset, threshold):
    dataset.loc[dataset['alpha']>threshold, 'alpha_label'] = 'Up'
    dataset.loc[dataset['alpha']<-threshold, 'alpha_label'] = 'Down'
    dataset.loc[(dataset['alpha']<=threshold) & (dataset['alpha']>=-threshold), 'alpha_label'] = 'Stable'
    return dataset


#________________________________________________________________________
#Paths

#------------------------------------------------------------
documents_path = '20200311-20080228.json'
metadata_path = 'metadata_20200325-20100106.json'
ric_path = 'RIC.csv'
stocks_path = 'stock_data.xlsx'
market_index_path = 'OMXS GI.csv'
risk_free_rate_data_path = 'Risk free rate Tbill 3 month.csv'

#________________________________________________________________________
#Functions

#Read data
#------------------------------------------------------------
documents = read_documents(documents_path)
metadata = read_metadata(metadata_path)
ric = read_ric(ric_path)
stocks = read_stocks(stocks_path)
market_index = read_market_index(market_index_path)
risk_free_rate_data = read_risk_free_rate_data(risk_free_rate_data_path)


#Three steps for merging documents with stock price movements
#------------------------------------------------------------
doc_meta = merge_docs_with_meta(documents, metadata)
doc_meta_ric = add_ric(doc_meta, ric)
doc_meta_ric_stocks = add_stock_prices(doc_meta_ric, stocks)

#Add market index data to dataset
#--------------------------------
dataset = doc_meta_ric_stocks.copy()
dataset = add_market_data(dataset, market_index)

#Calculate and add abnormal return to dataset.
#---------------------------------
dataset = price_movement(dataset)
dataset = price_movement_index_adjusted(dataset)
dataset = jensens_alpha(dataset, stocks, market_index, risk_free_rate_data)

Label dataset
#------------
def export(dataset):
    for threshold in np.arange(0.005, 0.05, 0.005):

        dataset = PM_label_dataset(dataset, threshold)
        dataset = PM_index_adjusted_label_dataset(dataset, threshold)
        dataset = PM_alpha_label_dataset(dataset, threshold)

        out_data = dataset[(dataset.date_time<'2020-01-01')]
        out_data = out_data.reset_index()
        out_data.drop('index', inplace=True, axis=1)

        percent = str(threshold*100)
        out_data.to_json(percent+'percent_threshold_20100101-20191231.json')

export(dataset)

