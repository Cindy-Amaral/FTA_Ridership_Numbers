import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet

#to run streamlit run main.py

st.title('Forecasting FTA Ridership Numbers')

#read in data
df = pd.read_csv('may2023_FTA_VRM.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')


#drop down menus for data selection
agency_list = df['Agency'].unique().tolist()

agency_select = st.selectbox(
    'Select Agency',agency_list)


mode_list = df.loc[df['Agency'] == agency_select]['Mode'].unique().tolist()

mode_select = st.selectbox(
    'Select Mode',mode_list)

#forecast
df2=df.loc[(df['Agency'] == agency_select)&(df['Mode']==mode_select)]
df2 = df2[['Date','Total']].copy().reset_index(drop=True)
df2.columns = ['ds', 'y']

m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df2)

future = m.make_future_dataframe(periods=12,freq='M')
forecast = m.predict(future)

merge = forecast.merge(df2, how='left', on='ds')
merge = merge[['ds','yhat','y']]
merge = merge[merge.y != 0]

st.write(merge)

#show forecast chart
st.line_chart(data=merge, x='ds', y=['yhat','y'])