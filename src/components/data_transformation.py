import os
import sys

from src.exception import CustmeException
from src.logger import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import pandas_ta as pta
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

stock_dataset = pd.read_csv(os.path.join(
    "artifacts/data_ingestion", "raw.csv"))

static_dir = 'static'

print(stock_dataset.tail(2))

def vol_close_chart():
    fig = go.Figure()
    stock_plot = go.Scatter(
        x=stock_dataset['Date'], y=stock_dataset['Close'], name="Close_Price")
    
    print(stock_dataset.tail(2))

    # Plot price changes
    fig.add_trace(stock_plot)

    # Plot volume as bar graph
    fig.add_trace(go.Bar(x=stock_dataset['Date'],
                  y=stock_dataset['Volume']/200000, name='Volume Traded'))

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    #fig.update_layout(height=600, width=1200, showlegend=True)
    fig.update_layout(title='Close-Volume Chart')
    chart_filename = 'close_vol_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    # chart_path = "static/close_vol_chart.html"
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path


candle = go.Candlestick(x=stock_dataset['Date'], open=stock_dataset['Open'],
                        high=stock_dataset['High'], low=stock_dataset['Low'],
                        close=stock_dataset['Close'], name='Candlestick')


def candlestick_chart():
    fig = go.Figure()

    # Define candlestick and moving average lines

    fig.add_trace(candle)
    fig.update_layout(title='CandleStick Chart')
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    chart_filename = 'candlestick_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path


sma_periods = [20, 50, 100, 200]
ema_periods = [5, 9, 21, 50, 100, 200]


def SMA(data, period, column='Close'):
    return data[column].rolling(window=period).mean()


def EMA(data, period, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()

# Simple Moving Avg


def SMA_chart():

    fig = go.Figure()
    sma20 = go.Scatter(x=stock_dataset['Date'], y=SMA(stock_dataset, period=sma_periods[0]),
                       line=dict(color='green', width=1), name="SMA20")

    sma50 = go.Scatter(x=stock_dataset['Date'], y=SMA(stock_dataset, period=sma_periods[1]),
                       line=dict(color='orange', width=1), name="SMA50")

    sma100 = go.Scatter(x=stock_dataset['Date'], y=SMA(stock_dataset, period=sma_periods[2]),
                        line=dict(color='violet', width=1), name="SMA100")

    sma200 = go.Scatter(x=stock_dataset['Date'], y=SMA(stock_dataset, period=sma_periods[3]),
                        line=dict(color='yellow', width=1), name="SMA200")

    # Add plots to the figure
    fig.add_trace(candle)
    fig.add_trace(sma20)
    fig.add_trace(sma50)
    fig.add_trace(sma100)
    fig.add_trace(sma200)

    # Add title
    fig.update_layout(title="SMA Chart")

    # Get rid of empty dates on the weekend
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    chart_filename = 'SMA_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path

# Exponential Moving Avg


def EMA_chart():

    fig = go.Figure()

    ema5 = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=ema_periods[0]),
                      line=dict(color='green', width=1), name="EMA05")

    ema9 = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=ema_periods[1]),
                      line=dict(color='orange', width=1), name="EMA09")

    ema21 = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=ema_periods[2]),
                       line=dict(color='violet', width=1), name="EMA21")

    ema50 = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=ema_periods[3]),
                       line=dict(color='yellow', width=1), name="EMA50")

    ema100 = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=ema_periods[4]),
                        line=dict(color='brown', width=1), name="EMA100")

    ema200 = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=ema_periods[5]),
                        line=dict(color='blue', width=1), name="EMA200")

    # Add plots to the figure
    fig.add_trace(candle)
    fig.add_trace(ema5)
    fig.add_trace(ema9)
    fig.add_trace(ema21)
    fig.add_trace(ema50)
    fig.add_trace(ema100)
    fig.add_trace(ema200)
    # Add title
    fig.update_layout(title="EMA Chart")

    # Get rid of empty dates on the weekend
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    chart_filename = 'EMA_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path



## Custom Chart
# def custom_sma_ema(s_period, e_period):
#     fig = go.Figure()

#     sma = go.Scatter(x=stock_dataset['Date'], y=SMA(stock_dataset, period=s_period),
#                      line=dict(color='green', width=1), name="SMA")

#     ema = go.Scatter(x=stock_dataset['Date'], y=EMA(stock_dataset, period=e_period),
#                      line=dict(color='green', width=1), name="EMA")

#     fig.add_trace(candle)
#     fig.add_trace(sma)
#     fig.add_trace(ema)

#     fig.update_layout(title="Custom SMA+EMA Chart")

#     # Get rid of empty dates on the weekend
#     fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

#     chart_filename = 'SMA+EMA_chart.html'
#     chart_path = os.path.join(static_dir, chart_filename)
#     plot(fig, filename=chart_path, auto_open=False)
#     return chart_path


# Bollinger Bands


st_df = stock_dataset.copy()

# rolling std for calculating the higher band and lower band
st_df['SD'] = st_df['Close'].rolling(window=20).std()


def sma_BB():

    st_df['bb_sma_20'] = st_df['Close'].rolling(window=20).mean()

    # Higher and Lower Bands
    st_df['BB_Sma_High'] = st_df['bb_sma_20'] + (2 * st_df['SD'])
    st_df['BB_Sma_Low'] = st_df['bb_sma_20'] - (2 * st_df['SD'])

    fig = go.Figure()
    sma20 = go.Scatter(x=st_df['Date'], y=SMA(st_df, period=20),
                       line=dict(color='blue', width=1), name="SMA20")

    bb_high = go.Scatter(x=st_df['Date'], y=st_df['BB_Sma_High'],
                         line=dict(color='green', width=1), name="BB High")

    bb_low = go.Scatter(x=st_df['Date'], y=st_df['BB_Sma_Low'],
                        line=dict(color='orange', width=1), name="BB Low")

    # Add plots to the figure
    fig.add_trace(candle)
    fig.add_trace(sma20)
    fig.add_trace(bb_high)
    fig.add_trace(bb_low)

    # Add title
    fig.update_layout(title="Bollinger Bands(SMA)")

    # Get rid of empty dates on the weekend
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    chart_filename = 'SMA_BB_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path


def ema_BB():

    st_df['bb_ema_21'] = st_df['Close'].ewm(span=21, adjust=False).mean()

    # Higher and Lower Bands
    st_df['BB_Ema_High'] = st_df['bb_ema_21'] + (2 * st_df['SD'])
    st_df['BB_Ema_Low'] = st_df['bb_ema_21'] - (2 * st_df['SD'])

    fig = go.Figure()
    ema21 = go.Scatter(x=st_df['Date'], y=st_df['bb_ema_21'],
                       line=dict(color='blue', width=1), name="EMA21")

    bb_high = go.Scatter(x=st_df['Date'], y=st_df['BB_Ema_High'],
                         line=dict(color='green', width=1), name="BB High")

    bb_low = go.Scatter(x=st_df['Date'], y=st_df['BB_Ema_Low'],
                        line=dict(color='orange', width=1), name="BB Low")

    # Add plots to the figure
    fig.add_trace(candle)
    fig.add_trace(ema21)
    fig.add_trace(bb_high)
    fig.add_trace(bb_low)

    # Add title
    fig.update_layout(title="Bollinger Bands(EMA)")

    # Get rid of empty dates on the weekend
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    chart_filename = 'EMA_BB_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path


# Ichimoku

def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


def ichimoku():
    ic_df = stock_dataset.copy()

    ichimoku = pta.ichimoku(high=st_df['High'], low=st_df['Low'], close=st_df['Close'],
                            tenkan=9, kijun=26, senkou=52, include_chikou=True, offset=0)

    ic_df = pd.concat([ic_df, ichimoku[0]], axis=1)
    ic_df = pd.concat([ic_df, ichimoku[1]], axis=0)
    ic_df.rename(columns={'ITS_9': 'Conversion',
                          'IKS_26': 'Baseline',
                          'ISA_9': 'SpanA', 'ISB_26': 'SpanB',
                          'ICS_26': 'Lagging'}, inplace=True)

    # We will use 2 dataframes
    # 1 will contain the fill data between the spans
    # The other will be stored with the original data in df1
    df1 = ic_df.copy()
    amd_df = ic_df.copy()

    fig = go.Figure()

    # Where SpanA is greater than SpanB give label a value of 1 or 0 if not
    ##amd_df['label'] = np.where(amd_df['SpanA'] > amd_df['SpanB'], 1, 0)

    # Shift 1 period, compare dataframe for inequality with the cumulative
    # sum and store in group
    ##amd_df['group'] = amd_df['label'].ne(amd_df['label'].shift()).cumsum()

    # Get a groupby object that contains information on the group
    ##amd_df = amd_df.groupby('group')

    # Cycle through the data pertaining to the fill between spans
    # dfs = []
    # for name, data in amd_df:
    #     dfs.append(data)

    # Add 2 traces to the fig object for each time the spans cross
    # and then define the fill using fill='tonexty' for the second trace
    # for df in dfs:
    #     fig.add_traces(go.Scatter(x=df.index, y=df.SpanA,
    #                               line=dict(color='rgba(0,0,0,0)')))

    #     fig.add_traces(go.Scatter(x=df.index, y=df.SpanB,
    #                               line=dict(color='rgba(0,0,0,0)'),
    #                               fill='tonexty',
    #                               fillcolor=get_fill_color(df['label'].iloc[0])))

    # Create plots for all of the nonfill data
    baseline = go.Scatter(x=df1['Date'], y=df1['Baseline'],
                          line=dict(color='pink', width=2), name="Baseline")

    conversion = go.Scatter(x=df1['Date'], y=df1['Conversion'],
                            line=dict(color='black', width=1), name="Conversion")

    lagging = go.Scatter(x=df1['Date'], y=df1['Lagging'],
                         line=dict(color='purple', width=2, dash='dot'), name="Lagging")

    span_a = go.Scatter(x=df1['Date'], y=df1['SpanA'],
                        line=dict(color='green', width=2, dash='dot'), name="Span A")

    span_b = go.Scatter(x=df1['Date'], y=df1['SpanB'],
                        line=dict(color='red', width=1, dash='dot'), name="Span B")

    # Add plots to the figure
    fig.add_trace(candle)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)

    # Add title
    fig.update_layout(title="Ichimoku Cloud)")

    chart_filename = 'ichimoku_chart.html'
    chart_path = os.path.join(static_dir, chart_filename)
    plot(fig, filename=chart_path, auto_open=False)

    return chart_path
