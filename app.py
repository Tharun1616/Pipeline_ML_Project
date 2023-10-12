from flask import Flask, jsonify, render_template, request, session, redirect
import pandas as pd
import os
from src.components import data_ingestion   # Import your data ingestion module
from src.components import data_transformation
import plotly.graph_objects as go
from src.components import data_modeling
os.environ["HTTP_PROXY"] = "<yourproxy>"


app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set_ticker', methods=['POST'])
def set_ticker():
    ticker = request.form.get('ticker')
    session['ticker'] = ticker  # Store the ticker in the session
    return redirect('/fetch_stock')


@app.route('/fetch_stock', methods=['GET', 'POST'])
def fetch_stock():
    message = None
    global s_data
    top10_data = None
    bot10_data = None
    description_data = None
    print("hi")
    # ticker = request.form.get('ticker')
    ticker = session.get('ticker')
    print(ticker)
    # session['ticker'] = ticker

    if ticker:
        try:
            obj = data_ingestion.DataIngestion()
            obj.inititate_data_ingestion(ticker)
            s_data = pd.read_csv(os.path.join(
                "artifacts/data_ingestion", "raw.csv"))
            table_html = s_data.to_html(
                classes=['table', 'table-dark', 'table-striped', 'table-bordered'])
            if len(s_data) != 0:
                message = "Stock Data Downloaded Sucessfully"
            else:
                not_found_message = "Please enter a valid stock ticker."
                return render_template('not_found.html', not_found_message=not_found_message)
            # Get top 10 rows and description
            top10_data = s_data.head(10).to_html(
                classes=['table', 'table-dark', 'table-striped', 'table-bordered'])
            bot10_data = s_data.tail(10).to_html(
                classes=['table', 'table-dark', 'table-striped', 'table-bordered'])
            description_data = s_data.describe().to_html(
                classes=['table', 'table-dark', 'table-striped', 'table-bordered'])
        except Exception as e:
            s_data = None
            message = f"Error: {str(e)}"
    else:
        return redirect('/')

    return render_template('stock_data.html', message=message, ticker=ticker, stock_data=s_data, table_html=table_html, top10_data=top10_data, bot10_data=bot10_data, description_data=description_data)


@app.route('/visual', methods=['GET', 'POST'])
def visual():
    ticker = session.get('ticker')
    s_data = pd.read_csv(os.path.join(
        "artifacts/data_ingestion", "raw.csv"))
    candle = go.Candlestick(x=s_data['Date'], open=s_data['Open'],
                            high=s_data['High'], low=s_data['Low'],
                            close=s_data['Close'], name='Candlestick')
    obj_t = data_transformation.DataTransformation()
    vol_close_chart_path = obj_t.vol_close_chart(ticker, s_data)
    candlestick_chart_path = obj_t.candlestick_chart(ticker, candle)
    sma_chart_path = obj_t.SMA_chart(ticker, s_data, candle)
    ema_chart_path = obj_t.EMA_chart(ticker, s_data, candle)

    return render_template('charts.html', ticker=ticker, close_vol_chart=vol_close_chart_path, candlestick_chart=candlestick_chart_path, sma_chart=sma_chart_path, ema_chart=ema_chart_path)


@app.route('/indicators')
def indicators():
    ticker = session.get('ticker')
    s_data = pd.read_csv(os.path.join(
        "artifacts/data_ingestion", "raw.csv"))
    candle = go.Candlestick(x=s_data['Date'], open=s_data['Open'],
                            high=s_data['High'], low=s_data['Low'],
                            close=s_data['Close'], name='Candlestick')
    obj_t = data_transformation.DataTransformation()
    sma_bb_chart = obj_t.sma_BB(s_data, candle)
    ema_bb_chart = obj_t.ema_BB(s_data, candle)
    ichimoku_chart_path = obj_t.ichimoku(s_data, candle)
    return render_template('indicators.html', ticker=ticker, sma_bb=sma_bb_chart, ema_bb=ema_bb_chart, ichimoku_chart=ichimoku_chart_path)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    train_data = pd.read_csv(os.path.join(
        "artifacts/data_ingestion", "train.csv"))
    test_data = pd.read_csv(os.path.join(
        "artifacts/data_ingestion", "test.csv"))
    ticker = session.get('ticker')
    obj_p = data_modeling.data_model()
    rmse_test_lstm, rmse_train_lstm, act_vs_pred_chart, future_data = obj_p.data_trans(
        train_data, test_data)
    return render_template('prediction.html', train_rmse=rmse_train_lstm, test_rmse=rmse_test_lstm, act_vs_pred_chart=act_vs_pred_chart, future_data=future_data)


if __name__ == '__main__':
    app.run(debug=True)
