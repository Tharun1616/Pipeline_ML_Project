from flask import Flask, jsonify, render_template, request, redirect, url_for
import pandas as pd
import os
from src.components import data_ingestion   # Import your data ingestion module
from src.components import data_transformation
from src.components import data_modeling
os.environ["HTTP_PROXY"] = "<yourproxy>"


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch_stock', methods=['GET', 'POST'])
def fetch_stock():
    message = None
    s_data = None
    top10_data = None
    bot10_data = None
    description_data = None
    ticker = request.form.get('ticker')

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
                message = "Please enter a valid stock ticker."
                return render_template('index.html')
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

    return render_template('stock_data.html', message=message, ticker=ticker, stock_data=s_data, table_html=table_html, top10_data=top10_data, bot10_data=bot10_data, description_data=description_data)


@app.route('/visual', methods=['GET', 'POST'])
def visual():
    vol_close_chart_path = data_transformation.vol_close_chart()
    candlestick_chart_path = data_transformation.candlestick_chart()
    sma_chart_path = data_transformation.SMA_chart()
    ema_chart_path = data_transformation.EMA_chart()

    return render_template('charts.html', close_vol_chart=vol_close_chart_path, candlestick_chart=candlestick_chart_path, sma_chart=sma_chart_path, ema_chart=ema_chart_path)


@app.route('/indicators')
def indicators():
    sma_bb_chart = data_transformation.sma_BB()
    ema_bb_chart = data_transformation.ema_BB()
    ichimoku_chart_path = data_transformation.ichimoku()
    return render_template('indicators.html', sma_bb=sma_bb_chart, ema_bb=ema_bb_chart, ichimoku_chart=ichimoku_chart_path)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    train_rmse, test_rmse, act_vs_pred_chart = data_modeling.test_v_act()
    future_data = data_modeling.forecasting()
    return render_template('prediction.html', train_rmse=train_rmse, test_rmse=test_rmse, act_vs_pred_chart=act_vs_pred_chart, future_data=future_data)


if __name__ == '__main__':
    app.run(debug=True)
