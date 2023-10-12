import pandas as pd
import numpy as np
import math
import os
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.base import BaseEstimator


class data_model:

    static_dir = 'static'

    # def __init__(self):
    #     self.test_data = None
    #     X_train = None
    #     self.y_train = None
    #     self.X_test = None
    #     self.y_test = None
    #     self.pred_values = None
    #     self.model = None

    def data_trans(self, train_data, test_data):

        def create_sequences(data, sequence_length):
            sequences = []
            targets = []
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:i+sequence_length])
                targets.append(data[i+sequence_length])
            return np.array(sequences), np.array(targets)

        scaler = StandardScaler()

        train_data['Close'] = scaler.fit_transform(
            train_data['Close'].values.reshape(-1, 1))
        test_data['Close'] = scaler.fit_transform(
            test_data['Close'].values.reshape(-1, 1))
        sequence_length = 30

        X_train, y_train = create_sequences(
            train_data['Close'], sequence_length)
        X_test, y_test = create_sequences(
            test_data['Close'], sequence_length)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        # self.test_data = test_data
        # self.X_test = X_test
        # self.y_test = y_test
        # X_train = X_train
        # self.y_train = y_train

        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True,
                  input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=32))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x=X_train, y=y_train, batch_size=50,
                  epochs=10, validation_data=(X_test, y_test))
        model = model

        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)
        pred_test = scaler.inverse_transform(pred_test)
        pred_train = scaler.inverse_transform(pred_train)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

        pred_values = pd.DataFrame(y_test_actual, columns=['actual_t_d'])
        pred_values['pred_t_d'] = pred_test
        rmse_test_lstm = math.sqrt(
            mean_squared_error(y_test_actual, pred_test))
        rmse_train_lstm = math.sqrt(
            mean_squared_error(y_train_actual, pred_train))
        pred_values = pred_values

        print("Data Modeling")
        print(test_data.tail(2))

        fig = go.Figure()

        pred_test_line = go.Scatter(x=test_data['Date'], y=pred_values['pred_t_d'],
                                    line=dict(color='green', width=1), name="Pred Test")

        actual_test_line = go.Scatter(x=test_data['Date'], y=pred_values['actual_t_d'],
                                      line=dict(color='orange', width=1), name="Actual Test")

        # Add plots to the figure
        fig.add_trace(pred_test_line)
        fig.add_trace(actual_test_line)

        # Add title
        fig.update_layout(title="Predicted Test Vs Actual Test ")
        chart_filename = 'p_v_a_chart.html'
        chart_path = os.path.join(self.static_dir, chart_filename)
        plot(fig, filename=chart_path, auto_open=False)

        # return chart_path

        t_test = test_data['Close'].values

        input_data = t_test[len(t_test)-sequence_length:].reshape(1, -1)
        temp_input_data = input_data.tolist()
        temp_input_data = temp_input_data[0]
        output_data = []

        i = 0
        prediction_days = 30
        while(i < prediction_days):

            if(len(temp_input_data) > sequence_length):
                input_data = np.array(temp_input_data[1:])
                input_data = input_data.reshape(1, -1)
                input_data = input_data.reshape(1, sequence_length, 1)
                predicted_data = model.predict(input_data, verbose=0)
                temp_input_data.extend(predicted_data[0].tolist())
                temp_input_data = temp_input_data[1:]
                output_data.extend(predicted_data.tolist())
                i += 1
            else:
                input_data = input_data.reshape(1, sequence_length, 1)
                predicted_data = model.predict(input_data, verbose=0)
                temp_input_data.extend(predicted_data[0].tolist())
                output_data.extend(predicted_data.tolist())
                i += 1

        print(len(output_data))
        future_data = scaler.inverse_transform(
            output_data).reshape(1, -1).tolist()[0]

        return rmse_test_lstm, rmse_train_lstm, chart_path, future_data
