
import os
import sys

from src.logger import logging
import pandas as pd
import numpy as np
from src.exception import CustmeException
#from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import yfinance as yf


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path = os.path.join("artifacts/data_ingestion", "test.csv")
    raw_data_path = os.path.join("artifacts/data_ingestion", "raw.csv")


class DataIngestion():

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_ingestion(self, stock_name):
        logging.info("Data Ingestion started")
        try:
            logging.info("Data Reading using Pandas library from local system")
            data = yf.download(tickers=stock_name, period="5y", interval='1d')
            print(data.tail(2))
            # data = pd.read_csv(os.path.join("notebook/data", "bats_symbols.csv"))
            logging.info("Data Reading completed")

            ''' if na then replacing it with previous and after closing price '''

            # data['imputed_price'] = data['Close']

            # for i in range(1, len(data) - 1):
            #     if pd.isna(data.at[i, 'Close']):
            #         data.at[i, 'imputed_price'] = (
            #             data.at[i - 1, 'Close'] + data.at[i + 1, 'Close']) / 2
            #         logging.info("Null value imputated at", i)

            # ''' dropping the price column and renaming the imputated_price into price'''
            # data.drop(columns=['Close'], inplace=True)
            # data.rename(columns={"imputed_price", "Close"}, inplace=True)

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=True)
            logging.info("Dat spliteted into train and test")

            train_size = int(len(data) * 0.8)
            train_set = data[:train_size]
            test_set = data[train_size:]

            # train_set, test_set = train_test_split(data, test_size=.30, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=True, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=True, header=True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.info("Erro occured in data ingestion stage")
            raise CustmeException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.inititate_data_ingestion()
