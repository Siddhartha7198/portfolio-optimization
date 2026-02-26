#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:06:34 2026
Download daily adjusted prices of Assets: 1. SPDR S&P 500 ETF Trust (SPY), 2. Invesco QQQ Trust (QQQ), 
3. iShares Russell 2000 ETF (IWM), 4. iShares 20+ Year Treasury Bond ETF (TLT), 5. SPDR Gold Shares (GLD),
6. iShares MSCI EAFE ETF (EFA)

Data source: yfinance, Window: last 5 years, Frequency: Daily
@author: poddar
"""


import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataLoader:
    """
    Handles ETF price download and cleaning.
    """

    def __init__(self, tickers, years=5):
        self.tickers = tickers
        self.years = years

    def download_data(self):
        """
        Download adjusted close prices for the selected ETFs.
        """
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * self.years)
    
        data = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # important fix
        )
    
        # If multiple tickers, data columns are tickers directly
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data
    
        return prices

    def clean_data(self, prices):
        """
        Align dates and remove missing observations.
        """
        # Drop rows with any missing values
        prices = prices.dropna(how="any")

        # Sort index (ensure chronological order)
        prices = prices.sort_index()

        return prices

    def load(self):
        prices = self.download_data()
        prices = self.clean_data(prices)
        return prices