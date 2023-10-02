import pandas as pd
import numpy as np
from scipy.stats import spearmanr


class Backtesting(object):

    def __init__(self, price: pd.DataFrame):
        """
        Initialize the Backtest class.

        Args:
            price (pd.DataFrame): DataFrame containing price data.
        """
        self.price = price
        self.all_dates = self.price.index
        self.all_tickers = self.price.columns
        self.tradable_matrix = ~np.isnan(self.price) # only trade stocks that have price data
        self.all_dates_int = np.array([int(''.join(date.split('-'))) for date in self.all_dates])
        self.all_years_int = np.array([date_int // 10000 for date_int in self.all_dates_int])

    def get_result(self, signal: pd.DataFrame):
        """
        Perform backtesting and calculate performance statistics.

        Args:
            signal (pd.DataFrame): DataFrame containing trading signals.

        Returns:
            pd.DataFrame: DataFrame containing performance statistics.
        """
        signal = signal.copy()  # Make a copy of the input signal DataFrame to avoid modifying the original data
        daily_return = self.price / self.price.shift(1) - 1.  # Calculate the daily returns based on the price data
        daily_return[~self.tradable_matrix] = np.nan  # Set non-tradable positions to NaN in the daily returns DataFrame
        signal[~self.tradable_matrix] = np.nan  # Set non-tradable positions to NaN in the signal DataFrame

        transform = self.scale(self.winsorize(signal))  # Apply winsorization and scaling to the signal DataFrame to avoid outliers
        transform = transform / np.nansum(transform * (transform > 0), axis=1).reshape(
            (-1, 1))  # Normalize the transformed values across each row

        mask = np.nansum(~np.isnan(transform),
                         axis=1) == 0  # Create a mask to identify rows with all NaN values in the transformed DataFrame

        tot_rtn = np.nansum(pd.DataFrame(transform).shift(1).values * daily_return,
                            axis=1)  # Calculate the total return based on the transformed values and daily returns
        tot_rtn[mask] = np.nan  # Set total return to NaN for rows with all NaN values in the transformed DataFrame

        ix = np.array([i for i in range(len(tot_rtn)) if i % 10 == 0])  # Create an index array with every 10th element

        stats_dict = self.calculate_stats(tot_rtn[ix], self.all_dates_int[
            ix])  # Calculate performance statistics for the selected indices
        dict_list = []  # Initialize an empty list to store dictionaries of performance statistics

        unique_years = np.unique(self.all_years_int)  # Find the unique years in the dataset

        for unique_year in unique_years:
            mask = self.all_years_int == unique_year  # Create a mask to filter data for the current year
            cur_rtn = tot_rtn[mask]  # Filter total returns for the current year
            cur_dates = self.all_dates_int[mask]  # Filter dates for the current year
            ix = np.array(
                [i for i in range(len(cur_rtn)) if i % 10 == 0])  # Create an index array with every 10th element
            dict_list.append(self.calculate_stats(cur_rtn[ix], cur_dates[
                ix]))  # Calculate performance statistics for the selected indices and append to the list

        dict_list.append(stats_dict)  # Append the overall performance statistics to the list

        result = pd.DataFrame(dict_list, index=list(unique_years) + [
            'total'])  # Create a DataFrame from the list of performance statistics
        return result.round(2)  # Return the DataFrame with rounded values

    @staticmethod
    def calculate_ic(signal, daily_return):
        """
        Calculate the information coefficient (IC) between signals and returns.

        Args:
            signal (pd.DataFrame): DataFrame containing trading signals.
            daily_return (pd.DataFrame): DataFrame containing daily returns.

        Returns:
            float: Information coefficient.
        """
        corr_ts = []
        for i in range(len(signal) - 1):
            cur_signal = signal.iloc[i, :]
            cur_return = daily_return.iloc[i + 1, :]
            mask = ~np.isnan(cur_signal) & ~np.isnan(cur_return)
            cur_signal = cur_signal[mask]
            cur_return = cur_return[mask]
            corr_ts.append(spearmanr(cur_signal, cur_return))
        return np.mean(corr_ts)

    @staticmethod
    def calculate_stats(rtn, ds):
        """
        Calculate performance statistics.

        Args:
            rtn (np.ndarray): Array of returns.
            ds (np.ndarray): Array of dates.

        Returns:
            dict: Dictionary containing performance statistics.
        """
        ix = np.where(~np.isnan(rtn))[0][0]
        prd_rtn = np.nansum(rtn[ix:])
        ann_rtn = np.nanmean(rtn[ix:]) * 252
        ann_vol = np.nanstd(rtn[ix:]) * np.sqrt(252)
        shrp = ann_rtn / ann_vol
        nav = np.cumsum(rtn[ix:])
        mdd, mdd_bgn, mdd_end = 0, 0, 0
        for i in range(1, len(nav)):
            dd_i = np.full(i, nav[i]) - nav[:i]
            mdd_i = np.nanmin(dd_i)
            if mdd_i <= mdd:
                mdd = mdd_i
                mdd_bgn = np.argmin(dd_i)
                mdd_end = i
        wrt = np.nansum(rtn[ix:] > 0) / len(rtn[ix:])
        return {
            'period return': prd_rtn * 100,
            'annual return': ann_rtn * 100,
            'annual volatility': ann_vol * 100,
            'sharpe ratio': shrp,
            'max drawdown': mdd * 100,
            'max drawdown begin date': ds[ix:][mdd_bgn],
            'max drawdown end date': ds[ix:][mdd_end],
            'win rate': wrt * 100
        }

    @staticmethod
    def normalize(signal):
        """
        Normalize a signal by subtracting the mean and dividing by the standard deviation.

        Args:
            signal (pd.DataFrame): DataFrame containing the signal.

        Returns:
            pd.DataFrame: Normalized signal.
        """
        return (signal - np.nanmean(signal, axis=1).reshape(-1, 1)) / np.nanstd(signal, axis=1).reshape(-1, 1)

    @staticmethod
    def scale(signal):
        """
        Scale a signal by converting it to ranks and then scaling to the range [-0.5, 0.5].

        Args:
            signal (pd.DataFrame): DataFrame containing the signal.

        Returns:
            pd.DataFrame: Scaled signal.
        """
        signal = signal.rank(axis=1)
        min_ts = np.nanmin(signal, axis=1).reshape(-1, 1)
        max_ts = np.nanmax(signal, axis=1).reshape(-1, 1)
        return (signal - min_ts) / (max_ts - min_ts) - 0.5

    @staticmethod
    def winsorize(signal):
        """
        Winsorize a signal by replacing extreme values with values at the 3rd and 97th percentiles.

        Args:
            signal (pd.DataFrame): DataFrame containing the signal.

        Returns:
            pd.DataFrame: Winsorized signal.
        """

        def func(row):
            median = np.nanmedian(row)
            absolute_deviation = np.abs(row - median)
            MAD = np.nanmedian(absolute_deviation)
            MAD_e = MAD * 1.4826
            row[row >= median + 3 * MAD_e] = median + 3 * MAD_e
            row[row <= median - 3 * MAD_e] = median - 3 * MAD_e
            return row

        signal = signal.apply(func, axis=1)
        return signal

