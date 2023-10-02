import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class Strategy(object):

    def __init__(self, price: pd.DataFrame, volume: pd.DataFrame):
        self.price = price
        self.volume = volume
        self.daily_return = self.price / self.price.shift(1) - 1.

    def calc_vwap(self, window):
        """
        Calculate the Volume Weighted Average Price (VWAP).

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: VWAP values.
        """
        return (self.price * self.volume).rolling(window, min_periods=int(window / 4)).sum() / self.volume.rolling(
            window, min_periods=int(window / 4)).sum()

    def momentum(self, window):
        """
        Calculate the momentum signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Momentum values.
        """
        return self.price / self.price.shift(window) - 1.

    def relative_strength(self, long, short):
        """
        Calculate the relative strength signal.

        Args:
            long (int): Long-term rolling window size.
            short (int): Short-term rolling window size.

        Returns:
            pd.DataFrame: Relative strength values.
        """
        return self.price.shift(short) / self.price.shift(long) - 1.

    def price_position(self, window):
        """
        Calculate the price position signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Price position values.
        """
        H = self.price.rolling(window, min_periods=int(window / 4)).max()
        L = self.price.rolling(window, min_periods=int(window / 4)).min()
        pos = (self.price - L) / (H - L)
        pos[H - L <= 0.01] = np.nan
        return pos

    def twap_position(self, window):
        """
        Calculate the TWAP (Time-Weighted Average Price) position signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: TWAP position values.
        """
        H = self.price.rolling(window, min_periods=int(window / 4)).max()
        L = self.price.rolling(window, min_periods=int(window / 4)).min()
        twap = self.price.rolling(window, min_periods=int(window / 4)).mean()
        pos = (twap - L) / (H - L)
        pos[H - L <= 0.01] = np.nan
        return pos

    def volatility(self, window):
        """
        Calculate the volatility signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Volatility values.
        """
        return self.daily_return.rolling(window, min_periods=int(window / 4)).std()

    def skewness(self, window):
        """
        Calculate the skewness signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Skewness values.
        """
        return self.daily_return.rolling(window, min_periods=int(window / 4)).skew()

    def kurtosis(self, window):
        """
        Calculate the kurtosis signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Kurtosis values.
        """
        return self.daily_return.rolling(window, min_periods=int(window / 4)).kurt()

    def path_distance_ratio(self, window):
        """
        Calculate the path distance ratio signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Path distance ratio values.
        """
        daily_return_abs = abs(self.daily_return)
        path = daily_return_abs.rolling(window).sum()
        distance = self.price / self.price.shift(window) - 1.
        distance[distance <= 0.01] = np.nan
        return path / distance

    def price_volume_correlation(self, window):
        """
        Calculate the correlation between price and volume signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Price-volume correlation values.
        """

        def corr_by_column(x, y):
            mu_x = np.nanmean(x, axis=0)
            mu_y = np.nanmean(y, axis=0)
            sigma_x = np.nanstd(x, axis=0)
            sigma_y = np.nanstd(y, axis=0)
            return np.nanmean((x - mu_x) * (y - mu_y), axis=0) / (sigma_x * sigma_y)

        corr = np.zeros(shape=self.price.shape)
        corr[:] = np.nan
        for i in range(corr.shape[0]):
            cur_price = self.price.iloc[(i - window + 1):(i + 1), :]
            cur_volume = self.volume.iloc[(i - window + 1):(i + 1), :]
            corr[i, :] = corr_by_column(cur_price, cur_volume)
        return pd.DataFrame(corr, index=self.price.index, columns=self.price.columns)

    def spearman_rank_correlation(self, window):
        """
        Calculate the Spearman rank correlation between price and volume signal.

        Args:
            window (int): Rolling window size.

        Returns:
            pd.DataFrame: Spearman rank correlation values.
        """
        corr = self.price.rolling(window).corr(self.volume)
        return corr.groupby(level=0).apply(lambda x: spearmanr(x.droplevel(0))[0])
