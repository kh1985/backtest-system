"""
出来高系インジケーター: VWAP, RelativeVolume, VolumeAnalysis
"""

from typing import List

import numpy as np
import pandas as pd

from .base import Indicator


class VWAP(Indicator):
    """
    Volume Weighted Average Price (日次リセット + 前日VWAP切替 + バンド)

    Parameters:
        switch_hour: 当日VWAPに切り替えるUTC時間（デフォルト1 = JST 10:00）
                     0-23の範囲。この時間以降は当日VWAP、以前は前日VWAPを使用。

    Output columns:
        vwap: 当日VWAP（日次リセット）
        vwap_prev: 前日の最終VWAP
        vwap_active: 時間帯に応じて切り替えた値（戦略で使用推奨）
        vwap_std: VWAP周りの標準偏差
        vwap_upper1, vwap_upper2: VWAP +1σ, +2σ
        vwap_lower1, vwap_lower2: VWAP -1σ, -2σ
    """

    def __init__(self, switch_hour: int = 1):
        self.name = "vwap"
        self.switch_hour = switch_hour  # UTC時間（1 = JST 10:00）

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # 日次でリセット
        if "datetime" in df.columns:
            date = df["datetime"].dt.date
            hour = df["datetime"].dt.hour
        else:
            # datetimeカラムがない場合は累積で計算（切替なし）
            date = pd.Series(0, index=df.index)
            hour = pd.Series(12, index=df.index)  # 常に切替後とみなす

        # groupby + cumsum でベクトル演算（forループ不要）
        tp_vol = typical_price * df["volume"]
        cum_tp_vol = tp_vol.groupby(date).cumsum()
        cum_vol = df["volume"].groupby(date).cumsum()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

        # VWAPバンド（標準偏差）の計算
        # σ = √(Σ((price - VWAP)² × volume) / Σvolume)
        # 累積で計算するため、まずVWAPを計算してから偏差を求める
        deviation_sq = (typical_price - df["vwap"]) ** 2
        deviation_sq_vol = deviation_sq * df["volume"]
        cum_deviation_sq_vol = deviation_sq_vol.groupby(date).cumsum()
        variance = cum_deviation_sq_vol / cum_vol.replace(0, np.nan)
        df["vwap_std"] = np.sqrt(variance)

        # バンドの計算
        df["vwap_upper1"] = df["vwap"] + df["vwap_std"]
        df["vwap_upper2"] = df["vwap"] + 2 * df["vwap_std"]
        df["vwap_lower1"] = df["vwap"] - df["vwap_std"]
        df["vwap_lower2"] = df["vwap"] - 2 * df["vwap_std"]

        # 前日VWAPを計算（各日の最終VWAP値を翌日に引き継ぐ）
        # 日ごとの最終VWAPを取得
        unique_dates = date.unique()
        date_to_prev_vwap = {}
        prev_vwap = np.nan
        for d in sorted(unique_dates):
            date_to_prev_vwap[d] = prev_vwap
            # この日の最終VWAPを次の日のprev_vwapに
            mask = date == d
            if mask.any():
                prev_vwap = df.loc[mask, "vwap"].iloc[-1]

        df["vwap_prev"] = date.map(date_to_prev_vwap)

        # 切替ロジック: UTC hour < switch_hour なら前日VWAP、以降は当日VWAP
        use_current = hour >= self.switch_hour
        df["vwap_active"] = np.where(
            use_current,
            df["vwap"],
            df["vwap_prev"]
        )

        # 前日バンドを計算（各日の最終バンド値を翌日に引き継ぐ）
        date_to_prev_upper1 = {}
        date_to_prev_upper2 = {}
        date_to_prev_lower1 = {}
        date_to_prev_lower2 = {}
        prev_upper1 = np.nan
        prev_upper2 = np.nan
        prev_lower1 = np.nan
        prev_lower2 = np.nan
        for d in sorted(unique_dates):
            date_to_prev_upper1[d] = prev_upper1
            date_to_prev_upper2[d] = prev_upper2
            date_to_prev_lower1[d] = prev_lower1
            date_to_prev_lower2[d] = prev_lower2
            mask = date == d
            if mask.any():
                prev_upper1 = df.loc[mask, "vwap_upper1"].iloc[-1]
                prev_upper2 = df.loc[mask, "vwap_upper2"].iloc[-1]
                prev_lower1 = df.loc[mask, "vwap_lower1"].iloc[-1]
                prev_lower2 = df.loc[mask, "vwap_lower2"].iloc[-1]

        df["vwap_upper1_prev"] = date.map(date_to_prev_upper1)
        df["vwap_upper2_prev"] = date.map(date_to_prev_upper2)
        df["vwap_lower1_prev"] = date.map(date_to_prev_lower1)
        df["vwap_lower2_prev"] = date.map(date_to_prev_lower2)

        # active版バンド（時間帯に応じて自動切替）
        df["vwap_upper1_active"] = np.where(use_current, df["vwap_upper1"], df["vwap_upper1_prev"])
        df["vwap_upper2_active"] = np.where(use_current, df["vwap_upper2"], df["vwap_upper2_prev"])
        df["vwap_lower1_active"] = np.where(use_current, df["vwap_lower1"], df["vwap_lower1_prev"])
        df["vwap_lower2_active"] = np.where(use_current, df["vwap_lower2"], df["vwap_lower2_prev"])

        return df

    @property
    def columns(self) -> List[str]:
        return [
            "vwap", "vwap_prev", "vwap_active",
            "vwap_std", "vwap_upper1", "vwap_upper2", "vwap_lower1", "vwap_lower2",
            "vwap_upper1_prev", "vwap_upper2_prev", "vwap_lower1_prev", "vwap_lower2_prev",
            "vwap_upper1_active", "vwap_upper2_active", "vwap_lower1_active", "vwap_lower2_active",
        ]

    @property
    def is_overlay(self) -> bool:
        return True


class VolumeSMA(Indicator):
    """出来高の単純移動平均（VolumeConditionで使用）"""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"volume_sma_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = df["volume"].rolling(window=self.period).mean()
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]


class RelativeVolume(Indicator):
    """相対出来高 (RVOL)"""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"rvol_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        avg_vol = df["volume"].rolling(window=self.period).mean()
        df[self.name] = df["volume"] / avg_vol.replace(0, float("nan"))
        return df

    @property
    def columns(self) -> List[str]:
        return [self.name]


class VolumeAnalysis(Indicator):
    """出来高分析（売買圧力分類）"""

    def __init__(self):
        self.name = "vol_analysis"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]

        candle_range = (df["high"] - df["low"]).replace(0, float("nan"))
        df["body_pct"] = (df["close"] - df["open"]).abs() / candle_range
        return df

    @property
    def columns(self) -> List[str]:
        return ["is_bullish", "is_bearish", "body_pct"]


class VolumeProfile(Indicator):
    """
    Volume Profile + LVN/HVN 検出 + 未テストLVN初タッチ

    Parameters:
        lookback: 計算に使うバー数（デフォルト: 全期間=0）
        n_bins: 価格帯の分割数（デフォルト: 50）
        smoothing: スムージング窓（デフォルト: 3）
        touch_tolerance: タッチ判定の許容幅（価格の%、デフォルト: 0.5%）

    Output columns:
        vp_poc: Point of Control（最大出来高価格）
        vp_lvn_1: 現在価格より下の最寄りLVN（押し目候補1）
        vp_lvn_2: 2番目に近いLVN（押し目候補2）
        vp_lvn_3: 3番目に近いLVN（押し目候補3）
        vp_hvn_1: 現在価格より下の最寄りHVN（サポート1）
        vp_hvn_2: 2番目に近いHVN（サポート2）
        vp_lvn_untested: 未テストで最寄りのLVN（ブレイク後まだタッチされていない）
        vp_lvn_first_touch: 未テストLVNに初タッチした瞬間（エントリーシグナル）
    """

    def __init__(
        self,
        lookback: int = 0,
        n_bins: int = 50,
        smoothing: int = 3,
        touch_tolerance: float = 0.5,
        break_margin: float = 2.0,
        min_bars_after_break: int = 10,
    ):
        self.lookback = lookback
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.touch_tolerance = touch_tolerance / 100.0  # %を小数に
        self.break_margin = break_margin / 100.0  # %を小数に
        self.min_bars_after_break = min_bars_after_break
        self.name = f"vp_{n_bins}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # 出力カラム初期化
        df["vp_poc"] = np.nan
        df["vp_lvn_1"] = np.nan
        df["vp_lvn_2"] = np.nan
        df["vp_lvn_3"] = np.nan
        df["vp_hvn_1"] = np.nan
        df["vp_hvn_2"] = np.nan
        df["vp_lvn_untested"] = np.nan
        df["vp_lvn_first_touch"] = False

        # 最低限のデータが必要
        min_bars = max(self.n_bins, 100)
        if n < min_bars:
            return df

        # ローリング計算用のインデックス範囲
        lookback = self.lookback if self.lookback > 0 else n

        # 各バーでVolume Profileを計算（効率化: 一定間隔で計算）
        calc_interval = 10
        calc_indices = np.arange(min_bars, n, calc_interval)

        poc_arr = np.full(n, np.nan)
        lvn1_arr = np.full(n, np.nan)
        lvn2_arr = np.full(n, np.nan)
        lvn3_arr = np.full(n, np.nan)
        hvn1_arr = np.full(n, np.nan)
        hvn2_arr = np.full(n, np.nan)

        # 全LVNを収集（未テスト判定用）
        all_lvn_prices = []

        for idx in calc_indices:
            start = max(0, idx - lookback)
            h = high[start:idx]
            l = low[start:idx]
            v = volume[start:idx]
            c = close[idx - 1]

            poc, lvns, hvns = self._compute_profile(h, l, v, c)

            poc_arr[idx] = poc
            if len(lvns) >= 1:
                lvn1_arr[idx] = lvns[0]
                all_lvn_prices.extend(lvns)
            if len(lvns) >= 2:
                lvn2_arr[idx] = lvns[1]
            if len(lvns) >= 3:
                lvn3_arr[idx] = lvns[2]
            if len(hvns) >= 1:
                hvn1_arr[idx] = hvns[0]
            if len(hvns) >= 2:
                hvn2_arr[idx] = hvns[1]

        # 前方向にfill
        df["vp_poc"] = pd.Series(poc_arr, index=df.index).ffill()
        df["vp_lvn_1"] = pd.Series(lvn1_arr, index=df.index).ffill()
        df["vp_lvn_2"] = pd.Series(lvn2_arr, index=df.index).ffill()
        df["vp_lvn_3"] = pd.Series(lvn3_arr, index=df.index).ffill()
        df["vp_hvn_1"] = pd.Series(hvn1_arr, index=df.index).ffill()
        df["vp_hvn_2"] = pd.Series(hvn2_arr, index=df.index).ffill()

        # === 未テストLVN + 初タッチ検出 ===
        if all_lvn_prices:
            self._detect_first_touch(df, high, low, close, all_lvn_prices)

        return df

    def _detect_first_touch(
        self,
        df: pd.DataFrame,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        all_lvn_prices: list,
    ) -> None:
        """
        未テストLVNの初タッチを検出（ベクトル化版）

        ロジック:
        1. 価格が上昇してLVNを「明確に突破」（LVN + マージン以上に上昇）
        2. その後、価格がLVNまで「押し戻してきた」
        3. 初めて押し戻してきた瞬間 = first_touch（エントリーシグナル）
        """
        n = len(df)
        break_margin = self.break_margin
        min_bars_after_break = self.min_bars_after_break

        # ユニークなLVN価格
        unique_lvns = np.unique(all_lvn_prices)
        m = len(unique_lvns)

        if m == 0:
            df["vp_lvn_untested"] = np.nan
            df["vp_lvn_first_touch"] = False
            return

        # 累積最高値
        cummax_high = np.maximum.accumulate(high)

        # 閾値計算 (shape: m)
        break_thresholds = unique_lvns * (1 + break_margin)
        touch_thresholds = unique_lvns * (1 + self.touch_tolerance)

        # ブロードキャスト用に reshape
        # cummax_high: (n,) -> (n, 1), break_thresholds: (m,) -> (1, m)
        # 結果: (n, m) の bool 配列

        # 突破判定: cummax_high >= break_threshold
        broken_mask = cummax_high[:, None] >= break_thresholds[None, :]  # (n, m)

        # 各LVNの最初の突破バーを検出
        # argmax は最初の True のインデックスを返す（全部 False なら 0）
        has_broken = broken_mask.any(axis=0)  # (m,) どのLVNが突破されたか
        break_indices = np.argmax(broken_mask, axis=0)  # (m,) 最初の突破バー

        # タッチ判定: low <= lvn + tolerance
        touch_mask = low[:, None] <= touch_thresholds[None, :]  # (n, m)

        # 有効なタッチ = 突破後 + min_bars_after_break 以降
        bar_indices = np.arange(n)[:, None]  # (n, 1)
        valid_after_break = bar_indices >= (break_indices[None, :] + min_bars_after_break)  # (n, m)

        # 最終的なタッチ条件
        valid_touch = touch_mask & valid_after_break & has_broken[None, :]  # (n, m)

        # 各LVNの最初のタッチバーを検出
        has_touch = valid_touch.any(axis=0)  # (m,)
        touch_indices = np.where(has_touch, np.argmax(valid_touch, axis=0), -1)  # (m,)

        # first_touch シグナル: 各バーで何かのLVNに初タッチしたか
        first_touch_arr = np.zeros(n, dtype=bool)
        valid_touches = touch_indices[touch_indices >= 0]
        if len(valid_touches) > 0:
            np.add.at(first_touch_arr, valid_touches, True)

        # === 未テストLVN計算（完全ベクトル化）===
        # 各バーで「broken だが まだ tested でない」LVNの最大値を見つける

        broken_end = np.where(has_touch, touch_indices, n)

        # 2D マスク: (n, m) - バー i で LVN j が未テストか
        # 条件: break_indices[j] <= i < broken_end[j] かつ lvn[j] < close[i]
        is_broken_at_bar = (bar_indices >= break_indices[None, :]) & (bar_indices < broken_end[None, :])
        is_below_price = unique_lvns[None, :] < close[:, None]  # (n, m)
        untested_mask = is_broken_at_bar & is_below_price  # (n, m)

        # 各バーで未テストLVNの最大値を計算
        # マスクされた値で max を取る（未テストがなければ NaN）
        lvn_values = np.where(untested_mask, unique_lvns[None, :], -np.inf)  # (n, m)
        max_untested = np.max(lvn_values, axis=1)  # (n,)
        untested_arr = np.where(max_untested > -np.inf, max_untested, np.nan)

        df["vp_lvn_untested"] = untested_arr
        df["vp_lvn_first_touch"] = first_touch_arr

    def _compute_profile(
        self, high: np.ndarray, low: np.ndarray, volume: np.ndarray, current_price: float
    ) -> tuple:
        """
        Volume Profileを計算し、POC/LVN/HVNを返す（ベクトル演算）
        """
        price_min = low.min()
        price_max = high.max()

        if price_max <= price_min:
            return np.nan, [], []

        # 価格ビン作成
        bin_edges = np.linspace(price_min, price_max, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 各バーの出来高を価格帯に分配（ベクトル化）
        # 簡易版: 各バーのTypical Priceでビン割当
        typical = (high + low) / 2
        bin_indices = np.digitize(typical, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # ビンごとの出来高集計
        profile = np.bincount(bin_indices, weights=volume, minlength=self.n_bins)

        # スムージング（移動平均）
        if self.smoothing > 1:
            kernel = np.ones(self.smoothing) / self.smoothing
            profile = np.convolve(profile, kernel, mode="same")

        # POC（最大出来高の価格）
        poc_idx = np.argmax(profile)
        poc = bin_centers[poc_idx]

        # LVN検出（局所的な谷）
        # 左右より小さい点を検出
        left = np.roll(profile, 1)
        right = np.roll(profile, -1)
        left[0] = np.inf  # 端は無視
        right[-1] = np.inf

        is_local_min = (profile < left) & (profile < right)
        lvn_indices = np.where(is_local_min)[0]
        lvn_prices = bin_centers[lvn_indices]

        # HVN検出（局所的な山）
        is_local_max = (profile > left) & (profile > right)
        hvn_indices = np.where(is_local_max)[0]
        hvn_prices = bin_centers[hvn_indices]

        # 現在価格より下のLVN/HVNを近い順にソート
        lvn_below = lvn_prices[lvn_prices < current_price]
        hvn_below = hvn_prices[hvn_prices < current_price]

        # 近い順にソート（降順 = 現在価格に近い順）
        lvn_below = np.sort(lvn_below)[::-1]
        hvn_below = np.sort(hvn_below)[::-1]

        return poc, lvn_below.tolist(), hvn_below.tolist()

    @property
    def columns(self) -> List[str]:
        return [
            "vp_poc", "vp_lvn_1", "vp_lvn_2", "vp_lvn_3",
            "vp_hvn_1", "vp_hvn_2", "vp_lvn_untested", "vp_lvn_first_touch"
        ]

    @property
    def is_overlay(self) -> bool:
        return True
