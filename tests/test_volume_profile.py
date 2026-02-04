"""VolumeProfile + LVN検出のテスト"""

import numpy as np
import pandas as pd
import pytest

from indicators.volume import VolumeProfile


class TestVolumeProfile:
    """VolumeProfileインジケーターのテスト"""

    @pytest.fixture
    def sample_uptrend_data(self) -> pd.DataFrame:
        """上昇トレンドのサンプルデータ（出来高に特徴あり）"""
        np.random.seed(42)
        n = 500

        # 上昇トレンド: 0.04 → 0.08
        base_price = np.linspace(0.04, 0.08, n)
        noise = np.random.randn(n) * 0.001

        close = base_price + noise
        high = close + np.abs(np.random.randn(n) * 0.0005)
        low = close - np.abs(np.random.randn(n) * 0.0005)
        open_ = close - np.random.randn(n) * 0.0003

        # 出来高: 特定の価格帯で高くする（HVN作成）
        volume = np.ones(n) * 1000
        # 0.05付近で出来高増加
        mask_05 = (close > 0.048) & (close < 0.052)
        volume[mask_05] = 5000
        # 0.065付近で出来高増加
        mask_065 = (close > 0.063) & (close < 0.067)
        volume[mask_065] = 4000

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_basic_calculation(self, sample_uptrend_data):
        """基本計算が動作することを確認"""
        df = sample_uptrend_data
        vp = VolumeProfile(n_bins=50, smoothing=3)
        result = vp.calculate(df.copy())

        # 出力カラムが存在
        assert "vp_poc" in result.columns
        assert "vp_lvn_1" in result.columns
        assert "vp_lvn_2" in result.columns
        assert "vp_hvn_1" in result.columns

        # 最後の行でNaNでない値がある
        assert not np.isnan(result["vp_poc"].iloc[-1])

    def test_poc_is_high_volume_area(self, sample_uptrend_data):
        """POCが高出来高エリアにあることを確認"""
        df = sample_uptrend_data
        vp = VolumeProfile(n_bins=50, smoothing=3)
        result = vp.calculate(df.copy())

        poc = result["vp_poc"].iloc[-1]
        # POCは0.05か0.065付近にあるはず
        assert (0.045 < poc < 0.055) or (0.060 < poc < 0.070)

    def test_lvn_below_current_price(self, sample_uptrend_data):
        """LVNが現在価格より下にあることを確認"""
        df = sample_uptrend_data
        vp = VolumeProfile(n_bins=50, smoothing=3)
        result = vp.calculate(df.copy())

        current_price = df["close"].iloc[-1]
        lvn1 = result["vp_lvn_1"].iloc[-1]

        if not np.isnan(lvn1):
            assert lvn1 < current_price

    def test_lvn_ordering(self, sample_uptrend_data):
        """LVNが現在価格に近い順に並んでいることを確認"""
        df = sample_uptrend_data
        vp = VolumeProfile(n_bins=50, smoothing=3)
        result = vp.calculate(df.copy())

        lvn1 = result["vp_lvn_1"].iloc[-1]
        lvn2 = result["vp_lvn_2"].iloc[-1]

        if not np.isnan(lvn1) and not np.isnan(lvn2):
            # lvn1の方が現在価格に近い（大きい）
            assert lvn1 >= lvn2

    def test_no_for_loop_performance(self, sample_uptrend_data):
        """大量データでも高速に処理できることを確認"""
        import time

        # 10000バーのデータ
        df = pd.concat([sample_uptrend_data] * 20, ignore_index=True)

        vp = VolumeProfile(n_bins=50, smoothing=3)

        start = time.time()
        vp.calculate(df.copy())
        elapsed = time.time() - start

        # 10000バーで1秒以内
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"

    def test_empty_dataframe(self):
        """空のDataFrameでエラーにならない"""
        df = pd.DataFrame({
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })
        vp = VolumeProfile()
        result = vp.calculate(df)
        assert len(result) == 0

    def test_small_dataframe(self):
        """少量データでNaNが返る"""
        df = pd.DataFrame({
            "open": [1.0, 1.1],
            "high": [1.05, 1.15],
            "low": [0.95, 1.05],
            "close": [1.02, 1.12],
            "volume": [100, 100],
        })
        vp = VolumeProfile()
        result = vp.calculate(df)
        # 少量なのでNaN
        assert np.isnan(result["vp_poc"].iloc[-1])

    def test_columns_property(self):
        """columnsプロパティが正しい"""
        vp = VolumeProfile()
        assert "vp_poc" in vp.columns
        assert "vp_lvn_1" in vp.columns
        assert "vp_hvn_1" in vp.columns

    def test_is_overlay(self):
        """オーバーレイインジケーター"""
        vp = VolumeProfile()
        assert vp.is_overlay is True


class TestVolumeProfileCompute:
    """_compute_profile メソッドの単体テスト"""

    def test_compute_profile_basic(self):
        """基本的なprofile計算"""
        vp = VolumeProfile(n_bins=10, smoothing=1)

        high = np.array([1.1, 1.2, 1.15, 1.25, 1.3])
        low = np.array([0.9, 1.0, 0.95, 1.05, 1.1])
        volume = np.array([100, 500, 100, 100, 100])  # 1.1付近で高出来高
        current_price = 1.25

        poc, lvns, hvns = vp._compute_profile(high, low, volume, current_price)

        # POCは高出来高エリア付近
        assert 1.0 < poc < 1.2

    def test_compute_profile_detects_lvn(self):
        """LVN（出来高の谷）が検出される"""
        vp = VolumeProfile(n_bins=20, smoothing=1)

        # 出来高パターン: 高-低-高（LVNを作る）
        n = 100
        high = np.linspace(1.0, 2.0, n) + 0.05
        low = np.linspace(1.0, 2.0, n) - 0.05
        volume = np.ones(n) * 100

        # 1.3付近と1.7付近で高出来高、間の1.5付近が低出来高
        mask_13 = (high > 1.25) & (high < 1.35)
        mask_17 = (high > 1.65) & (high < 1.75)
        volume[mask_13] = 500
        volume[mask_17] = 500

        current_price = 1.8

        poc, lvns, hvns = vp._compute_profile(high, low, volume, current_price)

        # LVNが検出される
        assert len(lvns) > 0
