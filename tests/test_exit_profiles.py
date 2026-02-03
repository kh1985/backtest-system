"""
Exit profiles と templates の exit_profiles 対応テスト
"""

import copy
import pytest

from optimizer.exit_profiles import (
    ALL_PROFILES,
    FIXED_PROFILES,
    ATR_PROFILES,
    NO_SL_PROFILES,
    HYBRID_PROFILES,
    get_profiles,
    profile_to_exit_config,
)
from optimizer.templates import BUILTIN_TEMPLATES, StrategyTemplate, ParameterRange


# ---------------------------------------------------------------------------
# Exit Profiles 定義
# ---------------------------------------------------------------------------

class TestExitProfiles:
    """exit_profiles モジュールのテスト"""

    def test_all_profiles_have_name(self):
        """全プロファイルが name を持つ"""
        for p in ALL_PROFILES:
            assert "name" in p, f"Profile missing name: {p}"
            assert len(p["name"]) > 0

    def test_all_profiles_unique_names(self):
        """プロファイル名がユニーク"""
        names = [p["name"] for p in ALL_PROFILES]
        assert len(names) == len(set(names))

    def test_fixed_profiles_have_pct(self):
        """固定%プロファイルが TP/SL pct を持つ"""
        for p in FIXED_PROFILES:
            assert "take_profit_pct" in p
            assert "stop_loss_pct" in p

    def test_atr_profiles_have_mult(self):
        """ATRプロファイルが ATR パラメータを持つ"""
        for p in ATR_PROFILES:
            assert p.get("use_atr_exit") is True
            assert "atr_tp_mult" in p
            assert "atr_sl_mult" in p

    def test_no_sl_profiles(self):
        """SLなしプロファイルの SL が 0"""
        for p in NO_SL_PROFILES:
            assert p.get("stop_loss_pct", 0) == 0
            assert p.get("trailing_stop_pct", 0) > 0

    def test_get_profiles_all(self):
        """get_profiles('all') == ALL_PROFILES"""
        assert get_profiles("all") == ALL_PROFILES

    def test_get_profiles_fixed(self):
        assert get_profiles("fixed") == FIXED_PROFILES

    def test_get_profiles_atr(self):
        assert get_profiles("atr") == ATR_PROFILES

    def test_get_profiles_unknown_returns_all(self):
        """不明なモード → ALL"""
        assert get_profiles("unknown") == ALL_PROFILES

    def test_profile_to_exit_config(self):
        """name を除いた exit config を生成"""
        p = {"name": "test", "take_profit_pct": 2.0, "stop_loss_pct": 1.0}
        conf = profile_to_exit_config(p)
        assert "name" not in conf
        assert conf["take_profit_pct"] == 2.0
        assert conf["stop_loss_pct"] == 1.0


# ---------------------------------------------------------------------------
# Templates × Exit Profiles 直積
# ---------------------------------------------------------------------------

class TestTemplateExitProfiles:
    """templates.generate_configs() の exit_profiles 対応テスト"""

    def _get_simple_template(self) -> StrategyTemplate:
        """テスト用の簡易テンプレート"""
        return StrategyTemplate(
            name="test_template",
            description="test",
            config_template={
                "name": "test",
                "side": "long",
                "indicators": [],
                "entry_conditions": [],
                "entry_logic": "and",
                "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
            },
            param_ranges=[
                ParameterRange("param_a", 1, 3, 1, "int"),
            ],
        )

    def test_no_exit_profiles_backward_compat(self):
        """exit_profiles=None → 従来動作"""
        t = self._get_simple_template()
        configs = t.generate_configs(exit_profiles=None)
        # param_a: 1, 2, 3 → 3通り
        assert len(configs) == 3
        # テンプレート内蔵の exit が使われる
        assert configs[0]["exit"]["take_profit_pct"] == 2.0

    def test_exit_profiles_cross_product(self):
        """exit_profiles 指定 → 直積"""
        t = self._get_simple_template()
        profiles = [
            {"name": "ep1", "take_profit_pct": 3.0, "stop_loss_pct": 1.5},
            {"name": "ep2", "take_profit_pct": 1.5, "stop_loss_pct": 0.5},
        ]
        configs = t.generate_configs(exit_profiles=profiles)
        # 3 entry × 2 exit = 6
        assert len(configs) == 6

    def test_exit_profiles_override_exit(self):
        """exit_profiles が exit セクションを上書き"""
        t = self._get_simple_template()
        profiles = [
            {"name": "custom", "take_profit_pct": 5.0, "stop_loss_pct": 3.0},
        ]
        configs = t.generate_configs(exit_profiles=profiles)
        for c in configs:
            assert c["exit"]["take_profit_pct"] == 5.0
            assert c["exit"]["stop_loss_pct"] == 3.0

    def test_exit_profiles_metadata(self):
        """_exit_profile メタデータが付加される"""
        t = self._get_simple_template()
        profiles = [{"name": "ep1", "take_profit_pct": 3.0, "stop_loss_pct": 1.5}]
        configs = t.generate_configs(exit_profiles=profiles)
        for c in configs:
            assert c["_exit_profile"] == "ep1"

    def test_exit_profiles_atr_mode(self):
        """ATR exit profile が正しく config に反映"""
        t = self._get_simple_template()
        profiles = [
            {
                "name": "atr_test",
                "use_atr_exit": True,
                "atr_tp_mult": 3.0,
                "atr_sl_mult": 1.5,
                "atr_period": 14,
            },
        ]
        configs = t.generate_configs(exit_profiles=profiles)
        for c in configs:
            assert c["exit"]["use_atr_exit"] is True
            assert c["exit"]["atr_tp_mult"] == 3.0
            assert c["exit"]["atr_sl_mult"] == 1.5

    def test_exit_profiles_name_includes_profile(self):
        """config名に exit profile 名が含まれる"""
        t = self._get_simple_template()
        profiles = [{"name": "ep1", "take_profit_pct": 3.0, "stop_loss_pct": 1.5}]
        configs = t.generate_configs(exit_profiles=profiles)
        for c in configs:
            assert "ep1" in c["name"]

    def test_builtin_templates_with_profiles(self):
        """組み込みテンプレートが exit_profiles に対応"""
        profiles = FIXED_PROFILES[:1]  # 1プロファイルのみ
        for tname, template in list(BUILTIN_TEMPLATES.items())[:2]:
            configs_without = template.generate_configs()
            configs_with = template.generate_configs(exit_profiles=profiles)
            # exit_profiles指定時は同数（1プロファイルなので）
            assert len(configs_with) == len(configs_without)

    def test_no_param_ranges_with_exit_profiles(self):
        """param_ranges なしでも exit_profiles との直積が動作"""
        t = StrategyTemplate(
            name="simple",
            description="test",
            config_template={
                "name": "simple",
                "side": "long",
                "indicators": [],
                "entry_conditions": [],
                "entry_logic": "and",
                "exit": {"take_profit_pct": 2.0, "stop_loss_pct": 1.0},
            },
        )
        profiles = [
            {"name": "ep1", "take_profit_pct": 3.0, "stop_loss_pct": 1.5},
            {"name": "ep2", "take_profit_pct": 1.5, "stop_loss_pct": 0.5},
        ]
        configs = t.generate_configs(exit_profiles=profiles)
        # 1 entry × 2 exit = 2
        assert len(configs) == 2
