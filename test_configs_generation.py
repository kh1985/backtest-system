#!/usr/bin/env python3
"""4戦略のconfigs生成数確認"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimizer.templates import BUILTIN_TEMPLATES

def test_configs_generation():
    """4戦略のconfigs生成数を確認"""
    
    filter_patterns = ["supertrend_short", "tsmom_short", "rsi_connors_short", "donchian_breakdown_short"]
    
    print("=== Configs生成テスト ===\n")
    
    total_configs = 0
    for tname, template in BUILTIN_TEMPLATES.items():
        if tname in filter_patterns:
            configs = template.generate_configs(exit_profiles=None)
            print(f"{tname}: {len(configs)} configs")
            total_configs += len(configs)
            
            # パラメータ範囲を表示
            print(f"  パラメータ範囲:")
            for param_range in template.param_ranges:
                values = param_range.values()
                print(f"    {param_range.name}: {values} (count={len(values)})")
            print()
    
    print(f"合計: {total_configs} configs")
    
    # 見つからなかった戦略
    found_templates = [tname for tname in BUILTIN_TEMPLATES.keys() if tname in filter_patterns]
    missing = set(filter_patterns) - set(found_templates)
    
    if missing:
        print(f"\n⚠️ 見つからなかった戦略: {missing}")
        print(f"\n利用可能な戦略（部分一致）:")
        for tname in BUILTIN_TEMPLATES.keys():
            if any(pattern in tname for pattern in filter_patterns):
                print(f"  - {tname}")
    else:
        print(f"\n✅ 全ての戦略が見つかりました")

if __name__ == "__main__":
    test_configs_generation()
