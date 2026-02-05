"""
ブラックリスト管理

OOS検証でFAILした組み合わせ（銘柄 × レジーム × テンプレート）を記録し、
次回の最適化時にスキップするための機能。
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BLACKLIST_FILE = "results/fail_blacklist.json"


@dataclass
class BlacklistEntry:
    """ブラックリストエントリ"""
    symbol: str
    regime: str
    template: str
    reason: str  # "OOS FAIL", "Manual" など
    added_at: str  # ISO形式日付
    test_pnl: Optional[float] = None
    train_pnl: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "BlacklistEntry":
        return cls(**d)

    def key(self) -> Tuple[str, str, str]:
        """一意キー（symbol, regime, template）"""
        return (self.symbol, self.regime, self.template)


class Blacklist:
    """ブラックリスト管理クラス"""

    def __init__(self, filepath: str = BLACKLIST_FILE):
        self.filepath = Path(filepath)
        self.entries: List[BlacklistEntry] = []
        self._keys: Set[Tuple[str, str, str]] = set()
        self.load()

    def load(self) -> None:
        """ファイルから読み込み"""
        if not self.filepath.exists():
            self.entries = []
            self._keys = set()
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.entries = [
                BlacklistEntry.from_dict(e) for e in data.get("blacklist", [])
            ]
            self._keys = {e.key() for e in self.entries}
            logger.info(f"ブラックリスト読み込み: {len(self.entries)}件")

        except Exception as e:
            logger.warning(f"ブラックリスト読み込みエラー: {e}")
            self.entries = []
            self._keys = set()

    def save(self) -> None:
        """ファイルに保存"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "blacklist": [e.to_dict() for e in self.entries],
            "updated_at": datetime.now().isoformat(),
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"ブラックリスト保存: {len(self.entries)}件")

    def add(
        self,
        symbol: str,
        regime: str,
        template: str,
        reason: str = "OOS FAIL",
        test_pnl: Optional[float] = None,
        train_pnl: Optional[float] = None,
    ) -> bool:
        """エントリを追加（重複時はスキップ）"""
        key = (symbol, regime, template)
        if key in self._keys:
            return False  # 既存

        entry = BlacklistEntry(
            symbol=symbol,
            regime=regime,
            template=template,
            reason=reason,
            added_at=datetime.now().strftime("%Y-%m-%d"),
            test_pnl=test_pnl,
            train_pnl=train_pnl,
        )
        self.entries.append(entry)
        self._keys.add(key)
        return True

    def remove(self, symbol: str, regime: str, template: str) -> bool:
        """エントリを削除"""
        key = (symbol, regime, template)
        if key not in self._keys:
            return False

        self.entries = [e for e in self.entries if e.key() != key]
        self._keys.discard(key)
        return True

    def is_blacklisted(self, symbol: str, regime: str, template: str) -> bool:
        """ブラックリストに含まれているか"""
        return (symbol, regime, template) in self._keys

    def get_blacklisted_templates(self, symbol: str, regime: str) -> Set[str]:
        """指定銘柄・レジームでブラックリストされたテンプレート一覧"""
        return {
            e.template for e in self.entries
            if e.symbol == symbol and e.regime == regime
        }

    def filter_templates(
        self,
        templates: List[str],
        symbol: str,
        regime: str,
    ) -> List[str]:
        """ブラックリストを除外したテンプレートリストを返す"""
        blacklisted = self.get_blacklisted_templates(symbol, regime)
        filtered = [t for t in templates if t not in blacklisted]

        if len(filtered) < len(templates):
            skipped = len(templates) - len(filtered)
            logger.info(
                f"ブラックリストスキップ: {symbol}/{regime} - {skipped}テンプレート除外"
            )

        return filtered

    def clear(self) -> int:
        """全エントリ削除"""
        count = len(self.entries)
        self.entries = []
        self._keys = set()
        return count

    def get_entries_by_regime(self, regime: str) -> List[BlacklistEntry]:
        """レジーム別のエントリ一覧"""
        return [e for e in self.entries if e.regime == regime]

    def get_entries_by_symbol(self, symbol: str) -> List[BlacklistEntry]:
        """銘柄別のエントリ一覧"""
        return [e for e in self.entries if e.symbol == symbol]

    def summary(self) -> Dict[str, int]:
        """統計サマリー"""
        by_regime: Dict[str, int] = {}
        by_template: Dict[str, int] = {}

        for e in self.entries:
            by_regime[e.regime] = by_regime.get(e.regime, 0) + 1
            by_template[e.template] = by_template.get(e.template, 0) + 1

        return {
            "total": len(self.entries),
            "by_regime": by_regime,
            "by_template": by_template,
        }


# シングルトンインスタンス
_blacklist: Optional[Blacklist] = None


def get_blacklist() -> Blacklist:
    """グローバルブラックリストインスタンスを取得"""
    global _blacklist
    if _blacklist is None:
        _blacklist = Blacklist()
    return _blacklist


def add_fail_to_blacklist(
    symbol: str,
    regime: str,
    template: str,
    test_pnl: Optional[float] = None,
    train_pnl: Optional[float] = None,
) -> bool:
    """FAILをブラックリストに追加して保存"""
    bl = get_blacklist()
    added = bl.add(
        symbol=symbol,
        regime=regime,
        template=template,
        reason="OOS FAIL",
        test_pnl=test_pnl,
        train_pnl=train_pnl,
    )
    if added:
        bl.save()
    return added
