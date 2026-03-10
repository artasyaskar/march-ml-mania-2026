from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def competition_data(self) -> Path:
        return self.root / "Competition_data"

    def file(self, name: str) -> Path:
        return self.competition_data / name
