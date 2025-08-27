from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Artifact:
    kind: str             # "csv", "txt", "json"
    path: str
    summary: Dict[str, Any] | None = None

@dataclass
class Artifacts:
    step: str
    outputs: List[Artifact]
    notes: Dict[str, Any] | None = None
