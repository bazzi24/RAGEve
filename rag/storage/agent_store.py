from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    system_prompt: str
    dataset_id: str
    embedding_model: str
    chat_model: str
    temperature: float = 0.7
    top_k: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    agent_id: str
    name: str
    description: str
    config: AgentConfig
    created_at: str
    updated_at: str


class AgentStore:
    """
    File-backed agent registry (no database dependency).
    Each agent maps to a dataset collection in Qdrant.
    """

    def __init__(self, registry_path: Path | str) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.registry_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def create(self, name: str, description: str, config: AgentConfig) -> Agent:
        agents = self._load()
        agent_id = str(uuid.uuid4())
        now = _iso_now()

        agent_dict: dict[str, Any] = {
            "agent_id": agent_id,
            "name": name,
            "description": description,
            "config": {
                "system_prompt": config.system_prompt,
                "dataset_id": config.dataset_id,
                "embedding_model": config.embedding_model,
                "chat_model": config.chat_model,
                "temperature": config.temperature,
                "top_k": config.top_k,
                "extra": config.extra,
            },
            "created_at": now,
            "updated_at": now,
        }

        agents[agent_id] = agent_dict
        self._save(agents)

        return self._dict_to_agent(agent_dict)

    def list(self) -> list[Agent]:
        agents = self._load()
        return [self._dict_to_agent(a) for a in agents.values()]

    def get(self, agent_id: str) -> Agent | None:
        agents = self._load()
        raw = agents.get(agent_id)
        if not raw:
            return None
        return self._dict_to_agent(raw)

    def update(self, agent_id: str, updates: dict[str, Any]) -> Agent | None:
        agents = self._load()
        if agent_id not in agents:
            return None

        raw = agents[agent_id]
        raw["updated_at"] = _iso_now()

        if "name" in updates:
            raw["name"] = updates["name"]
        if "description" in updates:
            raw["description"] = updates["description"]
        if "config" in updates:
            raw["config"].update(updates["config"])

        agents[agent_id] = raw
        self._save(agents)
        return self._dict_to_agent(raw)

    def delete(self, agent_id: str) -> bool:
        agents = self._load()
        if agent_id in agents:
            del agents[agent_id]
            self._save(agents)
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_agent(raw: dict[str, Any]) -> Agent:
        cfg = raw["config"]
        config = AgentConfig(
            system_prompt=cfg["system_prompt"],
            dataset_id=cfg["dataset_id"],
            embedding_model=cfg["embedding_model"],
            chat_model=cfg["chat_model"],
            temperature=cfg.get("temperature", 0.7),
            top_k=cfg.get("top_k", 5),
            extra=cfg.get("extra", {}),
        )
        return Agent(
            agent_id=raw["agent_id"],
            name=raw["name"],
            description=raw.get("description", ""),
            config=config,
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
        )


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
