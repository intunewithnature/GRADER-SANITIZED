from __future__ import annotations

import json
from pathlib import Path

from .models import FieldPolicy, Policy


def load_policy(path: str | Path) -> Policy:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {name: FieldPolicy(**value) for name, value in data.get("fields", {}).items()}
    return Policy(name=data["name"], scope=data["scope"], fields=fields)


def policy_to_dict(policy: Policy) -> dict:
    return {
        "name": policy.name,
        "scope": policy.scope,
        "fields": {
            name: {
                key: value
                for key, value in vars(field_policy).items()
                if value is not None
            }
            for name, field_policy in policy.fields.items()
        },
    }
