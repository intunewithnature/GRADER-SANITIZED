from __future__ import annotations

from .reveal_store import RevealStore


DEFAULT_FORMATS = {
    "person_name": "Person_{n}",
    "business_name": "Org_{n}",
    "email": "Email_{n}",
    "phone": "Phone_{n}",
    "url": "Url_{n}",
    "file_path": "Path_{n}",
    "account_id": "Account_{n}",
    "model_name": "Model_{n}",
}


def pseudonymize(
    store: RevealStore,
    artifact_id: str,
    scope_id: str,
    field_type: str,
    original_value: str,
    placeholder_format: str | None,
) -> str:
    fmt = placeholder_format or DEFAULT_FORMATS.get(field_type, f"{field_type.title()}_{{n}}")
    return store.get_or_create(artifact_id, scope_id, field_type, original_value, fmt).placeholder
