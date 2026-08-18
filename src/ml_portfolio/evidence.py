"""Small, deterministic contracts for committed execution evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import hashlib
import html
import json
from pathlib import Path
import re


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EMBEDDED_IMAGE_PATTERN = re.compile(
    r'<img\b(?P<attrs>[^>]*?)\s+src="data:image/'
    r'(?P<extension>png|jpeg|jpg|webp);base64,(?P<payload>[^"]+)"'
    r'(?P<tail>[^>]*)>',
    flags=re.IGNORECASE | re.DOTALL,
)
IMAGE_DATA_URI_URL_PATTERN = re.compile(
    r"url\(\s*data:image/(?:svg\+xml|png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=]+\s*\)",
    flags=re.IGNORECASE,
)
BANK_EVIDENCE_BOUNDARY = (
    "This is a historical execution snapshot for review, not a live report, "
    "deployed model, or policy recommendation."
)
BANK_FIGURE_ALT_TEXTS = (
    "Late-period precision-recall curve showing weak ranking under source-order validation.",
    "Late-period lift by contact budget showing exploratory top-1 concentration fading at wider budgets.",
    "Cumulative gains curve comparing source-order ranking with random ranking.",
    "Late-period calibration diagnostic showing predicted scores against observed response rates.",
    "Late-period permutation-importance chart showing small descriptive feature effects.",
)


def externalize_embedded_images(
    source_html: str,
    asset_dir: Path,
) -> tuple[str, list[dict[str, str | int]]]:
    """Move notebook image outputs into hashed, accessible local assets."""

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, str | int]] = []

    def replace_image(match: re.Match[str]) -> str:
        extension = match.group("extension").lower().replace("jpeg", "jpg")
        payload = base64.b64decode(match.group("payload"), validate=True)
        number = len(assets) + 1
        filename = f"figure-{number:02d}.{extension}"
        (asset_dir / filename).write_bytes(payload)
        relative_path = f"assets/{filename}"
        alt_text = (
            BANK_FIGURE_ALT_TEXTS[number - 1]
            if number <= len(BANK_FIGURE_ALT_TEXTS)
            else f"Bank Marketing execution figure {number:02d} from the executed notebook."
        )

        attrs = re.sub(
            r"\s+alt\s*=\s*(?:\"[^\"]*\"|'[^']*')",
            "",
            match.group("attrs"),
            count=1,
            flags=re.IGNORECASE,
        )
        tail = re.sub(
            r"\s+alt\s*=\s*(?:\"[^\"]*\"|'[^']*')",
            "",
            match.group("tail"),
            count=1,
            flags=re.IGNORECASE,
        )
        assets.append(
            {
                "path": relative_path,
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "alt": alt_text,
            }
        )
        return (
            f'<img{attrs} alt="{html.escape(alt_text)}" '
            f'src="{relative_path}"{tail}>'
        )

    sanitized = EMBEDDED_IMAGE_PATTERN.sub(replace_image, source_html)
    sanitized = IMAGE_DATA_URI_URL_PATTERN.sub("none", sanitized)
    return sanitized, assets


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""

    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON predictably for byte-for-byte comparisons."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_digest(value: str, *, label: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")


def build_bank_evidence_manifest(
    *,
    source_commit: str,
    source_tree: str,
    notebook_path: str,
    notebook_sha256: str,
    input_url: str,
    input_sha256: str,
    artifact_path: str,
    artifact_sha256: str,
    artifact_bytes: int,
    environment: Mapping[str, str],
    artifact_assets: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build the versioned Bank execution-evidence manifest."""

    _require_digest(source_commit, label="source commit", pattern=COMMIT_PATTERN)
    _require_digest(source_tree, label="source tree", pattern=COMMIT_PATTERN)
    for label, digest in (
        ("notebook SHA-256", notebook_sha256),
        ("input SHA-256", input_sha256),
        ("artifact SHA-256", artifact_sha256),
    ):
        _require_digest(digest, label=label, pattern=SHA256_PATTERN)
    if artifact_bytes < 1:
        raise ValueError("artifact bytes must be positive")

    artifact_record: dict[str, object] = {
        "path": artifact_path,
        "sha256": artifact_sha256,
        "bytes": artifact_bytes,
    }
    if artifact_assets is not None:
        artifact_record["assets"] = [dict(asset) for asset in artifact_assets]

    return {
        "schema_version": 1,
        "source": {
            "commit": source_commit,
            "tree": source_tree,
            "notebook": {
                "path": notebook_path,
                "sha256": notebook_sha256,
            },
        },
        "input": {
            "name": "UCI Bank Marketing archive",
            "url": input_url,
            "sha256": input_sha256,
            "verification": "downloaded and SHA-256 checked by the notebook loader",
        },
        "environment": dict(sorted(environment.items())),
        "artifact": artifact_record,
        "regeneration_command": "python scripts/build_bank_evidence.py",
        "boundary": BANK_EVIDENCE_BOUNDARY,
    }


def verify_bank_evidence_manifest(manifest: Mapping[str, object], *, root: Path) -> None:
    """Raise if committed Bank evidence no longer matches its manifest."""

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Bank evidence schema version")

    source = manifest.get("source")
    artifact = manifest.get("artifact")
    if not isinstance(source, Mapping) or not isinstance(artifact, Mapping):
        raise ValueError("Bank evidence manifest is missing source or artifact metadata")

    notebook = source.get("notebook")
    if not isinstance(notebook, Mapping):
        raise ValueError("Bank evidence manifest is missing notebook metadata")

    notebook_path = root / str(notebook.get("path", ""))
    artifact_path = root / str(artifact.get("path", ""))
    if sha256_file(notebook_path) != notebook.get("sha256"):
        raise ValueError("source notebook SHA-256 does not match the manifest")
    if sha256_file(artifact_path) != artifact.get("sha256"):
        raise ValueError("artifact SHA-256 does not match the manifest")
    if artifact_path.stat().st_size != artifact.get("bytes"):
        raise ValueError("artifact byte count does not match the manifest")

    assets = artifact.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Bank evidence assets must be a list")
    root_resolved = root.resolve()
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise ValueError("Bank evidence asset records must be mappings")
        asset_path = (root / str(asset.get("path", ""))).resolve()
        if not asset_path.is_relative_to(root_resolved):
            raise ValueError("Bank evidence asset path escapes repository root")
        if not asset_path.is_file():
            raise ValueError(f"Bank evidence asset is missing: {asset_path}")
        if sha256_file(asset_path) != asset.get("sha256"):
            raise ValueError("Bank evidence asset SHA-256 does not match its manifest")
        if asset_path.stat().st_size != asset.get("bytes"):
            raise ValueError("Bank evidence asset byte count does not match its manifest")
        alt = str(asset.get("alt", "")).strip()
        if not alt or alt.lower() == "no description has been provided for this image":
            raise ValueError("Bank evidence asset must have meaningful alt text")
