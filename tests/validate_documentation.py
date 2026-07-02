from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "README_ES.md",
    "docs/architecture.md",
    "docs/methodology.md",
    "docs/evaluation.md",
    "docs/demo_guide.md",
    "docs/presentation_guide_es.md",
    "docs/local_interface.md",
    "docs/environment_setup.md",
    "docs/local_ollama_setup.md",
    "reports/formal_retriever_comparison.md",
    "reports/formal_rag_generation_evaluation.md",
    "reports/formal_rag_generation_manual_review.md",
)

REQUIRED_METRICS = (
    "30 / 30",
    "10 / 10",
    "100%",
    "0.8617",
    "0.9544",
)

PLACEHOLDERS = (
    "TODO",
    "TBD",
    "FIXME",
    "<insert",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_required_files() -> None:
    for relative in REQUIRED_FILES:
        require(
            (PROJECT_ROOT / relative).exists(),
            f"Missing documentation file: {relative}",
        )


def validate_metrics() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    evaluation = (PROJECT_ROOT / "docs/evaluation.md").read_text(
        encoding="utf-8"
    )
    combined = readme + "\n" + evaluation

    for metric in REQUIRED_METRICS:
        require(
            metric in combined,
            f"Missing final metric in documentation: {metric}",
        )


def validate_relative_links() -> None:
    markdown_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README_ES.md",
        *sorted((PROJECT_ROOT / "docs").glob("*.md")),
    ]

    link_pattern = re.compile(
        r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)"
    )

    for markdown_path in markdown_files:
        text = markdown_path.read_text(encoding="utf-8")

        for match in link_pattern.finditer(text):
            target = match.group(1).split("#", 1)[0].strip()

            if not target:
                continue

            resolved = (markdown_path.parent / target).resolve()

            require(
                resolved.exists(),
                f"Broken relative link in "
                f"{markdown_path.relative_to(PROJECT_ROOT)}: "
                f"{target}",
            )


def validate_no_placeholders() -> None:
    files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README_ES.md",
        *sorted((PROJECT_ROOT / "docs").glob("*.md")),
    ]

    for path in files:
        text = path.read_text(encoding="utf-8")

        for placeholder in PLACEHOLDERS:
            require(
                placeholder not in text,
                f"Placeholder {placeholder!r} found in "
                f"{path.relative_to(PROJECT_ROOT)}",
            )


def main() -> None:
    validate_required_files()
    validate_metrics()
    validate_relative_links()
    validate_no_placeholders()

    print("=" * 72)
    print("FINAL DOCUMENTATION VALIDATION PASSED")
    print("=" * 72)
    print("Required files: PASS")
    print("Final metrics: PASS")
    print("Relative links: PASS")
    print("Placeholder scan: PASS")


if __name__ == "__main__":
    main()
