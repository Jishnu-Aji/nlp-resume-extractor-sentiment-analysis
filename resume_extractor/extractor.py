# extractor.py
"""
Resume Information Extractor

This script reads resumes from `sample_resumes.txt`, extracts email addresses
and phone numbers using regular expressions, and writes the structured results
to `output.json`.

Usage:
    python extractor.py
"""

import re
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Regular expression patterns
# ---------------------------------------------------------------------------
# Email pattern – matches most common email formats
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Phone pattern – supports formats such as:
#   123-456-7890, (123) 456-7890, 123 456 7890, +1 1234567890, 123.456.7890
PHONE_REGEX = re.compile(
    r"(?:(?:\+?\d{1,3})[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def extract_emails(text: str) -> list[str]:
    """Return a list of unique email addresses found in *text*.
    """
    return list(set(EMAIL_REGEX.findall(text)))


def extract_phones(text: str) -> list[str]:
    """Return a list of unique phone numbers found in *text*.
    The raw match is returned; further formatting can be applied if needed.
    """
    return list(set(PHONE_REGEX.findall(text)))


def process_resume(resume_text: str) -> dict:
    """Extract contact information from a single resume string.

    Returns a dictionary with keys `emails` and `phones`.
    """
    emails = extract_emails(resume_text)
    phones = extract_phones(resume_text)
    return {"emails": emails, "phones": phones}

# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------
def main():
    base_path = Path(__file__).parent
    sample_file = base_path / "sample_resumes.txt"
    output_file = base_path / "output.json"

    if not sample_file.is_file():
        print(f"[ERROR] Sample file not found: {sample_file}")
        return

    results = []
    # Each line in the sample file is treated as a separate resume entry.
    with sample_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            extracted = process_resume(line)
            results.append({"resume_id": idx, **extracted})

    # Write results to JSON with pretty formatting.
    try:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Extraction complete. Results saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}")

if __name__ == "__main__":
    main()
