from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "campaigns.csv"
ENRICHED_DATA_PATH = ROOT / "data" / "campaigns_enriched.csv"
SEARCH_CHUNKS_PATH = ROOT / "data" / "campaign_search_chunks.jsonl"

DEFAULT_IMAGE_URL = "/og_default.jpeg"
ROUNDING_TOLERANCE = 50.0
CHUNK_MAX_CHARS = 700

UNICODE_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)

THEME_PATTERNS: dict[str, list[str]] = {
    "medical": [
        "medical",
        "cancer",
        "dialysis",
        "hospital",
        "treatment",
        "patient",
        "patients",
        "therapy",
        "physio",
        "physiotherapy",
        "surgery",
        "health",
        "medicine",
        "blood",
        "injur",
    ],
    "education": [
        "education",
        "school",
        "student",
        "students",
        "university",
        "tuition",
        "faculty",
        "scholarship",
        "classroom",
        "teaching",
    ],
    "orphan_support": [
        "orphan",
        "orphans",
        "orphanage",
    ],
    "food": [
        "food",
        "bread",
        "iftar",
        "meal",
        "meals",
        "parcels",
        "parcel",
        "qurbani",
        "dessert",
        "desserts",
        "ramadan",
        "sadaqah",
        "loaf",
    ],
    "winter": [
        "winter",
        "warmth",
        "blanket",
        "blankets",
        "heating",
        "heater",
        "coat",
        "coats",
        "cold",
        "snow",
    ],
    "housing": [
        "housing",
        "house",
        "houses",
        "home",
        "homes",
        "roof",
        "tent",
        "tents",
        "village",
        "construction",
        "rebuild",
        "shelter",
        "rehabilitation",
        "housing units",
    ],
    "emergency": [
        "emergency",
        "earthquake",
        "urgent",
        "response",
        "under attack",
        "attack",
        "rescue",
        "relief",
        "crisis",
        "fire",
        "disaster",
    ],
    "water": [
        "water",
        "well",
        "wells",
        "drinking water",
        "thirst",
    ],
    "community": [
        "community",
        "initiative",
        "support",
        "festival",
        "giving day",
        "center",
        "centre",
        "fund",
        "campaign",
        "charitable",
        "waqf",
    ],
}

BENEFICIARY_PATTERNS: dict[str, list[str]] = {
    "orphans": ["orphan", "orphans", "orphanage"],
    "patients": [
        "patient",
        "patients",
        "cancer",
        "dialysis",
        "medical",
        "treatment",
        "therapy",
        "hospital",
        "injur",
    ],
    "students": [
        "student",
        "students",
        "school",
        "education",
        "university",
        "tuition",
        "faculty",
    ],
    "children": ["child", "children", "kid", "kids", "baby", "babies"],
    "families": [
        "family",
        "families",
        "widow",
        "widows",
        "household",
        "households",
        "refugee",
        "camp",
        "camps",
    ],
    "general": [],
}

LOCATION_PATTERNS: dict[str, list[str]] = {
    "Aleppo": ["aleppo"],
    "Arsal": ["arsal"],
    "Azaz": ["azaz"],
    "Columbus": ["columbus"],
    "Damascus": ["damascus"],
    "Deir al-Izz": ["deir al-izz", "deir al izz"],
    "Douma": ["douma"],
    "Daraa": ["daraa", "daraa"],
    "Flint": ["flint"],
    "Gaza": ["gaza"],
    "Gothenburg": ["gothenburg"],
    "Hama": ["hama"],
    "Homs": ["homs"],
    "Horan": ["horan"],
    "Idlib": ["idlib"],
    "Jordan": ["jordan"],
    "Kiswah": ["kiswah", "al-kiswah", "al kiswah"],
    "Lebanon": ["lebanon"],
    "Ohio": ["ohio"],
    "Palestine": ["palestine"],
    "Qusair": ["qusair", "al-qusair", "al qusair"],
    "Syria": ["syria", "syrian"],
    "Taftanaz": ["taftanaz"],
    "Turkey": ["turkey"],
}


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(UNICODE_REPLACEMENTS)
    text = re.sub(r"[\u200b-\u200f\ufeff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def normalize_key(value: object) -> object:
    text = normalize_text(value)
    if pd.isna(text):
        return pd.NA

    normalized = str(text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if normalized else pd.NA


def build_series_key(title_normalized: object) -> object:
    if pd.isna(title_normalized):
        return pd.NA

    series_key = re.sub(r"\b(?:19|20)\d{2}\b", " ", str(title_normalized))
    series_key = re.sub(r"\b\d+\b", " ", series_key)
    series_key = re.sub(r"\s+", " ", series_key).strip()
    return series_key if series_key else title_normalized


def join_text_parts(parts: Iterable[object]) -> object:
    cleaned_parts = [str(part).strip() for part in parts if not pd.isna(part) and str(part).strip()]
    if not cleaned_parts:
        return pd.NA
    return "\n\n".join(cleaned_parts)


def classify_amount_consistency(row: pd.Series) -> str:
    amount_fields = ["required_amount", "paid_amount", "left_amount"]
    if row[amount_fields].isna().any():
        return "missing_source"

    delta = float(row["left_amount"] - row["left_amount_clean"])
    if abs(delta) < 1e-9:
        return "ok"
    if abs(delta) <= ROUNDING_TOLERANCE:
        return "rounding_diff"
    return "major_mismatch"


def derive_year_confidence(row: pd.Series) -> str:
    if pd.isna(row["year_detected"]):
        return "unknown"
    if row["date_status"] == "certain" and row["year_source"] == "publishing_date":
        return "high"
    if row["date_status"] == "certain":
        return "medium"
    return "low"


def derive_year_label(row: pd.Series) -> str:
    if pd.isna(row["year_detected"]):
        return "unknown"
    if row["year_confidence"] in {"high", "medium"}:
        return str(int(row["year_detected"]))
    return "unknown"


def compute_keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.casefold()
    return sum(lowered.count(keyword.casefold()) for keyword in keywords)


def classify_from_patterns(text: object, patterns: dict[str, list[str]], default: str) -> str:
    if pd.isna(text):
        return default

    text_value = str(text)
    best_label = default
    best_score = 0
    for label, keywords in patterns.items():
        score = compute_keyword_score(text_value, keywords)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def extract_locations(text: object) -> str:
    if pd.isna(text):
        return "[]"

    lowered = str(text).casefold()
    matches: list[str] = []
    for canonical_name, aliases in LOCATION_PATTERNS.items():
        if any(alias.casefold() in lowered for alias in aliases):
            matches.append(canonical_name)
    return json.dumps(sorted(matches), ensure_ascii=False)


def split_into_chunks(text: object, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    if pd.isna(text):
        return []

    normalized_text = re.sub(r"\n{3,}", "\n\n", str(text)).strip()
    if not normalized_text:
        return []

    paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", normalized_text) if segment.strip()]
    raw_units: list[str] = []
    for paragraph in paragraphs:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", paragraph) if segment.strip()]
        raw_units.extend(sentences if sentences else [paragraph])

    chunks: list[str] = []
    current_chunk = ""
    for unit in raw_units:
        candidate = f"{current_chunk} {unit}".strip() if current_chunk else unit
        if len(candidate) <= max_chars:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        if len(unit) <= max_chars:
            current_chunk = unit
            continue

        start = 0
        while start < len(unit):
            chunks.append(unit[start : start + max_chars].strip())
            start += max_chars

    if current_chunk:
        chunks.append(current_chunk)

    return [chunk for chunk in chunks if chunk]


def build_search_chunk_records(row: pd.Series) -> tuple[str, list[dict[str, object]]]:
    chunk_texts = split_into_chunks(row["campaign_text"])
    location_mentions = json.loads(row["location_mentions"])
    chunk_records: list[dict[str, object]] = []

    for index, chunk_text in enumerate(chunk_texts):
        chunk_records.append(
            {
                "chunk_id": f"{int(row['campaign_id'])}-{index}",
                "campaign_id": int(row["campaign_id"]),
                "title": row["title"],
                "year_label": row["year_label"],
                "campaign_theme": row["campaign_theme"],
                "beneficiary_group": row["beneficiary_group"],
                "funding_status": row["funding_status"],
                "funding_ratio": row["funding_ratio"],
                "donations_count": int(row["donations_count"]),
                "location_mentions": location_mentions,
                "text": chunk_text,
            }
        )

    return json.dumps(chunk_records, ensure_ascii=False), chunk_records


def compute_record_quality_score(row: pd.Series) -> int:
    score = 100

    if row["amount_consistency_flag"] == "missing_source":
        score -= 35
    elif row["amount_consistency_flag"] == "major_mismatch":
        score -= 20
    elif row["amount_consistency_flag"] == "rounding_diff":
        score -= 5

    if row["year_confidence"] == "unknown":
        score -= 20
    elif row["year_confidence"] == "low":
        score -= 15
    elif row["year_confidence"] == "medium":
        score -= 5

    if not row["has_details_text"]:
        score -= 10
    if row["has_default_image"]:
        score -= 3

    return max(score, 0)


def validate_outputs(df: pd.DataFrame, chunk_records: list[dict[str, object]]) -> None:
    expected_columns = {
        "campaign_id",
        "url",
        "title",
        "overview_text",
        "details_text",
        "required_amount",
        "paid_amount",
        "left_amount",
        "scraped_at",
        "scrape_date",
        "left_amount_clean",
        "amount_consistency_flag",
        "is_overfunded",
        "overfunded_amount",
        "year_confidence",
        "campaign_text",
        "campaign_text_short",
        "funding_ratio",
        "funding_status",
        "engagement_total",
        "text_length",
        "has_details_text",
        "has_default_image",
        "has_uncertain_year",
        "title_normalized",
        "campaign_series_key",
        "record_quality_score",
        "campaign_theme",
        "beneficiary_group",
        "location_mentions",
        "year_label",
        "search_chunks",
    }

    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

    if df["campaign_id"].duplicated().any():
        raise ValueError("campaign_id must remain unique after enrichment")

    comparable = df[df[["required_amount", "paid_amount"]].notna().all(axis=1)].copy()
    recomputed_ratio = comparable["paid_amount"] / comparable["required_amount"]
    if not recomputed_ratio.fillna(-1).round(6).equals(comparable["funding_ratio"].fillna(-1).round(6)):
        raise ValueError("funding_ratio validation failed")

    if not comparable["left_amount_clean"].round(6).equals((comparable["required_amount"] - comparable["paid_amount"]).round(6)):
        raise ValueError("left_amount_clean validation failed")

    null_amount_rows = df[df[["required_amount", "paid_amount", "left_amount"]].isna().any(axis=1)]
    if not (null_amount_rows["amount_consistency_flag"] == "missing_source").all():
        raise ValueError("Rows with missing amounts must be flagged as missing_source")

    uncertain_year_rows = df[df["has_uncertain_year"]]
    if not uncertain_year_rows.empty and not uncertain_year_rows["year_label"].eq("unknown").all():
        raise ValueError("Uncertain years should be labeled as unknown")

    overfunded_rows = comparable[comparable["paid_amount"] > comparable["required_amount"]]
    if not overfunded_rows.empty and not overfunded_rows["is_overfunded"].all():
        raise ValueError("Overfunded campaigns must be preserved and flagged")

    if not chunk_records:
        raise ValueError("Search chunks output is empty")

    if not any("cancer" in record["text"].casefold() for record in chunk_records):
        raise ValueError("Expected cancer-related search chunk not found")

    if not any(record["year_label"] == "unknown" for record in chunk_records):
        raise ValueError("Expected uncertain-year search chunk not found")

    if not any(record["funding_status"] == "overfunded" for record in chunk_records):
        raise ValueError("Expected overfunded search chunk not found")


def prepare_campaigns_dataset() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    df = pd.read_csv(RAW_DATA_PATH)
    if "subtitle" in df.columns:
        df = df.drop(columns=["subtitle"])

    text_columns = ["title", "overview_text", "details_text", "image_url", "year_source", "date_status", "url"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(normalize_text)

    df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
    df["scrape_date"] = df["scraped_at"].dt.date.astype("string")

    df["campaign_text"] = df.apply(
        lambda row: join_text_parts([row["title"], row["overview_text"], row["details_text"]]),
        axis=1,
    )
    df["campaign_text_short"] = df.apply(
        lambda row: join_text_parts([row["title"], row["overview_text"]]),
        axis=1,
    )

    df["left_amount_clean"] = df["required_amount"] - df["paid_amount"]
    df.loc[df[["required_amount", "paid_amount"]].isna().any(axis=1), "left_amount_clean"] = pd.NA
    df["amount_consistency_flag"] = df.apply(classify_amount_consistency, axis=1)

    df["is_overfunded"] = (
        df[["required_amount", "paid_amount"]].notna().all(axis=1) & (df["paid_amount"] > df["required_amount"])
    )
    df["overfunded_amount"] = (df["paid_amount"] - df["required_amount"]).clip(lower=0)
    df.loc[df[["required_amount", "paid_amount"]].isna().any(axis=1), "overfunded_amount"] = pd.NA

    df["year_confidence"] = df.apply(derive_year_confidence, axis=1)
    df["has_uncertain_year"] = df["year_confidence"].isin(["low", "unknown"])
    df["year_label"] = df.apply(derive_year_label, axis=1)

    df["funding_ratio"] = df["paid_amount"] / df["required_amount"]
    df.loc[df[["required_amount", "paid_amount"]].isna().any(axis=1), "funding_ratio"] = pd.NA

    df["funding_status"] = "unknown"
    df.loc[df["funding_ratio"].isna(), "funding_status"] = "unknown"
    df.loc[df["funding_ratio"].eq(0), "funding_status"] = "not_started"
    df.loc[(df["funding_ratio"] > 0) & (df["funding_ratio"] < 1), "funding_status"] = "in_progress"
    df.loc[df["funding_ratio"].eq(1), "funding_status"] = "fully_funded"
    df.loc[df["funding_ratio"] > 1, "funding_status"] = "overfunded"

    df["engagement_total"] = (
        df["donations_count"] + df["comments_count"] + df["updates_count"] + df["shares_count"]
    )
    df["text_length"] = df["campaign_text"].fillna("").str.len()
    df["has_details_text"] = df["details_text"].notna()
    df["has_default_image"] = df["image_url"].fillna("").eq(DEFAULT_IMAGE_URL)

    df["title_normalized"] = df["title"].apply(normalize_key)
    df["campaign_series_key"] = df["title_normalized"].apply(build_series_key)

    df["campaign_theme"] = df["campaign_text"].apply(
        lambda text: classify_from_patterns(text, THEME_PATTERNS, default="other")
    )
    df["beneficiary_group"] = df["campaign_text"].apply(
        lambda text: classify_from_patterns(text, BENEFICIARY_PATTERNS, default="general")
    )
    df["location_mentions"] = df["campaign_text"].apply(extract_locations)
    df["record_quality_score"] = df.apply(compute_record_quality_score, axis=1)

    serialized_chunks: list[str] = []
    flat_chunk_records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        serialized, flat_records = build_search_chunk_records(row)
        serialized_chunks.append(serialized)
        flat_chunk_records.extend(flat_records)
    df["search_chunks"] = serialized_chunks

    validate_outputs(df, flat_chunk_records)
    return df, flat_chunk_records


def write_outputs(df: pd.DataFrame, chunk_records: list[dict[str, object]]) -> None:
    df_to_save = df.copy()
    df_to_save["scraped_at"] = df_to_save["scraped_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df_to_save.to_csv(ENRICHED_DATA_PATH, index=False, encoding="utf-8")

    with SEARCH_CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for record in chunk_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(df: pd.DataFrame, chunk_records: list[dict[str, object]]) -> None:
    print(f"Loaded {len(df)} campaigns from {RAW_DATA_PATH.name}")
    print(f"Wrote enriched dataset to {ENRICHED_DATA_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(chunk_records)} search chunks to {SEARCH_CHUNKS_PATH.relative_to(ROOT)}")
    print("Amount consistency flags:", df["amount_consistency_flag"].value_counts(dropna=False).to_dict())
    print("Funding status:", df["funding_status"].value_counts(dropna=False).to_dict())
    print("Year confidence:", df["year_confidence"].value_counts(dropna=False).to_dict())
    print("Top campaign themes:", df["campaign_theme"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    campaigns_df, search_chunks = prepare_campaigns_dataset()
    write_outputs(campaigns_df, search_chunks)
    print_summary(campaigns_df, search_chunks)
