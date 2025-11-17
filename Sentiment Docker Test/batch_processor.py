"""
Batch Spreadsheet Processor

Automated processing of multi-tab Excel files with sentiment and theme extraction.
Supports optional zero-shot labels for custom theme detection.
"""

import io
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from nlp import run_task, PRESETS, DEFAULT_ZS_LABELS

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    # Tasks to perform
    extract_sentiment: bool = True
    extract_themes: bool = True

    # Zero-shot configuration
    theme_labels: Optional[List[str]] = None  # Custom labels, or None for defaults
    sentiment_preset: str = "sentiment-twitter"
    theme_preset: str = "zeroshot-bart"

    # Column configuration
    text_column: Optional[str] = None  # Auto-detect if None

    # Sheet configuration
    sheets_to_process: Optional[List[str]] = None  # Process all if None

    # Output configuration
    add_confidence_scores: bool = True
    top_n_themes: int = 3  # Number of top themes to include

    # Performance
    batch_size: int = 32  # Process in batches for efficiency


def detect_text_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the text column in a dataframe

    Priority:
    1. Column named 'text' (case-insensitive)
    2. First column with 'text', 'comment', 'message', 'content' in name
    3. First string column with long enough content
    4. First string column
    """
    # Check for 'text' column (case-insensitive)
    for col in df.columns:
        if col.lower() == 'text':
            return col

    # Check for common text column names
    text_keywords = ['text', 'comment', 'message', 'content', 'description', 'post', 'tweet']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in text_keywords):
            if df[col].dtype == 'object':  # String type
                return col

    # Find first string column with substantial content
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check average length
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > 20:  # Likely text content
                    return col

    # Fallback: first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col

    raise ValueError("No suitable text column found in dataframe")


def process_sentiment_batch(
    texts: List[str],
    preset: str = "sentiment-twitter",
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Process sentiment analysis in batches

    Returns list of dicts with sentiment, confidence, and scores
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing sentiment batch {i//batch_size + 1} ({len(batch)} texts)")

        # Run sentiment analysis
        batch_results = run_task(batch, preset=preset)

        # Format results
        for result in batch_results:
            if isinstance(result, dict):
                labels = result.get("labels", [])
                scores = result.get("scores", [])

                if labels and scores:
                    max_idx = scores.index(max(scores))
                    sentiment = labels[max_idx]
                    confidence = scores[max_idx]

                    results.append({
                        "sentiment": sentiment,
                        "confidence": float(confidence),
                        "scores": {labels[j]: float(scores[j]) for j in range(len(labels))}
                    })
                else:
                    results.append({"sentiment": "unknown", "confidence": 0.0, "scores": {}})
            else:
                results.append({"sentiment": "unknown", "confidence": 0.0, "scores": {}})

    return results


def process_themes_batch(
    texts: List[str],
    labels: Optional[List[str]] = None,
    preset: str = "zeroshot-bart",
    batch_size: int = 32,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Process theme extraction (zero-shot classification) in batches

    Args:
        texts: List of texts to analyze
        labels: Custom theme labels (or None for defaults)
        preset: Zero-shot preset to use
        batch_size: Batch size for processing
        top_n: Number of top themes to return

    Returns list of dicts with themes and scores
    """
    if labels is None:
        labels = DEFAULT_ZS_LABELS

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing themes batch {i//batch_size + 1} ({len(batch)} texts)")

        # Run zero-shot classification
        batch_results = run_task(batch, preset=preset, labels=labels)

        # Format results
        for result in batch_results:
            if isinstance(result, dict):
                result_labels = result.get("labels", [])
                result_scores = result.get("scores", [])

                if result_labels and result_scores:
                    # Get top N themes
                    themes = []
                    for j in range(min(top_n, len(result_labels))):
                        themes.append({
                            "theme": result_labels[j],
                            "score": float(result_scores[j])
                        })

                    # Create theme columns
                    theme_dict = {}
                    for idx, theme_info in enumerate(themes):
                        theme_dict[f"theme_{idx+1}"] = theme_info["theme"]
                        theme_dict[f"theme_{idx+1}_score"] = theme_info["score"]

                    results.append(theme_dict)
                else:
                    # Empty result
                    theme_dict = {}
                    for idx in range(top_n):
                        theme_dict[f"theme_{idx+1}"] = ""
                        theme_dict[f"theme_{idx+1}_score"] = 0.0
                    results.append(theme_dict)
            else:
                # Empty result
                theme_dict = {}
                for idx in range(top_n):
                    theme_dict[f"theme_{idx+1}"] = ""
                    theme_dict[f"theme_{idx+1}_score"] = 0.0
                results.append(theme_dict)

    return results


def process_sheet(
    df: pd.DataFrame,
    config: BatchProcessingConfig,
    sheet_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process a single sheet with sentiment and theme extraction

    Args:
        df: Input dataframe
        config: Processing configuration
        sheet_name: Name of the sheet (for logging)

    Returns:
        Tuple of (processed_df, stats)
    """
    logger.info(f"Processing sheet '{sheet_name}' with {len(df)} rows")

    # Detect text column
    text_col = config.text_column
    if text_col is None:
        text_col = detect_text_column(df)

    logger.info(f"Using text column: '{text_col}'")

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in sheet '{sheet_name}'")

    # Extract texts and handle missing values
    texts = df[text_col].fillna("").astype(str).tolist()

    # Create result dataframe (copy original)
    result_df = df.copy()

    stats = {
        "sheet_name": sheet_name,
        "rows_processed": len(texts),
        "text_column": text_col
    }

    # Process sentiment
    if config.extract_sentiment:
        logger.info(f"Extracting sentiment from {len(texts)} texts...")
        sentiment_results = process_sentiment_batch(
            texts,
            preset=config.sentiment_preset,
            batch_size=config.batch_size
        )

        # Add sentiment columns
        result_df['sentiment'] = [r['sentiment'] for r in sentiment_results]

        if config.add_confidence_scores:
            result_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]

        stats['sentiment_distribution'] = result_df['sentiment'].value_counts().to_dict()

    # Process themes
    if config.extract_themes:
        logger.info(f"Extracting themes from {len(texts)} texts...")
        theme_results = process_themes_batch(
            texts,
            labels=config.theme_labels,
            preset=config.theme_preset,
            batch_size=config.batch_size,
            top_n=config.top_n_themes
        )

        # Add theme columns
        theme_df = pd.DataFrame(theme_results)
        result_df = pd.concat([result_df, theme_df], axis=1)

        # Collect theme stats
        theme_counts = {}
        for idx in range(1, config.top_n_themes + 1):
            col_name = f"theme_{idx}"
            if col_name in result_df.columns:
                counts = result_df[col_name].value_counts().head(10).to_dict()
                theme_counts[col_name] = counts

        stats['theme_distribution'] = theme_counts

    logger.info(f"Sheet '{sheet_name}' processing complete")
    return result_df, stats


def process_excel_file(
    file_bytes: bytes,
    config: BatchProcessingConfig
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Process multi-tab Excel file with sentiment and theme extraction

    Args:
        file_bytes: Excel file bytes
        config: Processing configuration

    Returns:
        Tuple of (output_excel_bytes, processing_stats)
    """
    logger.info("Starting batch Excel processing")

    # Read Excel file
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = excel_file.sheet_names

    logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")

    # Determine which sheets to process
    sheets_to_process = config.sheets_to_process
    if sheets_to_process is None:
        sheets_to_process = sheet_names
    else:
        # Validate sheet names
        invalid_sheets = [s for s in sheets_to_process if s not in sheet_names]
        if invalid_sheets:
            raise ValueError(f"Invalid sheet names: {invalid_sheets}")

    # Process each sheet
    processed_sheets = {}
    all_stats = {
        "total_sheets": len(sheets_to_process),
        "sheets": []
    }

    for sheet_name in sheets_to_process:
        logger.info(f"Reading sheet: {sheet_name}")
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)

        # Process sheet
        result_df, sheet_stats = process_sheet(df, config, sheet_name)
        processed_sheets[sheet_name] = result_df
        all_stats["sheets"].append(sheet_stats)

    # Write output Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    output_bytes = output.getvalue()

    logger.info("Batch processing complete")
    return output_bytes, all_stats
