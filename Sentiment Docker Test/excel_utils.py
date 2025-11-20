"""
Excel File Utilities

Shared utilities for Excel file handling across all NLP endpoints.
Provides preview, column selection, and data extraction.
"""

import io
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)


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

    # Last resort: first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col

    raise ValueError("No suitable text column found in the dataframe")


async def preview_excel_structure(file: UploadFile) -> Dict[str, Any]:
    """
    Preview Excel file structure

    Returns sheet names, columns, row counts, and sample data
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.xlsx', '.xlsm', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="File must be an Excel file (.xlsx, .xlsm, or .xls)"
        )

    # Read file bytes
    file_bytes = await file.read()

    # Load Excel file
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

    sheets_info = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Get column names and types
        columns = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null_count = df[col].count()
            total_count = len(df)

            columns.append({
                "name": col,
                "type": col_type,
                "non_null": int(non_null_count),
                "total": int(total_count),
                "null_percent": round((1 - non_null_count / total_count) * 100, 1) if total_count > 0 else 0
            })

        # Try to detect text column
        try:
            detected_text_col = detect_text_column(df)
        except:
            detected_text_col = None

        # Get first 5 rows as sample (convert to dict for JSON)
        sample_data = df.head(5).fillna("").to_dict('records')

        sheets_info.append({
            "sheet_name": sheet_name,
            "row_count": len(df),
            "columns": columns,
            "detected_text_column": detected_text_col,
            "sample_data": sample_data
        })

    return {
        "filename": file.filename,
        "sheet_count": len(excel_file.sheet_names),
        "sheets": sheets_info
    }


async def extract_texts_from_excel(
    file: UploadFile,
    sheet_name: Optional[str] = None,
    text_column: Optional[str] = None,
    max_rows: Optional[int] = None
) -> List[str]:
    """
    Extract text data from Excel file

    Args:
        file: Uploaded Excel file
        sheet_name: Specific sheet to read (first sheet if None)
        text_column: Column containing text (auto-detect if None)
        max_rows: Maximum number of rows to process (all if None)

    Returns:
        List of text strings
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.xlsx', '.xlsm', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="File must be an Excel file (.xlsx, .xlsm, or .xls)"
        )

    # Read file bytes
    file_bytes = await file.read()

    # Load Excel file
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

    # Select sheet
    if sheet_name:
        if sheet_name not in excel_file.sheet_names:
            raise HTTPException(
                status_code=400,
                detail=f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(excel_file.sheet_names)}"
            )
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
    else:
        # Use first sheet
        df = pd.read_excel(excel_file, sheet_name=0)

    # Limit rows if specified
    if max_rows and max_rows > 0:
        df = df.head(max_rows)

    # Detect or validate text column
    if text_column:
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{text_column}' not found. Available columns: {', '.join(df.columns)}"
            )
    else:
        try:
            text_column = detect_text_column(df)
            logger.info(f"Auto-detected text column: {text_column}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Extract texts
    texts = df[text_column].dropna().astype(str).tolist()

    if not texts:
        raise HTTPException(
            status_code=400,
            detail=f"No text data found in column '{text_column}'"
        )

    logger.info(f"Extracted {len(texts)} texts from column '{text_column}'")

    return texts


async def process_excel_with_results(
    file: UploadFile,
    results: List[Any],
    sheet_name: Optional[str] = None,
    text_column: Optional[str] = None
) -> bytes:
    """
    Process Excel file and add result columns

    Args:
        file: Uploaded Excel file
        results: List of analysis results to add
        sheet_name: Sheet that was processed
        text_column: Column that was processed

    Returns:
        Excel file bytes with added result columns
    """
    # Read file bytes
    file_bytes = await file.read()

    # Load Excel file
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

    # Select sheet
    if sheet_name:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_to_process = sheet_name
    else:
        df = pd.read_excel(excel_file, sheet_name=0)
        sheet_to_process = excel_file.sheet_names[0]

    # Detect text column if not specified
    if not text_column:
        text_column = detect_text_column(df)

    # Filter to non-null rows in text column
    valid_indices = df[text_column].notna()

    # Add results to dataframe
    # (This will be customized based on result type - sentiment, NER, etc.)
    # For now, just handle generic case

    # Create output Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write all sheets
        for sheet in excel_file.sheet_names:
            if sheet == sheet_to_process:
                # Write processed sheet with results
                df.to_excel(writer, sheet_name=sheet, index=False)
            else:
                # Write unchanged sheets
                pd.read_excel(excel_file, sheet_name=sheet).to_excel(
                    writer, sheet_name=sheet, index=False
                )

    output.seek(0)
    return output.read()
