"""
Training Data Store for Few-Shot Learning

Stores human-labeled examples and retrieves similar ones for few-shot prompting.
Uses sentence embeddings for similarity matching.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Storage path for training data
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/app/training_data"))
TRAINING_DATA_FILE = TRAINING_DATA_DIR / "examples.json"
EMBEDDINGS_FILE = TRAINING_DATA_DIR / "embeddings.npy"

# Lazy-load embedding model to avoid startup cost
_embedding_model = None

def get_embedding_model():
    """Lazy-load the sentence transformer model"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model for embeddings
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback")
            _embedding_model = "fallback"
    return _embedding_model


class TrainingExample:
    """A single training example from human-labeled data"""
    def __init__(
        self,
        text: str,
        stance: Optional[str] = None,
        stance_confidence: Optional[float] = None,
        themes: Optional[List[str]] = None,
        themes_confidence: Optional[List[float]] = None,
        narrative: Optional[str] = None,
        source_file: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.stance = stance
        self.stance_confidence = stance_confidence
        self.themes = themes or []
        self.themes_confidence = themes_confidence or []
        self.narrative = narrative
        self.source_file = source_file
        self.url = url
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "stance": self.stance,
            "stance_confidence": self.stance_confidence,
            "themes": self.themes,
            "themes_confidence": self.themes_confidence,
            "narrative": self.narrative,
            "source_file": self.source_file,
            "url": self.url,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        return cls(**data)

    def has_stance(self) -> bool:
        return self.stance is not None and self.stance.strip() != ""

    def has_themes(self) -> bool:
        return len(self.themes) > 0

    def has_narrative(self) -> bool:
        return self.narrative is not None and self.narrative.strip() != ""


class TrainingStore:
    """
    Storage and retrieval system for training examples.

    Supports:
    - Adding examples from human-labeled Excel files
    - Generating embeddings for similarity matching
    - Retrieving most similar examples for few-shot learning
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or TRAINING_DATA_DIR
        self.data_file = self.data_dir / "examples.json"
        self.embeddings_file = self.data_dir / "embeddings.npy"

        self.examples: List[TrainingExample] = []
        self.embeddings: Optional[np.ndarray] = None

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self._load()

    def _load(self):
        """Load existing training data from disk"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.examples = [TrainingExample.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.examples)} training examples")
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                self.examples = []

        if self.embeddings_file.exists():
            try:
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self.embeddings = None

    def _save(self):
        """Save training data to disk"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in self.examples], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.examples)} training examples")
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

        if self.embeddings is not None:
            try:
                np.save(self.embeddings_file, self.embeddings)
                logger.info(f"Saved embeddings: {self.embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to save embeddings: {e}")

    def add_example(self, example: TrainingExample) -> int:
        """Add a single training example"""
        self.examples.append(example)
        return len(self.examples) - 1

    def add_examples(self, examples: List[TrainingExample]) -> int:
        """Add multiple training examples"""
        self.examples.extend(examples)
        return len(examples)

    def clear(self):
        """Clear all training data"""
        self.examples = []
        self.embeddings = None
        # Remove files
        if self.data_file.exists():
            self.data_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        logger.info("Cleared all training data")

    def generate_embeddings(self) -> int:
        """Generate embeddings for all examples"""
        if not self.examples:
            logger.warning("No examples to generate embeddings for")
            return 0

        model = get_embedding_model()
        if model == "fallback":
            # Simple fallback using word overlap (not ideal but works)
            logger.warning("Using fallback embedding method (word overlap)")
            self.embeddings = None
            return 0

        texts = [e.text for e in self.examples]
        logger.info(f"Generating embeddings for {len(texts)} examples...")

        self.embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Generated embeddings: {self.embeddings.shape}")

        # Save after generating
        self._save()

        return len(texts)

    def find_similar(
        self,
        query_text: str,
        top_k: int = 5,
        task: Optional[str] = None
    ) -> List[Tuple[TrainingExample, float]]:
        """
        Find the most similar training examples to the query text.

        Args:
            query_text: The text to find similar examples for
            top_k: Number of similar examples to return
            task: Filter by task type ('stance', 'themes', 'narrative')

        Returns:
            List of (example, similarity_score) tuples
        """
        if not self.examples:
            return []

        # Filter examples by task if specified
        if task:
            if task == 'stance':
                filtered_indices = [i for i, e in enumerate(self.examples) if e.has_stance()]
            elif task == 'themes':
                filtered_indices = [i for i, e in enumerate(self.examples) if e.has_themes()]
            elif task == 'narrative':
                filtered_indices = [i for i, e in enumerate(self.examples) if e.has_narrative()]
            else:
                filtered_indices = list(range(len(self.examples)))
        else:
            filtered_indices = list(range(len(self.examples)))

        if not filtered_indices:
            return []

        model = get_embedding_model()

        if model == "fallback" or self.embeddings is None:
            # Fallback: simple word overlap similarity
            return self._find_similar_fallback(query_text, filtered_indices, top_k)

        # Generate query embedding
        query_embedding = model.encode([query_text], convert_to_numpy=True)[0]

        # Calculate cosine similarity
        filtered_embeddings = self.embeddings[filtered_indices]
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            original_idx = filtered_indices[idx]
            results.append((self.examples[original_idx], float(similarities[idx])))

        return results

    def _find_similar_fallback(
        self,
        query_text: str,
        filtered_indices: List[int],
        top_k: int
    ) -> List[Tuple[TrainingExample, float]]:
        """Fallback similarity using word overlap (Jaccard)"""
        query_words = set(query_text.lower().split())

        scores = []
        for idx in filtered_indices:
            example_words = set(self.examples[idx].text.lower().split())
            intersection = len(query_words & example_words)
            union = len(query_words | example_words)
            jaccard = intersection / union if union > 0 else 0
            scores.append((idx, jaccard))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [(self.examples[idx], score) for idx, score in scores[:top_k]]

    def get_few_shot_examples(
        self,
        query_text: str,
        task: str,
        num_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get formatted few-shot examples for a specific task.

        Returns examples in a format ready for prompt construction.
        """
        similar = self.find_similar(query_text, top_k=num_examples, task=task)

        examples = []
        for example, score in similar:
            if task == 'stance':
                examples.append({
                    "text": example.text[:500],  # Truncate for prompt
                    "stance": example.stance,
                    "confidence": example.stance_confidence,
                    "similarity": score
                })
            elif task == 'themes':
                examples.append({
                    "text": example.text[:500],
                    "themes": example.themes,
                    "confidence": example.themes_confidence,
                    "similarity": score
                })
            elif task == 'narrative':
                examples.append({
                    "text": example.text[:500],
                    "themes": example.themes,
                    "narrative": example.narrative,
                    "similarity": score
                })

        return examples

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the training data"""
        total = len(self.examples)
        with_stance = sum(1 for e in self.examples if e.has_stance())
        with_themes = sum(1 for e in self.examples if e.has_themes())
        with_narrative = sum(1 for e in self.examples if e.has_narrative())

        # Get unique themes
        all_themes = set()
        for e in self.examples:
            all_themes.update(e.themes)

        # Get stance distribution
        stance_dist = {}
        for e in self.examples:
            if e.stance:
                stance_dist[e.stance] = stance_dist.get(e.stance, 0) + 1

        return {
            "total_examples": total,
            "with_stance": with_stance,
            "with_themes": with_themes,
            "with_narrative": with_narrative,
            "unique_themes": list(all_themes),
            "stance_distribution": stance_dist,
            "has_embeddings": self.embeddings is not None,
            "embedding_shape": list(self.embeddings.shape) if self.embeddings is not None else None
        }


# Global training store instance
_training_store: Optional[TrainingStore] = None

def get_training_store() -> TrainingStore:
    """Get the global training store instance"""
    global _training_store
    if _training_store is None:
        _training_store = TrainingStore()
    return _training_store


def parse_excel_training_data(
    excel_bytes: bytes,
    text_column: str = "Body",
    stance_column: str = "Stance",
    themes_column: str = "Themes",
    narrative_column: str = "Narrative",
    url_column: str = "URL",
    source_filename: str = "unknown"
) -> List[TrainingExample]:
    """
    Parse a human-labeled Excel file into training examples.

    Tries to auto-detect column names if exact matches not found.
    """
    import pandas as pd
    import io

    excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
    examples = []

    for sheet_name in excel_file.sheet_names:
        # Skip guidance/pivot sheets
        if sheet_name.lower() in ['guidance', 'pivot'] or 'pivot' in sheet_name.lower():
            continue

        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Auto-detect columns (case-insensitive)
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()

            # Text column
            if col_lower in ['body', 'text', 'content', 'post', 'tweet', 'message']:
                col_map['text'] = col
            elif 'body' in col_lower or 'text' in col_lower or 'content' in col_lower:
                col_map.setdefault('text', col)
            elif 'translated' in col_lower:
                col_map['text_translated'] = col

            # Stance/Sentiment column (sentiment is analogous to stance)
            if col_lower == 'stance' or col_lower == 'sentiment':
                col_map['stance'] = col
            elif ('stance' in col_lower or 'sentiment' in col_lower) and 'confidence' not in col_lower:
                col_map.setdefault('stance', col)

            # Stance/Sentiment confidence
            if ('stance' in col_lower or 'sentiment' in col_lower) and 'confidence' in col_lower:
                col_map['stance_confidence'] = col

            # Themes column
            if col_lower == 'themes':
                col_map['themes'] = col
            elif 'theme' in col_lower and 'confidence' not in col_lower:
                col_map.setdefault('themes', col)

            # Themes confidence
            if 'theme' in col_lower and 'confidence' in col_lower:
                col_map['themes_confidence'] = col

            # Narrative column
            if col_lower == 'narrative':
                col_map['narrative'] = col
            elif 'narrative' in col_lower or 'summary' in col_lower:
                col_map.setdefault('narrative', col)

            # URL column
            if col_lower == 'url':
                col_map['url'] = col
            elif 'url' in col_lower or 'link' in col_lower:
                col_map.setdefault('url', col)

        # Override with explicit column names if provided
        if text_column in df.columns:
            col_map['text'] = text_column
        if stance_column in df.columns:
            col_map['stance'] = stance_column
        if themes_column in df.columns:
            col_map['themes'] = themes_column
        if narrative_column in df.columns:
            col_map['narrative'] = narrative_column
        if url_column in df.columns:
            col_map['url'] = url_column

        # Prefer translated text if available
        text_col = col_map.get('text_translated') or col_map.get('text')

        if not text_col:
            logger.warning(f"Sheet '{sheet_name}': No text column found, skipping")
            continue

        logger.info(f"Sheet '{sheet_name}': Using columns {col_map}")

        # Extract examples
        for idx, row in df.iterrows():
            text = row.get(text_col)
            if pd.isna(text) or not str(text).strip():
                continue

            text = str(text).strip()

            # Skip very short texts
            if len(text) < 20:
                continue

            # Parse stance
            stance = None
            stance_conf = None
            if 'stance' in col_map:
                stance_val = row.get(col_map['stance'])
                if not pd.isna(stance_val):
                    stance = str(stance_val).strip().upper()
            if 'stance_confidence' in col_map:
                conf_val = row.get(col_map['stance_confidence'])
                if not pd.isna(conf_val):
                    try:
                        stance_conf = float(conf_val)
                    except:
                        pass

            # Parse themes
            themes = []
            themes_conf = []
            if 'themes' in col_map:
                themes_val = row.get(col_map['themes'])
                if not pd.isna(themes_val):
                    # Themes might be comma-separated or a single value
                    themes_str = str(themes_val).strip()
                    if ',' in themes_str:
                        themes = [t.strip() for t in themes_str.split(',') if t.strip()]
                    elif themes_str:
                        themes = [themes_str]
            if 'themes_confidence' in col_map:
                conf_val = row.get(col_map['themes_confidence'])
                if not pd.isna(conf_val):
                    try:
                        # Might be comma-separated floats
                        conf_str = str(conf_val)
                        if ',' in conf_str:
                            themes_conf = [float(c.strip()) for c in conf_str.split(',')]
                        else:
                            themes_conf = [float(conf_str)]
                    except:
                        pass

            # Parse narrative
            narrative = None
            if 'narrative' in col_map:
                narr_val = row.get(col_map['narrative'])
                if not pd.isna(narr_val):
                    narrative = str(narr_val).strip()

            # Parse URL
            url = None
            if 'url' in col_map:
                url_val = row.get(col_map['url'])
                if not pd.isna(url_val):
                    url = str(url_val).strip()

            # Only add if there's at least one label
            if stance or themes or narrative:
                example = TrainingExample(
                    text=text,
                    stance=stance,
                    stance_confidence=stance_conf,
                    themes=themes,
                    themes_confidence=themes_conf,
                    narrative=narrative,
                    url=url,
                    source_file=f"{source_filename}:{sheet_name}",
                    metadata={"row_index": idx}
                )
                examples.append(example)

    return examples
