# Portfolio

This is my portfolio showcasing NLP coursework and practical solutions I've developed. This repository includes both academic projects and a full-featured web application for NLP, graph analytics, and audio deepfake detection.

## Featured Project: CiceroWatch

**CiceroWatch** is a comprehensive web application that combines three powerful analysis tools:

### 1. NLP Tasks
- **Sentiment Analysis**: Analyze text sentiment using fine-tuned transformer models (Twitter, SST-2)
- **Zero-Shot Classification**: Classify text into custom categories without training
- **Named Entity Recognition (NER)**: Extract entities using HuggingFace, spaCy, or Stanza
- **Topic Modeling**: Discover topics using BERTopic, NMF, or K-means clustering
- **Text Embeddings**: Generate sentence embeddings using Sentence-BERT
- **POS Tagging & Dependency Parsing**: Linguistic analysis with spaCy and Stanza

### 2. Graph Analytics
- **Network Analysis**: Compute PageRank, betweenness, eigenvector centrality
- **Graph Algorithms**: BFS, triangle counting, degree analysis
- **Interactive Visualization**: Explore networks with vis.js
- **Ego Networks**: Analyze social network circles and features
- **GPU Acceleration**: Optional RAPIDS cuGraph support for large graphs

### 3. Audio Deepfake Detection
- **AI-Generated Audio Detection**: Identify spoofed or AI-generated audio using wav2vec 2.0
- **SSL Anti-Spoofing**: State-of-the-art model combining wav2vec 2.0 XLS-R (300M) with AASIST backend
- **Supported Formats**: FLAC and WAV audio files
- **Python 3.12 Compatible**: Uses HuggingFace transformers (no fairseq required)

### Getting Started

**Quick Setup (Linux/Mac):**
```bash
cd "Sentiment Docker Test"
bash setup_local.sh cpu          # For CPU-only
# OR
bash setup_local.sh cuda121      # For CUDA 12.1 GPU
```

**Manual Setup:**
```bash
cd "Sentiment Docker Test"
# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then install other requirements
pip install -r requirements-local.txt
# Run the server
python run_local.py
```

Open `http://localhost:8080` in your browser and explore the three tabs: NLP Tasks, Graph Analytics, and Audio Deepfake Detection.

See [AUDIO_SETUP.md](Sentiment%20Docker%20Test/AUDIO_SETUP.md) for detailed audio detection setup.

---

## Academic Projects

This portfolio also includes coursework from my NLP studies:

- **Naive Bayes**: Multinomial Naive Bayes document classifier (Advanced NLP)
- **N-Grams, POS Tagger, Viterbi Tagger**: Modified assignments from Intro to NLP
- **Folder Walker, Extension Finder, Pandas Practice, Folder Mapper**: Practical solutions from previous positions


Please see my current reading list for areas where I'm trying to develop my knowledge in my spare time:

# My Reading List

Here's what's on my bookshelf to be read when I'm not busy with schoolwork:

Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow -Geron

Python Machine Learning -Raschka & Mirjalili

Statistics for Linguists: an Introduction Using R -Winter

Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python -Jansen (There's a dedicated chapter on NLP and mining market analysis sites for clues)


# BERT resources:

https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

https://jalammar.github.io/illustrated-transformer/

# General NLP knowledge:

https://www.tensorflow.org/tutorials/text/word_embeddings

https://github.com/mhagiwara/100-nlp-papers

https://rare-technologies.com/word2vec-tutorial/

https://web.stanford.edu/~jurafsky/slp3/

https://github.com/flairNLP/flair
