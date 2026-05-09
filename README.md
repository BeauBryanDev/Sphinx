# 𓁹 Sphinx: Middle Egypt Transliteration Pipeline

Welcome to the **Sphinx Project**, an advanced, end-to-end Computer Vision and AI pipeline designed to detect, semantically correct, and transliterate Middle Egyptian hieroglyphs from raw images.

## 📖 Overview

The goal of this project is to provide a robust AI-powered tool capable of interpreting ancient Egyptian glyphs. By combining state-of-the-art object detection algorithms with a custom-built semantic correction engine and Large Language Models, this pipeline smoothly extracts and reads Gardiner codes from images.

## 🏗️ Architecture & Tech Stack

The application is built across a robust, modern full-stack ecosystem:

- **Frontend:** Next.js
- **Backend:** FastAPI
- **Database:** PostgreSQL

## 🧠 Machine Learning & Inference Pipeline

Our full inference pipeline consists of several specialized layers:

1. **Object Detection (YOLOv11):** 
   - We successfully trained an initial **YOLOv11 Large** model (v1).
   - Currently aiming to train **v2**, which focuses on detecting multiple glyphs per image in complex, dense visual groupings.
2. **Semantic Correction Engine (Trie DSA):**
   - Raw YOLO predictions can misinterpret sequences.
   - To semantically correct these predictions, we have engineered a **Lexicon Trie (Prefix Tree) Data Structure**. It effectively aligns and validates the raw sequence to known dictionaries of Middle Egyptian Gardiner codes.
3. **LLM Translation (OpenAI GPT):**
   - The corrected sequence of Gardiner codes is compiled and dispatched to the **OpenAI API (GPT)** via prompt engineering, bringing forth a highly accurate transliteration and English translation.

## 🚀 Roadmap

- [x] YOLOv11 Large (v1) Training
- [x] Lexicon Trie Data Structure for Semantic Correction
- [ ] YOLOv11 v2 Training (Multi-glyph inference)
- [ ] FastAPI Backend Implementation
- [ ] PostgreSQL Database Setup
- [ ] Next.js Frontend Development
- [ ] GPT Integation and End-to-End Pipeline Linking

## 🛠️ Repository Highlights

- `model_train.py`, `annotate_yolo.py`, `yolo_annotate.py`: YOLO training and annotation utilities.
- `sphinx_trie.py`, `sphinx_corrector.py`: Core logic for the semantic Trie and Gardiner code auto-correction.
- `pre_processing.py`, `enhance_img.py`: Image preprocessing to prepare glyphs for prediction.

---
*Built to bring ancient texts smoothly into the digital age.*
