# Intents-based NLP Chatbot (Keras + NLTK + tkinter)

## Setup

1. Create a virtual environment (recommended) and install requirements:

```bash
python -m venv .venv
. .venv/Scripts/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Download required NLTK data is handled automatically on first run.

## Data
- Edit `data/intents.json` to add your intents, patterns, and responses.

## Train
```bash
python train.py
```
This produces `model.h5`, `words.pkl`, and `classes.pkl` in the project root.

## Run GUI
```bash
python gui.py
```
A window will open; type a message and press Enter or click Send.

## Notes
- The model is a simple feedforward network trained on bag-of-words features.
- For better accuracy, expand patterns and add more intents.

