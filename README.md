# Emotion Recognizer

A Python-based emotion recognition system that analyzes text and predicts the emotional content. The system can identify five basic emotions: happiness, anger, sadness, fear, and surprise.

## Features

- Text preprocessing using NLTK
- Emotion classification using TF-IDF and Naive Bayes
- Interactive command-line interface
- Cross-validation and performance metrics
- Support for multiple emotions (happy, angry, sad, fear, surprise)

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/emotion_recognizer.git
cd emotion_recognizer
```

2. Install the required packages:
```bash
pip install pandas numpy scikit-learn nltk
```

3. Download required NLTK resources:
```bash
python download_nltk.py
```

## Usage

Run the emotion recognizer:
```bash
python emotion_recognizer.py
```

Type any text when prompted, and the system will analyze its emotional content. The system will display:
- The predicted emotion
- Confidence scores for each possible emotion
- A visual representation of the confidence levels

To exit the program, type 'quit'.

## Example

```
> I'm really excited about this project!
📊 Prediction: HAPPY

Confidence scores:
happy    : ████████████████░░░░ 0.82
surprise : ████░░░░░░░░░░░░░░░░ 0.12
fear     : ░░░░░░░░░░░░░░░░░░░░ 0.03
sad      : ░░░░░░░░░░░░░░░░░░░░ 0.02
angry    : ░░░░░░░░░░░░░░░░░░░░ 0.01
```

## License

MIT License 