from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, model_path="models/trained_model"):
        """Initialize sentiment analyzer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def predict(self, text):
        """Predict sentiment of a single text"""
        result = self.pipeline(text)
        return result[0]['label'], result[0]['score']

    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        return self.pipeline(texts)

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    text = "ይህ መርከብ በጣም ጥሩ ነው"  # "This ship is very good"
    label, score = analyzer.predict(text)
    print(f"Text: {text}\nSentiment: {label} (confidence: {score:.2f})")