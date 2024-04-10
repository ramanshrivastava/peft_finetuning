from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
import torch

def predict_emotion(text, model_checkpoint, model_name="gpt2", num_labels=6):
    # Load the PEFT model for sequence classification
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')

    # Perform inference
    with torch.inference_mode():
        logits = model(**inputs).logits

    # Get the predicted class index
    predicted_class_index = logits.argmax(-1).item()

    return predicted_class_index

if __name__ == "__main__":
    text = "im feeling quite sad and sorry for myself but ill snap out of it soon"
    model_checkpoint = "../models/fine_tuned/checkpoint-2000"
    predicted_class = predict_emotion(text, model_checkpoint)
    print("Predicted class:", predicted_class)