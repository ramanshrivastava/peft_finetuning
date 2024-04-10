# metrics to be computed to compare performance

import numpy as np
import evaluate

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {'accuracy': accuracy}


