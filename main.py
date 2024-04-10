from src.load_preprocess_dataset import load_tokenize_data
from src.evaluation_metric import compute_metrics
from src.foundation_model_evaluation import foundation_model_eval
from src.finetune_lora import lora_model_tune_eval
from src.finetuned_infer import predict_emotion

def main():
    #configuration
    dataset_name = "emotion"
    model_name = "gpt2"
    num_labels = 6
    text = "im feeling quite sad and sorry for myself but ill snap out of it soon"
    model_checkpoint = "./models/fine_tuned/checkpoint-2000"

    #tokenized data
    tokenized_datset = load_tokenize_data(dataset_name, model_name)
    print(tokenized_datset)
    eval_dataset = tokenized_datset["validation"]

    #evaluate foundation model
    results = foundation_model_eval(model_name,eval_dataset, num_labels, compute_metrics)
    print("FOUNDATION model metrics", results)

    #fine-tune foundation model using lora and evaluate fine-tuned model
    results_tuned = lora_model_tune_eval(model_name, tokenized_datset, num_labels, compute_metrics)
    print("TUNED model metrics", results_tuned)

    #predict emotion
    predicted_class = predict_emotion(text, model_checkpoint)
    print("Predicted emotion class:", predicted_class)

if __name__ == "__main__":
    main()


