# Fine-tuning GPT-2 using LORA to infer Emotion class of Twitter messages

Fine-tuning the GPT-2 model to classify emotions in tweets. The process includes loading a pre-trained foundation model, performing parameter-efficient fine-tuning (PEFT) using LoRA, evaluating performance, and comparing the foundation model with the fine-tuned model.

## Choices

- **Model**:
  - GPT-2 is used because it is compatible with the sequence classification task and compatible with LoRA.
- **PEFT Technique**:
  - LoRA (Low-Rank Adaptation) is utilized as it allows for efficient fine-tuning without significant impact on the original model weights.
- **Evaluation**:
  - Hugging Face's `Trainer` `.evaluate` method is used to compare the performance of both the foundation and the fine-tuned models.
- **Dataset**:
  - The dataset consists of labeled tweets on emotion, provided by Hugging Face `datasets`.
    - [Dataset Link](https://huggingface.co/datasets/dair-ai/emotion)

## Steps

1. Choose the dataset: **DONE**
2. Choose the foundation model: **DONE**
3. Perform inference for the text classification task with the foundation model: **DONE**
4. Evaluate the performance of the foundation model: **DONE**
5. Load the foundation model as a PEFT model: **DONE**
6. Define the PEFT/LORA configuration: **DONE**
7. Train the LoRA model with Hugging Face Trainer: **DONE**
8. Evaluate the PEFT model: **DONE**
9. Save the PEFT model: **DONE**
10. Load the saved PEFT model from local storage: **DONE**
11. Run inference and generate text/label with the tuned model: **DONE**

## Results:
- Can be found in the notebook run 
- Evaluation accuracy of the foundation model on this task is 'eval_accuracy': 0.096
- While the Evaluation accuracy of the tuned model is 'eval_accuracy': 0.9225
- This is almost a 10x increase