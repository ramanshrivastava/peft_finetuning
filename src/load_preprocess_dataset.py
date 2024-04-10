from datasets import load_dataset
from transformers import AutoTokenizer

def load_tokenize_data(dataset_name, model_for_tokenizer):
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    #tokenize
    def tokenize(example):
        tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset

if __name__ == "__main__":
    dataset_name = "emotion"
    model_for_tokenizer = 'gpt2'
    tokenized_dataset = load_tokenize_data(dataset_name, model_for_tokenizer)
    print(tokenized_dataset)
    for split in  tokenized_dataset.keys():
        print(f"First example from {split}:")
        print(tokenized_dataset[split][0])
    print(type(tokenized_dataset))

