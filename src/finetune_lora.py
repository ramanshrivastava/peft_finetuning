
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification

def lora_model_tune_eval(model_name, dataset, num_labels, compute_metrics, output_dir="./models/fine_tuned"):
    lora_config = LoraConfig(
                            task_type=TaskType.SEQ_CLS, 
                            r=2, 
                            lora_alpha=16, 
                            lora_dropout=0.1, 
                            bias="none")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    lora_model = get_peft_model(model, lora_config)
    print(lora_model.config)
    lora_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set tokenizer padding token to EOS token
    tokenizer.pad_token = tokenizer.eos_token
    lora_model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-3,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
)

    trainer.train()

    results = trainer.evaluate()

    return results

