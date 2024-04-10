from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

def foundation_model_eval(model_name, eval_dataset, num_labels, compute_metrics, output_dir="./results"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set tokenizer padding token to EOS token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=False,  # Disable training
        do_eval=True,    # Enable evaluation
        per_device_eval_batch_size=64,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,  # Use the validation split for evaluation
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()
   
    return results
