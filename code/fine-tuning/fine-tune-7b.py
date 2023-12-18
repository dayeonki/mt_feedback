import argparse
import datetime
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub.hf_api import HfFolder
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def main(args):
    dataset = load_dataset(args.hf_dataset)
    print("\n====================Prepare dataset=======================\n")
    print(dataset)

    print("\n====================Prepare model=======================\n")
    model_name = "meta-llama/Llama-2-7b-hf"
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(peft_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        cache_dir = args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = args.cache_dir)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        num_train_epochs=5, 
        output_dir=args.output_dir,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=2
    )
    print(training_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="template",
        max_seq_length=512,
        peft_config=peft_config,
    )
    print(trainer)
    
    print("\n====================Start training=======================\n")
    trainer.train()

    print("\n====================Save model/tokenizer=======================\n")
    trainer.save_model(args.trainer_save_dir)
    tokenizer.save_pretrained(args.tokenizer_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune")
    parser.add_argument("--hf_dataset", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--trainer_save_dir", type=str)
    parser.add_argument("--tokenizer_save_dir", type=str)
    parser.add_argument("--cache_dir", type=str)
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)
    seed = 42

    main(args)

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")