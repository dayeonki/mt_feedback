import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune")
    parser.add_argument("--base_model_name", type=str)
    parser.add_argument("--adapter_model_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--cache_dir", type=str)
    args = parser.parse_args()

    print("Model: ", args.adapter_model_name)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name,
                                                cache_dir = args.cache_dir,
                                                device_map = "auto")
    model = PeftModel.from_pretrained(model, args.adapter_model_name, device_map='auto')
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    print("\n==================================\n")