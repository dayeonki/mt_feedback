import torch
import argparse
import datetime
import jsonlines
import random
from huggingface_hub.hf_api import HfFolder
from transformers import pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer

random.seed(24)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stop_tokens=None, prompt_len=0):
        super().__init__()
        if stop_tokens is None:
            stop_tokens = []
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist) + i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen


def generate_text(args, pipe, tokenizer, prompt):
    stop_token = f"{args.source_lang}:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:],
                                                                  prompt_len=input_ids.shape[1])])
    return pipe(prompt,
                stopping_criteria=stopping_criteria,
                return_full_text=False)[0]['generated_text'][:-len(stop_token)].strip()


def find_few_shots(args, pool_data_path, few_shot_type, source_sent, target_sent, target_error_severity, error_span, error_type, num_few_shots):
    total_dataset = []
    with jsonlines.open(pool_data_path) as f:
        for line in f.iter():
            total_dataset.append(line)
    
    error_aligned_shots = []
    for data in total_dataset:
        for severity in data["severity"]:
            if severity in target_error_severity and data["original_src"] != source_sent and data["original_tgt"] != target_sent:
                error_aligned_shots.append(data)
    
    if len(error_aligned_shots) < num_few_shots:
        remaining_shots = num_few_shots - len(error_aligned_shots)
        remain = random.sample(total_dataset, remaining_shots)
        for r in remain:
            error_aligned_shots.append(r)
    
    if few_shot_type == "random":
        few_shots = random.sample(total_dataset, num_few_shots)
    elif few_shot_type == "error_aligned":
        few_shots = random.sample(error_aligned_shots, num_few_shots)
    
    few_shot_prompts = []
    for shot in few_shots:
        few_shot_prompts.append(instructscore_fewshot_prompt_template(args.source_lang, args.target_lang, shot["severity"], shot["error_span"], shot["error_type"], shot["original_src"], shot["original_tgt"], shot["gold_tgt"]))
    return few_shot_prompts


# InstructScore
def instructscore_prompt_template(source_lang, target_lang, error_severity, error_span, error_type, source_sent, target_sent):
    all_listings = []
    for i in range(len(error_severity)):
        error_type_i = error_type[i].lower().replace(".", "/")
        listing = f"({i+1}) There is a {error_severity[i].lower()} error of {error_type_i} at \"{error_span[i]}\"."
        all_listings.append(listing)
    single_listing = "\n".join(all_listings)
    prompt = f"{single_listing}\n{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}:"
    return prompt

def instructscore_fewshot_prompt_template(source_lang, target_lang, error_severity, error_span, error_type, source_sent, target_sent, gold_sent):
    all_listings = []
    for i in range(len(error_severity)):
        error_type_i = error_type[i].lower().replace(".", "/")
        listing = f"({i+1}) There is a {error_severity[i].lower()} error of {error_type_i} at \"{error_span[i]}\"."
        all_listings.append(listing)
    single_listing = "\n".join(all_listings)
    prompt = f"{single_listing}\n{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}: {gold_sent}"
    return prompt


def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, help="Name of the model on hugging face")
    parser.add_argument("--model_name", type=str, help="(Nick)name of the model in directory")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str)

    parser.add_argument("--pool_data_path", type=str)
    parser.add_argument("--few_shot_type", type=str)
    parser.add_argument("--num_few_shots", type=int)

    parser.add_argument("--source_lang", type=str)
    parser.add_argument("--target_lang", type=str)

    args = parser.parse_args()
    hf_model_name = args.model_name_hf

    # =========================================== Load Model ===========================================
    dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "auto": "auto",
    }['auto']
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.cache_dir)
    pipe = pipeline(
        model=hf_model_name,
        device_map="auto",
        torch_dtype=dtype,
        min_new_tokens=50,
        max_new_tokens=4096,
        tokenizer=tokenizer,
        model_kwargs={"cache_dir": args.cache_dir, "temperature": 0.0, "do_sample": False}
    )

    # =========================================== Load Dataset ===========================================
    instructscore_instruction = f"Improve the translation from {args.source_lang} to {args.target_lang} based on the identified errors without any explanation."
    
    generations = []
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                prompts = []
                source_sent = line["original_src"]
                target_sent = line["original_tgt"]
                if args.information_type == "prompting":
                    target_sent = target_sent.replace("<v>", "")
                    target_sent = target_sent.replace("</v>", "")
                error_span = line["error_span"]
                error_severity = line["severity"]
                error_type = line["error_type"]

    # ====================================== Few shot Construction ========================================
                prompts.append(instructscore_instruction)
                if args.num_few_shots > 0:
                    few_shots = find_few_shots(args, args.pool_data_path, args.few_shot_type, source_sent, target_sent, error_severity, error_span, error_type, args.num_few_shots)
                    for shot in few_shots:
                        prompts.append(shot)
                query_prompt = instructscore_prompt_template(args.source_lang, args.target_lang, error_severity, error_span, error_type, source_sent, target_sent)
                prompts.append(query_prompt)

                whole_input = "\n\n".join([item for item in prompts])
                print(whole_input)

    # =========================================== Generation =============================================
                generation = generate_text(args, pipe, tokenizer, whole_input)
                if "\n" in generation:
                    generation = generation.split("\n")[0]
                else: pass
                print(f"> {generation}")
                print("\n==================================\n")
                generations.append(generation)

                line["prompt"] = whole_input
                line["generation"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == '__main__':
    main()