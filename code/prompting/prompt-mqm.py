import torch
import argparse
import datetime
import jsonlines
import random
import re
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


def find_few_shots(pool_data_path, few_shot_type, source_sent, target_sent, target_error_type, target_error_severity, num_few_shots):
    total_dataset = []
    with jsonlines.open(pool_data_path) as f:
        for line in f.iter():
            total_dataset.append(line)
    
    error_aligned_shots = []
    for data in total_dataset:
        if data["error_type"].lower() == target_error_type and data["severity"].lower() == target_error_severity and data["original_src"] != source_sent and data["original_tgt"] != target_sent:
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
    return few_shots


# Generic
def generic_prompt_template(source_lang, target_lang, error_type, error_severity, source_sent, target_sent):
    prompt = f"{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}:"
    return prompt

def generic_fewshot_prompt_template(source_lang, target_lang, error_type, error_severity, source_sent, target_sent, gold_sent):
    prompt = f"{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}: {gold_sent}"
    return prompt


# MQM
def mqm_prompt_template(source_lang, target_lang, error_type, error_severity, source_sent, target_sent):
    pattern = r"<v>(.*?)</v>"
    match = re.search(pattern, target_sent)
    if match:
        error_span = match.group(1)
        error_span = error_span.strip()
        prompt = f"(1) There is a {error_severity} {error_type} error at \"{error_span}\" within <v></v>.\n{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}:"
    else:
        prompt = f"(1) There is a {error_severity} {error_type} error within <v></v>.\n{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}:"
    return prompt

def mqm_fewshot_prompt_template(source_lang, target_lang, error_type, error_severity, source_sent, target_sent, gold_sent):
    pattern = r"<v>(.*?)</v>"
    match = re.search(pattern, target_sent)
    if match:
        error_span = match.group(1)
        error_span = error_span.strip()
        prompt = f"(1) There is a {error_severity} {error_type} error at \"{error_span}\" within <v></v>.\n{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}: {gold_sent}"
    else:
        prompt = f"(1) There is a {error_severity} {error_type} error within <v></v>.\n{source_lang}: {source_sent}\n{target_lang}: {target_sent}\nImproved {target_lang}: {gold_sent}"
    return prompt


def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, help="Name of the model on hugging face")
    parser.add_argument("--model_name", type=str, help="(Nick)name of the model in directory")
    parser.add_argument("--information_type", type=str, help="Type of information (generic/mqm)")
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
    generic_instruction = f"Improve the translation from {args.source_lang} to {args.target_lang} without any explanation."
    mqm_instruction = f"Improve the translation from {args.source_lang} to {args.target_lang} based on the identified errors without any explanation."

    INSTRUCTION_TO_TYPE = {
        "generic": generic_instruction,
        "score": mqm_instruction,
        "mqm": mqm_instruction
    }
    PROMPT_TO_TYPE = {
        "generic": generic_prompt_template,
        "mqm": mqm_prompt_template
    }
    FEWSHOT_TO_TYPE = {
        "generic": generic_fewshot_prompt_template,
        "mqm": mqm_fewshot_prompt_template
    }
    
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
                error_type = line["error_type"].lower()
                error_severity = line["severity"].lower()

    # ====================================== Few shot Construction ========================================
                prompts.append(INSTRUCTION_TO_TYPE[args.information_type])
                if args.num_few_shots > 0:
                    few_shots = find_few_shots(args.pool_data_path, args.few_shot_type, source_sent, target_sent, error_type, error_severity, args.num_few_shots)
                    for shot in few_shots:
                        prompts.append(FEWSHOT_TO_TYPE[args.information_type](args.source_lang, args.target_lang, error_type, error_severity, shot["original_src"], shot["original_tgt"], shot["gold_tgt"]))
                query_prompt = PROMPT_TO_TYPE[args.information_type](args.source_lang, args.target_lang, error_type, error_severity, source_sent, target_sent)
                prompts.append(query_prompt)

                whole_input = "\n\n".join([item for item in prompts])
                print(whole_input)

    # =========================================== Generation =============================================
                try:
                    generation = generate_text(args, pipe, tokenizer, whole_input)
                except:
                    pass
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