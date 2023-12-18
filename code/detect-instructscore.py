# Divide batch into severity, error span position, type
import re
import json
import jsonlines
import datetime
import argparse
from huggingface_hub.hf_api import HfFolder


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]
    

def batch_instructscore(instructscore_path, output_path):
    instructscore = read_jsonl(instructscore_path)
    jsonls = []
    with jsonlines.open(output_path, mode="w") as outfile:
        for i_line in instructscore:
            batch = i_line["batch"]
            error_type_pattern = r"Error type \d+: ([^\n]+)"
            major_minor_pattern = r"Major/minor: ([^\n]+)"
            error_location_pattern = r"Error location \d+: ([^\n]+)"

            # Finding all matches
            error_types = re.findall(error_type_pattern, batch)
            major_minors = re.findall(major_minor_pattern, batch)
            error_locations = re.findall(error_location_pattern, batch)

            for error_location in error_locations:
                error_location = error_location.replace('\"', "")

            # {"original_src": "你好", "original_tgt": "Hello to you.", "gold_tgt": "Hello,", "batch": "You are evaluating Chinese-to-English Machine translation task. The correct translation is \"Hello,\". The model generated translation is \"Hello to you.\". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed.Your Translation contains 1 error:\nError type 1: Wrong grammatical register (eg, inappropriately informal pronouns). Internal inconsistency (not related to terminology)\nMajor/minor: Minor\nError location 1: \"to you\"\nExplanation for error 1: The phrase \"to you\" is informal and should be replaced with \"you\" to maintain the correct grammatical register.", "score": "99"}
            jsonl_item = {
                "original_src": i_line["original_src"],
                "original_tgt": i_line["original_tgt"],
                "gold_tgt": i_line["gold_tgt"],
                "error_type": error_types,
                "severity": major_minors,
                "error_span": error_locations,
                "score": i_line["score"],
            }

            # Results
            jsonls.append(jsonl_item)
            outfile.write(jsonl_item)


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    parser = argparse.ArgumentParser()
    parser.add_argument("--instructscore_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")

    # ====================================== Annotation with InstructScore ========================================
    batch_instructscore(args.instructscore_path, args.output_path)
