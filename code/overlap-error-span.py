import re
import json
import argparse
from scipy.stats import kendalltau


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# Correlation of scores
def score_overlap(args):
    instructscore = read_jsonl(args.instructscore_path)[:200]
    mqm = read_jsonl(args.mqm_path)

    instructscore_scores = []
    mqm_scores =[]
    for i_line in instructscore:
        matching_entry = next((m_line for m_line in mqm if m_line["original_src"] == i_line["original_src"]), None)

        if matching_entry:
            mqm_score = matching_entry["score"]
            instructscore_score = i_line["score"]
            instructscore_scores.append(instructscore_score)
            mqm_scores.append(mqm_score)
    assert len(instructscore_scores) == len(mqm_scores)
    kendall_corr, kendall_p = kendalltau(instructscore_scores, mqm_scores)
    print("Kendall's Tau: ", kendall_corr, f" (p-value: {kendall_p})")



# Overlap of error spans
def span_overlap_mqm_xcomet(args):
    mqm = read_jsonl(args.mqm_path)
    xcomet = read_jsonl(args.xcomet_path)[:200]

    comparison_results = []
    matching_results = []
    pattern = r"<v>(.*?)</v>"

    for x_line in xcomet:
        matching_entry = next((m_line for m_line in mqm if m_line["original_src"] == x_line["original_src"]), None)
        
        if matching_entry:
            target_sent = matching_entry["original_tgt"]
            match = re.search(pattern, target_sent)
            if match:
                mqm_error_span = match.group(1)
                mqm_error_span = mqm_error_span.strip()
            else: pass
            print("xCOMET ERROR SPAN: ", x_line["error_span"])
            print("MQM ERROR SPAN: ", mqm_error_span)
            is_equal = mqm_error_span in x_line["error_span"]
            print("EQUAL?: ", is_equal)
            comparison = {
                "xcomet_span": x_line["error_span"],
                "mqm_span": mqm_error_span,
            }
            comparison_results.append(comparison)
            
            if is_equal:
                matching_results.append(comparison)
            print("\n============================\n")

    print("\nTotal length: ", len(comparison_results))
    print("Equal length: ", len(matching_results))
    print("Overlap ratio: ", len(matching_results) / len(comparison_results))


# Overlap of error spans
def span_overlap_mqm_instruct(args):
    mqm = read_jsonl(args.mqm_path)
    instructscore = read_jsonl(args.instructscore_path)[:200]

    comparison_results = []
    matching_results = []
    pattern = r"<v>(.*?)</v>"

    for in_line in instructscore:
        matching_entry = next((m_line for m_line in mqm if m_line["original_src"] == in_line["original_src"]), None)
        
        if matching_entry:
            target_sent = matching_entry["original_tgt"]
            match = re.search(pattern, target_sent)
            if match:
                mqm_error_span = match.group(1)
                mqm_error_span = mqm_error_span.strip()
            else: pass
            print("InstructScore ERROR SPAN: ", in_line["error_span"])
            print("MQM ERROR SPAN: ", mqm_error_span)
            is_equal = mqm_error_span in in_line["error_span"]
            print("EQUAL?: ", is_equal)
            comparison = {
                "instructscore_span": in_line["error_span"],
                "mqm_span": mqm_error_span,
            }
            comparison_results.append(comparison)
            
            if is_equal:
                matching_results.append(comparison)
            print("\n============================\n")

    print("\nTotal length: ", len(comparison_results))
    print("Equal length: ", len(matching_results))
    print("Overlap ratio: ", len(matching_results) / len(comparison_results))


# Overlap of error spans
def span_overlap_xcomet_instruct(args):
    instructscore = read_jsonl(args.instructscore_path)
    xcomet = read_jsonl(args.xcomet_path)[:200]

    comparison_results = []
    matching_results = []

    for x_line in xcomet:
        matching_entry = next((in_line for in_line in instructscore if in_line["original_src"] == x_line["original_src"]), None)
        
        if matching_entry:
            print("xCOMET ERROR SPAN: ", x_line["error_span"])
            print("InstructScore ERROR SPAN: ", matching_entry["error_span"])
            common_elements = set(matching_entry["error_span"]) & set(x_line["error_span"])
            is_equal = len(common_elements) > 0
            print("EQUAL?: ", is_equal)
            comparison = {
                "xcomet_span": x_line["error_span"],
                "instructscore_span": matching_entry["error_span"],
            }
            comparison_results.append(comparison)
            
            if is_equal:
                matching_results.append(comparison)
            print("\n============================\n")

    print("\nTotal length: ", len(comparison_results))
    print("Equal length: ", len(matching_results))
    print("Overlap ratio: ", len(matching_results) / len(comparison_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--mqm_path", type=str)
    parser.add_argument("--xcomet_path", type=str)
    parser.add_argument("--instructscore_path", type=str)
    args = parser.parse_args()

    span_overlap_xcomet_instruct(args)