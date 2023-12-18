import jsonlines
import json
import datetime
import argparse


def make_template(args):
    templates = []
    test_sources = []
    test_targets = []
    
    with jsonlines.open(args.test_path) as file:
        for line in file.iter():
            test_sources.append(line["original_src"])
            test_targets.append(line["original_tgt"])
    
    with jsonlines.open(args.input_path) as file:
        for line in file.iter():
            source_sent = line["original_src"]
            target_sent = line["original_tgt"]
            gold_sent = line["gold_tgt"]
            error_span = line["error_span"]
            error_type = line["error_type"]
            severity = line["severity"]

            if source_sent not in test_sources and target_sent not in test_targets:
                all_listings = []
                for i in range(len(severity)):
                    listing = f"There is a {severity[i].lower()} error of {error_type[i].lower()} at \"{error_span[i]}\"."
                    all_listings.append(listing)
                single_listing = " ".join(all_listings)
                
                template = {"template": f"""### {args.source_lang}: {source_sent}\n### {args.target_lang}: {target_sent}\n### Errors: {single_listing}\n\n### Improved {args.target_lang}: {gold_sent}"""}
                templates.append(template)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for data in templates:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--dev_path", type=str)

    parser.add_argument("--source_lang", type=str)
    parser.add_argument("--target_lang", type=str)

    args = parser.parse_args()
    start_time = datetime.datetime.now()
    
    make_template(args)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")