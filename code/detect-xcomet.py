import torch
import datetime
import argparse
import jsonlines
from huggingface_hub.hf_api import HfFolder
from comet import download_model, load_from_checkpoint


def mqm_xcomet(args, model):
    total_triplets = []
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as f:
            for line in f.iter():
                src = line["original_src"]
                mt = line["original_tgt"]
                mt = mt.replace("<v>", "")
                mt = mt.replace("</v>", "")
                ref = line["gold_tgt"]
                triplet = {"src": src, "mt": mt, "ref": ref}
                total_triplets.append(triplet)

        model_output = model.predict(total_triplets, batch_size=2, gpus=1)
        error_spans = model_output.metadata.error_spans

        for i in range(len(error_spans)):  
            print(error_spans[i])
            if len(error_spans[i]) != 0:
                line = {
                    "original_src": total_triplets[i]["src"],
                    "original_tgt": total_triplets[i]["mt"],
                    "gold_tgt": total_triplets[i]["ref"],
                    "severity": [item["severity"] for item in error_spans[i]],
                    "error_span": [item["text"] for item in error_spans[i]],
                    "confidence": [item["confidence"] for item in error_spans[i]],
                    "error_start_idx": [item["start"] for item in error_spans[i]],
                    "error_end_idx": [item["end"] for item in error_spans[i]],
                }
                outfile.write(line)


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, help="Name of the model on hugging face")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    
    args = parser.parse_args()
    hf_model_name = args.model_name_hf

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")

    # ====================================== Load xCOMET model ========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = download_model(hf_model_name, saving_directory=args.cache_dir)
    model = load_from_checkpoint(model_path).to(device)
    
    mqm_xcomet(args, model)