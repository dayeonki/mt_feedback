import pickle
import argparse
import jsonlines
from torchmetrics.text import TranslationEditRate, SacreBLEUScore
from tabulate import tabulate
from comet import download_model, load_from_checkpoint

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune")
    parser.add_argument("--pickle_dir", type=str)
    parser.add_argument("--test_data_path", type=str)
    args = parser.parse_args()
    print("Data: ", args.pickle_dir)

    sources = []
    references = []
    json_list = []
    with open(args.pickle_dir, 'rb') as handle:
        outputs = pickle.load(handle)
    
    with jsonlines.open(args.test_data_path) as f:
        i = 0
        for line in f.iter():
            source = line["original_src"]
            reference = line["gold_tgt"]
            
            sources.append(source)
            references.append([reference])

            vanilla_json = {"src": source, "mt": outputs[i], "ref": reference}
            json_list.append(vanilla_json)
            i += 1
    
    # Calculate sacreBLEU / TER / COMET DA
    bleu = SacreBLEUScore(n_gram=1, smooth=True)
    bleu_score = round(float(bleu(outputs, references)), 2)

    ter = TranslationEditRate(return_sentence_level_score=True, asian_support=True, normalize=True)
    ter_score = round(float(ter(outputs, references)[0]), 2)

    comet_da = round(float(comet_model.predict(json_list, batch_size=8, gpus=1)[1]), 2)
    
    final_cluster = [
        ["BLEU (Fine-tuning)", bleu_score],
        ["TER (Fine-tuning)", ter_score],
        ["COMET (Fine-tuning)", comet_da],
    ]
    table = tabulate(final_cluster, headers=["Metric", "Value"], tablefmt="grid")
    print(table)
    print("\n\n")