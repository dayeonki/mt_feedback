import jsonlines
import argparse
import torch
from torchmetrics.text import TranslationEditRate, SacreBLEUScore
from comet import download_model, load_from_checkpoint
from tabulate import tabulate
from scipy import stats

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)


def inference(generic_input_path, score_input_path, specific_input_path, instructscore_input_path, xcomet_input_path):
    generic_targets = []
    score_targets = []
    specific_targets = []
    instructscore_targets = []
    xcomet_targets = []
    
    original_targets = []
    original_sources = []
    
    generic_references_list = []
    score_references_list = []
    specific_references_list = []
    instructscore_references_list = []
    xcomet_references_list = []

    original_json_list = []
    generic_json_list = []
    score_json_list = []
    specific_json_list = []
    instructscore_json_list = []
    xcomet_json_list = []

    # (1) Generic
    with jsonlines.open(generic_input_path) as f:
        for line in f.iter():
            original_source = line["original_src"]
            original_target = line["original_tgt"]
            generic = line["generation"]
            reference = line["gold_tgt"]

            # Replace special tokens if any
            generic = generic.strip()
            generic = generic.replace("<unk>", "")
            generic = generic.replace("<pad>", "")
            generic = generic.replace("[PAD]", "")
            generic = generic.replace("[UNK]", "")
            generic = generic.replace("<v>", "")
            generic = generic.replace("</v>", "")
            if "\n" in generic:
                generic = generic.split("\n")[0]
            else: pass

            original_sources.append(original_source)
            original_targets.append(original_target)
            generic_references_list.append([reference])
            generic_targets.append(generic)

            original_json = {"src": original_source, "mt": original_target, "ref": reference}
            generic_json = {"src": original_source, "mt": generic, "ref": reference}
            original_json_list.append(original_json)
            generic_json_list.append(generic_json)

    # (2) Score
    with jsonlines.open(score_input_path) as f:
        for line in f.iter():
            original_source = line["original_src"]
            score = line["generation"]
            reference = line["gold_tgt"]

            score = score.strip()
            score = score.replace("<unk>", "")
            score = score.replace("<pad>", "")
            score = score.replace("[PAD]", "")
            score = score.replace("[UNK]", "")
            if "\n" in score:
                score = score.split("\n")[0]
            else: pass
            
            score_targets.append(score)
            score_references_list.append([reference])

            score_json = {"src": original_source, "mt": score, "ref": reference}
            score_json_list.append(score_json)
    
    # (3) Specific
    with jsonlines.open(specific_input_path) as f:
        for line in f.iter():
            original_source = line["original_src"]
            specific = line["generation"]
            reference = line["gold_tgt"]

            specific = specific.strip()
            specific = specific.replace("<unk>", "")
            specific = specific.replace("<pad>", "")
            specific = specific.replace("[PAD]", "")
            specific = specific.replace("[UNK]", "")
            if "\n" in specific:
                specific = specific.split("\n")[0]
            else: pass
            
            specific_targets.append(specific)
            specific_references_list.append([reference])

            specific_json = {"src": original_source, "mt": specific, "ref": reference}
            specific_json_list.append(specific_json)
        
    # (3) Instructscore
    with jsonlines.open(instructscore_input_path) as f:
        for line in f.iter():
            original_source = line["original_src"]
            instruct = line["generation"]
            reference = line["gold_tgt"]

            instruct = instruct.strip()
            instruct = instruct.replace("<unk>", "")
            instruct = instruct.replace("<pad>", "")
            instruct = instruct.replace("[PAD]", "")
            instruct = instruct.replace("[UNK]", "")
            if "\n" in instruct:
                instruct = instruct.split("\n")[0]
            else: pass
            
            instructscore_targets.append(instruct)
            instructscore_references_list.append([reference])

            instructscore_json = {"src": original_source, "mt": instruct, "ref": reference}
            instructscore_json_list.append(instructscore_json)
    
    # (4) xCOMET
    with jsonlines.open(xcomet_input_path) as f:
        for line in f.iter():
            original_source = line["original_src"]
            xcomet = line["generation"]
            reference = line["gold_tgt"]

            xcomet = xcomet.strip()
            xcomet = xcomet.replace("<unk>", "")
            xcomet = xcomet.replace("<pad>", "")
            xcomet = xcomet.replace("[PAD]", "")
            xcomet = xcomet.replace("[UNK]", "")
            if "\n" in xcomet:
                xcomet = xcomet.split("\n")[0]
            else: pass
            
            xcomet_targets.append(xcomet)
            xcomet_references_list.append([reference])

            xcomet_json = {"src": original_source, "mt": xcomet, "ref": reference}
            xcomet_json_list.append(xcomet_json)


    # Calculate sacreBLEU / TER / COMET DA
    bleu = SacreBLEUScore(n_gram=1, smooth=True)
    bleu_1 = round(float(bleu(original_targets, generic_references_list)), 2)
    bleu_2 = round(float(bleu(generic_targets, generic_references_list)), 2)
    bleu_3 = round(float(bleu(score_targets, score_references_list)), 2)
    bleu_4 = round(float(bleu(specific_targets, specific_references_list)), 2)
    bleu_5 = round(float(bleu(instructscore_targets, instructscore_references_list)), 2)
    bleu_6 = round(float(bleu(xcomet_targets, xcomet_references_list)), 2)

    ter = TranslationEditRate(return_sentence_level_score=True, asian_support=True, normalize=True)
    ter_1 = round(float(ter(original_targets, generic_references_list)[0]), 2)
    ter_2 = round(float(ter(generic_targets, generic_references_list)[0]), 2)
    ter_3 = round(float(ter(score_targets, score_references_list)[0]), 2)
    ter_4 = round(float(ter(specific_targets, specific_references_list)[0]), 2)
    ter_5 = round(float(ter(instructscore_targets, instructscore_references_list)[0]), 2)
    ter_6 = round(float(ter(xcomet_targets, xcomet_references_list)[0]), 2)

    comet_da_1 = round(float(comet_model.predict(original_json_list, batch_size=8, gpus=1)[1]), 2)
    comet_da_2 = round(float(comet_model.predict(generic_json_list, batch_size=8, gpus=1)[1]), 2)
    comet_da_3 = round(float(comet_model.predict(score_json_list, batch_size=8, gpus=1)[1]), 2)
    comet_da_4 = round(float(comet_model.predict(specific_json_list, batch_size=8, gpus=1)[1]), 2)
    comet_da_5 = round(float(comet_model.predict(instructscore_json_list, batch_size=8, gpus=1)[1]), 2)
    comet_da_6 = round(float(comet_model.predict(xcomet_json_list, batch_size=8, gpus=1)[1]), 2)


    # Calculate p-values
    bleu_original = [bleu([target], [ref]) for target, ref in zip(original_targets, generic_references_list)]
    bleu_generic = [bleu([target], [ref]) for target, ref in zip(generic_targets, generic_references_list)]   
    bleu_score = [bleu([target], [ref]) for target, ref in zip(score_targets, score_references_list)]   
    bleu_specific = [bleu([target], [ref]) for target, ref in zip(specific_targets, specific_references_list)]   
    bleu_instructscore = [bleu([target], [ref]) for target, ref in zip(instructscore_targets, instructscore_references_list)]  
    bleu_xcomet = [bleu([target], [ref]) for target, ref in zip(xcomet_targets, xcomet_references_list)]   
    bleu_t_g, blue_p_g = stats.ttest_rel(bleu_original, bleu_generic)
    bleu_t_s, blue_p_s = stats.ttest_rel(bleu_original, bleu_score)
    bleu_t_m, blue_p_m = stats.ttest_rel(bleu_original, bleu_specific)
    bleu_t_i, blue_p_i = stats.ttest_rel(bleu_original, bleu_instructscore)
    bleu_t_x, blue_p_x = stats.ttest_rel(bleu_original, bleu_xcomet)

    ter_original = [ter([target], [ref])[0] for target, ref in zip(original_targets, generic_references_list)]
    ter_generic = [ter([target], [ref])[0] for target, ref in zip(generic_targets, generic_references_list)] 
    ter_score = [ter([target], [ref])[0] for target, ref in zip(score_targets, generic_references_list)]
    ter_specific = [ter([target], [ref])[0] for target, ref in zip(specific_targets, generic_references_list)]   
    ter_instructscore = [ter([target], [ref])[0] for target, ref in zip(instructscore_targets, generic_references_list)]  
    ter_xcomet = [ter([target], [ref])[0] for target, ref in zip(xcomet_targets, generic_references_list)]
    ter_t_g, ter_p_g = stats.ttest_rel(ter_original, ter_generic)
    ter_t_s, ter_p_s = stats.ttest_rel(ter_original, ter_score)
    ter_t_m, ter_p_m = stats.ttest_rel(ter_original, ter_specific)
    ter_t_i, ter_p_i = stats.ttest_rel(ter_original, ter_instructscore)
    ter_t_x, ter_p_x = stats.ttest_rel(ter_original, ter_xcomet)

    comet_original = comet_model.predict(original_json_list, batch_size=8, gpus=1)[0]
    comet_generic = comet_model.predict(generic_json_list, batch_size=8, gpus=1)[0]
    comet_score = comet_model.predict(score_json_list, batch_size=8, gpus=1)[0]
    comet_specific = comet_model.predict(specific_json_list, batch_size=8, gpus=1)[0]
    comet_instructscore = comet_model.predict(instructscore_json_list, batch_size=8, gpus=1)[0]
    comet_xcomet = comet_model.predict(xcomet_json_list, batch_size=8, gpus=1)[0]

    # Calculating p-values using paired t-test for COMET scores
    comet_t_g, comet_p_g = stats.ttest_rel(comet_original, comet_generic)
    comet_t_s, comet_p_s = stats.ttest_rel(comet_original, comet_score)
    comet_t_m, comet_p_m = stats.ttest_rel(comet_original, comet_specific)
    comet_t_i, comet_p_i = stats.ttest_rel(comet_original, comet_instructscore)
    comet_t_x, comet_p_x = stats.ttest_rel(comet_original, comet_xcomet)

    # Visualize by tabulate
    final_cluster = [
        ["BLEU (Original)", bleu_1, "-", "-"],
        ["BLEU (Generic)", bleu_2, bleu_t_g, blue_p_g],
        ["BLEU (Score)", bleu_3, bleu_t_s, blue_p_s],
        ["BLEU (MQM)", bleu_4, bleu_t_m, blue_p_m],
        ["BLEU (InstructScore)", bleu_5, bleu_t_i, blue_p_i],
        ["BLEU (xCOMET)", bleu_6, bleu_t_x, blue_p_x],

        ["TER (Original)", ter_1, "-", "-"],
        ["TER (Generic)", ter_2, ter_t_g, ter_p_g],
        ["TER (Score)", ter_3, ter_t_s, ter_p_s],
        ["TER (MQM)", ter_4, ter_t_m, ter_p_m],
        ["TER (InstructScore)", ter_5, ter_t_i, ter_p_i],
        ["TER (xCOMET)", ter_6, ter_t_x, ter_p_x],

        ["COMET (Original)", comet_da_1, "-", "-"],
        ["COMET (Generic)", comet_da_2, comet_t_g, comet_p_g],
        ["COMET (Score)", comet_da_3, comet_t_s, comet_p_s],
        ["COMET (MQM)", comet_da_4, comet_t_m, comet_p_m],
        ["COMET (InstructScore)", comet_da_5, comet_t_i, comet_p_i],
        ["COMET (xCOMET)", comet_da_6, comet_t_x, comet_p_x],
    ]
    table = tabulate(final_cluster, headers=["Metric", "t-statistics", "p-value"], tablefmt="grid")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--generic_input_path", type=str)
    parser.add_argument("--score_input_path", type=str)
    parser.add_argument("--specific_input_path", type=str)
    parser.add_argument("--instructscore_input_path", type=str)
    parser.add_argument("--xcomet_input_path", type=str)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("PATH: ", args.generic_input_path)
    print("DEVICE: ", device)
    
    inference(args.generic_input_path, args.score_input_path, args.specific_input_path, args.instructscore_input_path, args.xcomet_input_path)
