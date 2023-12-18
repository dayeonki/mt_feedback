# Guiding Large Language Models to Post-Edit Machine Translation with Error Annotations
This is the repository for CMSC898 Research Project: **Guiding Large Language Models to Post-Edit Machine Translation with Error Annotations**.

<p align="center">
  <img src="https://github.com/dayeonki/mt_feedback/assets/68190200/dcd76164-9c88-4fe2-8a0d-17dfa348b312">
</p>


### (1) Automatic Error Annotation
- Automatically annotate error spans, error type, and severity level using [InstructScore](https://github.com/xu1998hz/SEScore3/tree/main)
```bash
python -u code/detect-instructscore.py \
  --instructscore_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE
```
- Automatically annotate error spans and severity level using [xCOMET](https://huggingface.co/Unbabel/XCOMET-XL)
```bash
python -u code/detect-xcomet.py \
  --model_name_hf Unbabel/XCOMET-XL \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --cache_dir $PATH_TO_CACHE_DIR
```

### (2) Prompting
- Prompting experiments using different granularity levels of feedback
  - Generic
  - Score-based
  - Fine-grained : MQM, InstructScore, xCOMET
```
python -u code/prompting/prompt-mqm.py \
  --model_name_hf meta-llama/Llama-2-7b \
  --model_name llama-2-7b \
  --information_type generic \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --cache_dir $PATH_TO_CACHE_DIR \
  --pool_data_path $PATH_TO_POOL_DATASET (for few-shot prompting only) \
  --few_shot_type random/error-aligned \
  --num_few_shots $NUM_FEW_SHOTS \
  --source_lang $SOURCE_LANGUAGE \
  --target_lang $TARGET_LANGUAGE
```

- Evaluation of output translations
```
python -u code/prompting/evaluate.py \
  --generic_input_path $GENERIC_OUTPUT_FILE \
  --score_input_path $SCORE_OUTPUT_FILE \
  --specific_input_path $MQM_OUTPUT_FILE \
  --instructscore_input_path $INSTRUCTSCORE_OUTPUT_FILE \
  --xcomet_input_path $XCOMET_OUTPUT_FILE \
```


### (3) Fine-tuning 
- Fine-tuning of LLaMA-2 7B or 13B model with error annotations using QLoRA
- Make MQM error annotations into instruction-following style
  - Need test and dev (validation) data to ensure no leaking between training and test instances
```
python -u code/fine-tuning/make-template.py \
  --input_path $PATH_TO_MQM_DATA \
  --output_path $PATH_TO_OUTPUT_FILE \
  --test_path $TEST_DATA_FILE \
  --dev_path $DEV_DATA_FILE \
  --source_lang $SOURCE_LANGUAGE \
  --target_lang $TARGET_LANGUAGE
```
- Randomly shuffle the error annotation instructions
```
python -u code/fine-tuning/random-shuffle.py \
  --file_path $PATH_TO_INSTRUCTIONS \
  --output_path $PATH_TO_OUTPUT_FILE
```
- Fine-tune with LLaMA-2 7B (`fine-tune-7b.py`) or 13B (`fine-tune-13b.py`)
  - After fine-tuning, it will produce (1) Trainer file, (2) Tokenizer file of the instruction fine-tuned model
```
python -u code/fine-tuning/fine-tune-7b.py \
  --hf_dataset $PATH_TO_HF_DATASET \
  --output_dir $PATH_TO_OUTPUT_FILE \
  --trainer_save_dir $PATH_TO_OUTPUT_TRAINER \
  --tokenizer_save_dir $PATH_TO_OUTPUT_TOKENIZER \
  --cache_dir $PATH_TO_CACHE_DIR
```
- Evaluation of output translations from the custom fine-tuned model
```
python -u code/fine-tuning/evaluate.py \
  --pickle_dir $OUTPUT_TRANSLATIONS_FILE \
  --test_data_path $PATH_TO_OUTPUT_FILE
```
