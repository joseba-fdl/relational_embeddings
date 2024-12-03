from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from huggingface_hub import login
import re
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import argparse
import json
import pathlib
from typing import List, Dict
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7





with open(#CONFIG FILE, "r") as f:
    config = json.load(f)

def parse_args():
    os.environ['TRANSFORMERS_CACHE'] = #Set your cache path

    parser = argparse.ArgumentParser(
            description="Finetune a transformers model on a text classification task"
        )
    
    parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    help="Model name in config", 
    choices=["llama2", "llama2chat", "llama3", "llama3instruct", "mixtral8", "mistral7", "mixtralinstruct", "mistralinstruct"]
    )
    
    parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    required=True,
    help="Output path to dump predictions"
    )
    
    parser.add_argument(
    "--task",
    type=str,
    default=None,
    required=True,
    help="Type of task formulation",
    choices=["binary", "trilabel"]
    )
    
    parser.add_argument(
    "--prompt_type",
    type=str,
    default=None,
    required=True,
    help="Type of prompt"
    )
    
    args = parser.parse_args()
    
    return args

def load_dataset(data_path: str) -> pd.DataFrame:
    df = None
    extension = pathlib.Path(data_path).suffix
    if extension.endswith("json"):
        df = pd.read_json(data_path)
    elif extension.endswith("jsonl"):
        df = pd.read_json(data_path, lines=True)
    elif extension.endswith("tsv"):
        df = pd.read_csv(data_path, sep="\t", dtype="str")
    else:
        df = pd.read_csv(data_path, dtype="str")
    return df


def load_few_shot(data_path, config_ds, dataset_name, n):

    df = None
    extension = pathlib.Path(data_path).suffix
    if extension.endswith("tsv"):
        df = pd.read_csv(data_path, sep="\t", dtype="str")
    else:
        df = pd.read_csv(data_path, dtype="str")

    label_col=config_ds.get(dataset_name, "").get("label_col", "")
    text_col=config_ds.get(dataset_name, "").get("text_col", "")
    sampled_df = df.groupby(label_col).apply(lambda x: x.sample(n=min(len(x), n))).reset_index(drop=True)

    str_out = "Given these examples:"
    for t, l in zip(sampled_df[text_col].tolist(),sampled_df[label_col].tolist()):
        str_out +=" '"+ t+"' with '"+l.lower()+"' stance; "
    return str_out

def load_cot(data_path, config_ds, dataset_name, n, target):

    df = None
    extension = pathlib.Path(data_path).suffix
    if extension.endswith("tsv"):
        df = pd.read_csv(data_path, sep="\t", dtype="str")
    else:
        df = pd.read_csv(data_path, dtype="str")

    label_col=config_ds.get(dataset_name, "").get("label_col", "")
    text_col=config_ds.get(dataset_name, "").get("text_col", "")
    sampled_df = df.groupby(label_col).apply(lambda x: x.sample(n=min(len(x), n))).reset_index(drop=True)

    str_out = ": \n\n"
    for t, l in zip(sampled_df[text_col].tolist(),sampled_df[label_col].tolist()):
        str_out +="Tweet: "+t+"\n Stance: Lets think step by step. The stance of this tweet is "+l.lower()+" in regards to" +target+ ". \n\n"
    #print(str_out)
    return str_out


def dump_predictions(out_path: str, texts: List, gold_labels: List, predictions: List):
    with open(out_path, "w") as o:
        o.write("text\tgold_label\tprediction\n")
        for t, g, pr in zip(texts, gold_labels, predictions):
            o.write(f"{t}\t{g}\t{pr}\n")
    
def map_labels(predictions: List[str], label_mapping: Dict):
    #TODO: regex for string, not split
    predictions_clean = [pred.strip(".,") for pred in predictions.lower().split()]
    for pred in predictions_clean:
        if pred in label_mapping:
            return label_mapping[pred]
    return "neutral"


def map_labels_SR (prediction, label_mapping):
    # TODO: regex for string, not split

    x = re.findall("stance: \w+", prediction)
    pred = x[0].split()[-1]

    if pred in label_mapping:
        return label_mapping[pred]

    return "neutral"



def get_column_values(df, col_id):
        return df[col_id].tolist()



def main():
    
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    logger_path = os.path.join(args.output_dir, f"{args.prompt_type}_{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log")
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(logger_path), encoding='utf-8', level=logging.INFO)

    


    login(token=#SET YOUR TOKEN)
    model_id = config.get("models", {}).get(args.model, "")
    logger.info(f"Model used: {model_id}")
    logger.info(f"Prompt task: {args.task}")
    logger.info(f"Prompt config: {args.prompt_type}")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device in use: {device}")



    datasets_config = config.get("datasets", {})
    prompt_config = config.get("prompts", {}).get(args.task, {})




    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if args.model == "mixtralinstruct":
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    sd_datasets = ["se-abo","se-ath","se-cli","se-fem","se-hil"]+["cic-es","cic-es-r","cic-ca","cic-ca-r","sardi","vax-es","vax-eu"]

    for dataset in sd_datasets:

        assert args.task in datasets_config.get(dataset, {}).get("prompts", [])
        
        data_path =  datasets_config.get(dataset, {}).get("data_path", "")

        logger.info(f"Dataset loaded from: {data_path}")
        df = load_dataset(data_path)
        ##print(df)
        logger.info(f"Loaded samples: {len(df)}")
        texts = get_column_values(df, datasets_config.get(dataset, "").get("text_col", ""))
        target = datasets_config.get(dataset, "").get("target", "") # Target or topic

        gold_labels = [l.lower() for l in get_column_values(df, datasets_config.get(dataset, "").get("label_col", ""))]
        labels = list(set(gold_labels))

        set_seed(5)


        predictions = []
        for txt, l in zip(texts, gold_labels):

            if args.prompt_type == "sr-few-lang" or args.prompt_type == "sr-cot-few-lang":
                if dataset in ["vax-eu"]:
                    prompt_raw = prompt_config.get(args.prompt_type, {}).get("eu", "")
                elif dataset in ["cic-es","cic-es-r","vax-es"]:
                    prompt_raw = prompt_config.get(args.prompt_type, {}).get("es", "")
                elif dataset in ["cic-ca", "cic-ca-r"]:
                    prompt_raw = prompt_config.get(args.prompt_type, {}).get("ca", "")
                elif dataset in ["sardi"]:
                    prompt_raw = prompt_config.get(args.prompt_type, {}).get("it", "")

            else:
                prompt_raw = prompt_config.get(args.prompt_type, {}).get("pref", "")
                
            prompt = prompt_raw.format(target=target, text=txt)

            label_mappings = prompt_config.get(args.prompt_type, {}).get("label_mapping")
            logger.info(f"{txt}\t{l}")

            logger.info(f"Prompt: {prompt}")

            inputs = tokenizer([prompt], return_tensors="pt").to(device)


            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, return_dict_in_generate=True, output_scores=True, temperature=TEMPERATURE)
        
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logger.info(f"{outputs.sequences}\t{outputs.scores}")

            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                # | token | token string | log probability | probability
                logger.info(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score}")


            answers = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            logger.info(f"Answers: {answers}, split: {answers.split()}")


            if args.prompt_type == "sr-cot-few":
                logger.info(f"Mapped label: {map_labels_SR(answers, label_mappings)}")
                predictions.append(map_labels_SR(answers, label_mappings))
                #print(map_labels_SR(answers, label_mappings), flush=True)

            else:
                logger.info(f"Mapped label: {map_labels(answers, label_mappings)}")
                predictions.append(map_labels(answers, label_mappings))

            logger.info("Label added to predictions.")
            
                
        logger.debug(gold_labels[:5], predictions[:5], flush=True)
        assert len(gold_labels) == len(predictions)
        logger.info(f"Gold: {len(gold_labels)}, Pred: {len(predictions)}")
        
    
        predictions_path = os.path.join(args.output_dir, f"{args.prompt_type}_{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.tsv")
        

        dump_predictions(predictions_path, texts, gold_labels, predictions)
        
        logger.info(f"Predictions dumped to {predictions_path}")
        
        accuracy = accuracy_score(gold_labels, predictions, normalize=True)
        logger.info(f"Accuracy {len(gold_labels)}, {len(predictions)}: {accuracy}\n")
        logger.info(f"{dataset} Accuracy {len(gold_labels)}, {len(predictions)}: {accuracy}\n")

        f1macro = f1_score(gold_labels, predictions, average='macro')
        logger.info(f"{dataset} f1: {f1macro}\n")
        f1stance_all = f1_score(gold_labels, predictions, average=None)  # f1 for all classes
        logger.info(f"{dataset} f1 all: {f1stance_all}\n")
        f1stance = (f1stance_all[0] + f1stance_all[1]) / 2  # f1 avg for Fav & Aga
        logger.info(f"{dataset} f1 stance: {f1stance}\n")

        print(f"{dataset} f1 stance: {f1stance}", flush=True)


if __name__ == "__main__":
    main()
