import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import time

import torch.nn.functional as F
from numpy import argmax, stack


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

# @torch.inference_mode()
# def get_logprobs(model, tokenizer, inputs, label_ids=None, label_attn=None):
#     inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')    
    
#     logits = model(**inputs).logits
#     output_ids = inputs["input_ids"][:, 1:]
#     logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(dim=-1)
#     logprobs[inputs["attention_mask"][:, :-1] == 0] = 0
#     return logprobs.sum(dim=1).cpu()

# @torch.inference_mode()
# def predict_classification(model, tokenizer, prompts, labels):
#     probs = []
#         for label in labels:
#             inputs = [prompt.replace('[LABELS_CHOICE]', label) for prompt in prompts]
#             probs.append(get_logprobs(model, tokenizer, inputs))
#     return probs

@torch.inference_mode()
def eval_decoder_only(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    label_indices = [
        tokenizer("A").input_ids[0],
        tokenizer("B").input_ids[0],
        tokenizer("C").input_ids[0],
        tokenizer("D").input_ids[0]
    ]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        input = tokenizer(prompt, return_tensors="pt", padding = True, truncation = True, max_length=1024).to('cuda') 
        logits = model(**input).logits
        probs = torch.softmax(logits[0, -1,label_indices], dim=-1).cpu().numpy()
        pred_int = argmax(probs, axis=-1).tolist()
        pred = chr(ord('A')+pred_int)
        
        # probs = []
        # for choice in choices:
        #     prompt_input = f'{prompt} {choice}'
        #     input = tokenizer(prompt_input, return_tensors="pt", padding = True, truncation = True, max_length=1024).to('cuda') 
        #     logits = model(**input).logits
        #     output_ids = input['input_ids'][:,1:]
        #     logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(dim=-1)
        #     logprobs[input["attention_mask"][:, :-1] == 0] = 0
        #     prob = logprobs.mean(dim=1).cpu()
        #     probs.append(prob)
        # pred_int = argmax(stack(probs, axis=-1), axis=-1).tolist()      
        # pred = chr(ord('A')+pred_int[0])
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, truncation_side='left', padding_side='right')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "bloom" in args.model or "llama" in args.model.lower() or "falcon" in args.model.lower():
        print("bloom")
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_8bit=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        heads_per_gpu = len(model.encoder.block) // args.ngpu
        device_map = {
            gpu: list(
                range(
                    0 + (gpu * heads_per_gpu),
                    (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
                )
            )
            for gpu in range(args.ngpu)
        }
        model.parallelize(device_map)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        if model.config.is_encoder_decoder:
            cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        else:
            cors, acc, probs = eval_decoder_only(args, subject, model, tokenizer, dev_df, test_df)
        
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=2)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/flan-t5-small",
    )
    args = parser.parse_args()
    main(args)
