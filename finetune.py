


import torch
import random
from P5.src.pretrain_data import P5_Amazon_Dataset, get_loader
from transformers import T5TokenizerFast as P5Tokenizer
from all_amazon_templates import all_tasks as task_templates
import pandas as pd
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--no_other_data', type=bool, default=False, help='Experiment index')
parser.add_argument('--exp_num', type=int, default=1000, help='Experiment index')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--weights_preset_file', type=str, default=None, help='Weights preset file')
parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

args = parser.parse_args()
if args.exp_name is None:
    args.exp_name = "finetune-mixture-untitled"
downstream = 'traditional'
DEBUG = args.debug

print(f"Starting experiment with seed {args.seed}")

data_mixture_seed = args.seed
no_other_data = args.no_other_data
exp_num = args.exp_num
n_epochs = args.n_epochs
exp_name = args.exp_name

print(f"Starting experiment with config: {args}\n")

if args.weights_preset_file is not None:
    all_weights = np.load(args.weights_preset_file)
    exp_num = len(all_weights)
    print(f"Loaded {exp_num} weights from {args.weights_preset_file}")
from tqdm import tqdm
def sample(datasets, num_tokens, subtask_names, token_count_df, shuffle=False):
    
    sampled_dataset = []
    sample_indices_df = {'subtask_name': [], 'idx': [], 'num_tokens': []}
    target_num_tokens = num_tokens

    task_df = token_count_df[token_count_df['task'].isin(subtask_names)]
    if shuffle:
        task_df = task_df.sample(frac=1).reset_index(drop=True)
    else:
        task_df = task_df.sort_values(by='entry_id', ascending=True).reset_index(drop=True)

    cum_sum_tokens = task_df['num_tokens'].cumsum()
    larger_than_target = cum_sum_tokens >= target_num_tokens
    first_larger_than_target = larger_than_target.idxmax()
    # print(task_df, cum_sum_tokens)
    print(cum_sum_tokens.iloc[first_larger_than_target], target_num_tokens, task_df.iloc[:first_larger_than_target+1]['num_tokens'].sum())

    for idx in tqdm(range(first_larger_than_target+1)):
        sampled_dataset.append([{'role': 'user', 'content': datasets[task_df.iloc[idx]['task']][task_df.iloc[idx]['entry_id']][0]}, {'role': 'assistant', 'content': datasets[task_df.iloc[idx]['task']][task_df.iloc[idx]['entry_id']][1]}])
        sample_indices_df['subtask_name'].append(task_df.iloc[idx]['task'])
        sample_indices_df['idx'].append(task_df.iloc[idx]['entry_id'])
        sample_indices_df['num_tokens'].append(task_df.iloc[idx]['num_tokens'])
    print(pd.DataFrame(sample_indices_df)['num_tokens'].sum())
    return sampled_dataset, sample_indices_df

token_counts_df = pd.read_csv('token_counts.csv')
token_counts_df['task_name'] = token_counts_df['task'].apply(lambda x: x.split('_')[0])
category_num_tokens = token_counts_df.groupby('task_name')['num_tokens'].sum()


class DataArgs:
    backbone = 't5-base'
    max_text_length = 512
    do_lower_case = True
    gen_max_length = 64

data_args = DataArgs()

# Define task list and sample numbers for Amazon dataset

import multiprocessing as mp
from functools import partial

sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5)}



task_list = {
    'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
    'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
    'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
    'review': ['4-1', '4-2', '4-3'],
    'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
}


if DEBUG:
    task_list = {
        'rating': ['1-1'],
        'sequential': ['2-1'],
        'explanation': ['3-1'],
        'review': ['4-1'],
        'traditional': ['5-1'],
    }

def create_dataset(task_name, sub_task_name):
    sub_task_dict = {task_name: [sub_task_name]}
    return (
        f"{task_name}_{sub_task_name}",
        P5_Amazon_Dataset(
            task_templates,
            sub_task_dict,
            None,
            data_args,
            sample_numbers,
            mode='train',
            split='beauty',
        )
    )

# Create a list of all task/subtask combinations

task_combinations = [
    (task_name, sub_task_name)
    for task_name in task_list
    for sub_task_name in task_list[task_name]
]

# Initialize multiprocessing pool
print(f"Creating dataset with {mp.cpu_count()} processes")  
with mp.Pool(processes=mp.cpu_count()) as pool:
    # Map the dataset creation function across all combinations
    results = pool.starmap(create_dataset, task_combinations)

# Convert results to dictionary
datasets = dict(results)


task_names = list(task_list.keys())



for exp_idx in range(exp_num):

    random.seed(data_mixture_seed)


    if no_other_data:

        data_mixture_weights = {
            'rating': 0,
            'sequential': 0,
            'explanation': 0,
            'review': 0,
        }
    elif args.weights_preset_file is not None:
        data_mixture_weights = {
            'rating': all_weights[exp_idx][0],
            'sequential': all_weights[exp_idx][1],
            'explanation': all_weights[exp_idx][2],
            'review': all_weights[exp_idx][3],
        }
    else:
        data_mixture_weights = {
            'rating': random.randint(1, 10),
            'sequential': random.randint(1, 10),
            'explanation': random.randint(1, 10),
            'review': random.randint(1, 10),
        }

    print(f"Data mixture weights: {data_mixture_weights}")



    str_weights = "_".join([f"{task_name}_{category_num_tokens[task_name]}" for task_name in task_names])
    run_name = f"{args.exp_name}_{data_mixture_seed}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{str_weights}"




    add_additional_data = True

    for task_name in data_mixture_weights.keys():
        add_additional_data = add_additional_data and data_mixture_weights[task_name] > 0

    if add_additional_data:
        
        if DEBUG:
            total_other_data_size = 100
        else:
            total_other_data_size = min(category_num_tokens[task_name] for task_name in data_mixture_weights.keys())
        weight_sum = sum(data_mixture_weights.values())
        for task_name in data_mixture_weights.keys():
            data_mixture_weights[task_name] = data_mixture_weights[task_name] / float(weight_sum)
        
        data_mixture_sizes = {task_name: int(data_mixture_weights[task_name] * total_other_data_size) for task_name in data_mixture_weights.keys()}

    else:
        total_other_data_size = min(category_num_tokens[task_name] for task_name in data_mixture_weights.keys())
        data_mixture_sizes = {}


    

    all_downstream_subtasks = [f"{downstream}_{subtask}" for subtask in task_list[downstream]]
    
    

    # Split the downstream_task dataset into validation and train sets
    import random

    # Get the full dataset for the downstream task
    num_train_tokens = total_other_data_size // 4
    num_val_tokens = min(4096 * 512,int(total_other_data_size))

    print(f"Adding downstream data")
    train_dataset, train_sample_indices_df = sample(datasets, num_train_tokens, all_downstream_subtasks, token_counts_df, shuffle=False)
    train_dataset = train_dataset * 4
    for key in train_sample_indices_df.keys():
        train_sample_indices_df[key] = train_sample_indices_df[key] * 4

    print(f"Adding downstream validation data")
    val_dataset, val_sample_indices_df = sample(datasets, num_val_tokens, all_downstream_subtasks, token_counts_df, shuffle=False)

    for task_name in data_mixture_sizes.keys():
        if task_name != downstream:
            print(f"Adding {task_name} data")
            all_other_subtasks = [f"{task_name}_{subtask}" for subtask in task_list[task_name]]
            other_dataset, other_sample_indices_df = sample(datasets, data_mixture_sizes[task_name], all_other_subtasks, token_counts_df, shuffle=True)
        
            train_dataset.extend(other_dataset)
            train_sample_indices_df['subtask_name'].extend(other_sample_indices_df['subtask_name'])
            train_sample_indices_df['idx'].extend(other_sample_indices_df['idx'])
            train_sample_indices_df['num_tokens'].extend(other_sample_indices_df['num_tokens'])


    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    from datasets import Dataset
    train_dataset = Dataset.from_dict({"messages": train_dataset})
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = Dataset.from_dict({"messages": val_dataset})

    import pickle
    metadata = {
        "exp_name": exp_name,
        "run_name": run_name,
        "data_mixture": data_mixture_weights,
        "data_mixture_sizes": data_mixture_sizes,
        'train_sample_indices_df': train_sample_indices_df,
        'val_sample_indices_df': val_sample_indices_df
    }

    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    if not os.path.exists(f"ckpt/{exp_name}"):
        os.makedirs(f"ckpt/{exp_name}")
    if not os.path.exists(f"ckpt/{exp_name}/{run_name}"):
        os.makedirs(f"ckpt/{exp_name}/{run_name}")

    if not os.path.exists(f"metadata"):
        os.makedirs(f"metadata")
    if not os.path.exists(f"metadata/{exp_name}"):
        os.makedirs(f"metadata/{exp_name}")
    with open(f"metadata/{exp_name}/{run_name}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/shared/share_mala/hc3295/new_cache")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",cache_dir="/shared/share_mala/hc3295/new_cache",device_map="auto")

    import wandb
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    wandb.init(project=exp_name, name=run_name)

    # Log data mixture as metadata
    wandb.config.update({
        "data_mixture": data_mixture_weights,
        "data_mixture_sizes": data_mixture_sizes,
        "run_name": run_name,
        "exp_name": exp_name
    })

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/shared/share_mala/hc3295/new_cache")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",cache_dir="/shared/share_mala/hc3295/new_cache",device_map="auto")

    training_args = SFTConfig(
        eval_steps=500,
        report_to="wandb",
        num_train_epochs = n_epochs,
        fp16=True,
        logging_steps=500,
        auto_find_batch_size=True,
        evaluation_strategy="steps",
        do_eval=True,
        output_dir=f"ckpt/{exp_name}/{run_name}",
    )

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   
    )

    trainer.train()

    wandb.finish()