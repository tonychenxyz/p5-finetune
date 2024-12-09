# `finetune.py`

Usage: `python finetune.py --seed <seed> --no_other_data <no_other_data> --exp_num <exp_num> --n_epochs <n_epochs> --weights_preset_file <weights_preset_file> --exp_name <exp_name> --debug <debug>`

Arguments:
- `--seed`: Random seed for reproducibility (default: 0)
- `--no_other_data`: Whether to use only downstream task data without mixing other tasks (default: False)
- `--exp_num`: Experiment index number (default: 1000)
- `--n_epochs`: Number of training epochs (default: 1)
- `--weights_preset_file`: Path to preset weights file, if using predefined mixture weights (default: None)
    - This file is a numpy array of shape (number of experiments, number of categories).
- `--exp_name`: Name of the experiment for logging purposes (default: None)
- `--debug`: Enable debug mode for additional logging (default: False)


### Outputs

Run names are automatically generated

#### Local outputs
- `ckpt/{exp_name}/{run_name}`: Model checkpoints 
    - TODO: check model checkpoints can be loaded
- `metadata/{exp_name}/{run_name}.pkl`: a pickle file containing a dictionary of experiment metadata
    - Contain
        ```python
        metadata = {
            "exp_name": exp_name,
            "run_name": run_name,
            "data_mixture": data_mixture_weights,
            "data_mixture_sizes": data_mixture_sizes,
            'train_sample_indices_df': train_sample_indices_df,
            'val_sample_indices_df': val_sample_indices_df
        }
        ```


#### Wandb logging

- Eval loss history
- Metadata
    ```python
    wandb.config.update({
        "data_mixture": data_mixture_weights,
        "data_mixture_sizes": data_mixture_sizes,
        "run_name": run_name,
        "exp_name": exp_name
    })
    ```


# How to launch multiple experiments easily

Use `start_experiments.sh`


1. Modify exp_name in `start_experiments.sh` (TODO: make this a command line argument)
2. Modify weights_preset_file template in `start_experiments.sh` (TODO: make this a command line argument)
3. Run `bash start_experiments.sh --devices <device1> <device2> ...` (for example, `bash start_experiments.sh --devices 0 1 2 3` starts 4 experiments on 4 different GPUs, loading the weights_{0,1,2,3}.npy files for each run)