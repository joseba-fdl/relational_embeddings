#!/bin/bash
#SBATCH --job-name=llm-SD
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/output.log
#SBATCH --error=.slurm/error.err


MODEL=llama3instruct
TASK=trilabel
PROMPT_TYPE=sr-cot-few
OUTPUT='./'${MODEL}'_'${PROMPT_TYPE}

python3 llm_SD_SR.py --model $MODEL --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE 
    
