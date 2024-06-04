#!/bin/bash
#SBATCH --job-name=AUC  # Specify a name for your job
#SBATCH --output=outputs/AUC_trial_minimax_pretrain.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/errors_AUC_minimax_pretrain.log # AUC_p_tuning_main_errors.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=32G                  # Memory per node (4GB in this example)

#SBATCH --qos huge-long
#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng

LEARNING_RATE=5e-4 # working for AUC maximization: 5e-4
POSITIVE_RATE=1e-4 # working for AUC maximization: 1e-4
LEARNING_RATE_2=0.5 # working for AUC maximization: 0.05
LOSS="AUC"
WEIGHT_DECAY=0 # working for AUC maximization: 0
# PRETRAIN=True # working for AUC maximization: False
# NUM_PRETRAIN_EPOCHS=0.1 # working for AUC maximization: 0
NUM_TRAIN_EPOCHS=1 
SEED=42

cd /fs/nexus-scratch/peiran/Prompt_tuning_AUC

# python3 main_AUC_trainer.py

for LEARNING_RATE_2 in  0.5 0.05 
# LEARNIN_RATE woking for AUC: 5e-4 
# LEARNING_RATE_2 working for AUC: 0.05
# WEIGHT DECAY working for AUC: 0
# pretrain 0.2 0.4 0.6 0.8 epoch acc increase when pretrain more but no better than no pretrain in the case where p=1e-4 
do
    python3 main_AUC_trainer.py --learning_rate=$LEARNING_RATE --learning_rate_2=$LEARNING_RATE_2 \
                                --num_train_epochs=$NUM_TRAIN_EPOCHS \
                                --positive_rate=$POSITIVE_RATE --weight_decay=$WEIGHT_DECAY \
                                --loss=$LOSS \
                                --seed=$SEED
                                
done

# baseline of the entropy loss
# python3 main_AUC_trainer.py
# Deactivate the environment (if you want to)
# conda deactivate

# Your job is done!

# lower lr plot the training loss, see the curve, 
# plot: wandb = weights and bias 
# training loss, each batch 
# pretrain the discraminator
