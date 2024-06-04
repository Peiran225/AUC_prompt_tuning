from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerState
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit, 
    TaskType,
)
from packaging import version
from datasets import load_dataset, concatenate_datasets
import evaluate
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging


from utils_AUC import AUCLOSS
import torch.nn.functional as F
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import wandb
from trainers import AUCTrainer
from utils_AUC import p_of_positive
from transformers import set_seed

# class PrintStepCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         print(f"Current step number: {state.global_step}")

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )




def main(args):
    set_seed(args.seed)
    wandb.init(project='prompting_AUC', 
               entity='pyu123',
               name= 'loss-'+ str(args.loss) + '-lr_2-' + str(args.learning_rate_2) + '-seed-' + str(args.seed),
           config={ "model": args.model,
                    "learning_rate": args.learning_rate,
                    "dataset": args.dataset_name,
                    "epochs": 1,
                    "positive_rate": args.positive_rate,
                    "loss": args.loss
                    })
    
    model_name_or_path = args.model  
    dataset = load_dataset('sst2')
    
    positive_samples = dataset['train'].filter(lambda example: example['label'] == 1)
    negative_samples = dataset['train'].filter(lambda example: example['label'] == 0)
    num_positive_to_keep = int(args.positive_rate * len(positive_samples))
    reduced_positive_samples = positive_samples.shuffle(seed=args.seed).select(range(num_positive_to_keep))
    dataset['train'] = concatenate_datasets([negative_samples, reduced_positive_samples]).shuffle(seed=args.seed)

    positive = p_of_positive(dataset)

    

    metric = evaluate.load('roc_auc') #("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions_tensor = torch.tensor(predictions)
        predictions_tensor = F.softmax(predictions_tensor,dim=1)
        predictions_numpy = predictions_tensor.numpy()
        prediction_scores = np.array(predictions_numpy, dtype='float32')

        
        return metric.compute(prediction_scores=prediction_scores[:,1], references=labels)

    


    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id


    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence"], padding=True, truncation=True) #, max_length=None)
        return outputs

    
    tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence"],
            )
    
    

    print("finishing tokeninzing")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Checks if CUDA is available and uses it if possible, otherwise defaults to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenized_datasets = tokenized_datasets.to(device)
    # tokenized_datasets = {split: dataset.to(device) for split, dataset in tokenized_datasets.items()}


    t = tokenized_datasets['train'][0]
    
    ini_prompt = "What is the sentiment of this sentence?\nPositive, Negative."
    org_input = tokenizer(ini_prompt
                          , return_tensors='pt')
    num_virtual_tokens = len(org_input['input_ids'][0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init=PromptTuningInit.TEXT, #.TEXT/RANDOM
    num_virtual_tokens=num_virtual_tokens,
    prompt_tuning_init_text=ini_prompt,
    tokenizer_name_or_path=model_name_or_path,
)

    if args.pretrain==True:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        pretrained_model = get_peft_model(pretrained_model, peft_config)
        if model_name_or_path == "gpt2":
            pretrained_model.config.pad_token_id = tokenizer.pad_token_id
        pretraining_args = TrainingArguments(
                                output_dir="your-name/gpt2-peft-prompt-tuning",
                                learning_rate=args.pre_learning_rate, #1e-3
                                per_device_train_batch_size=args.per_device_train_batch_size,
                                per_device_eval_batch_size=args.per_device_eval_batch_size,
                                num_train_epochs=args.num_pretrain_epochs,
                                weight_decay=args.pre_weight_decay, #originally 0.01
                                seed=args.seed
                            )
        pretrainer = AUCTrainer(
            model=pretrained_model,
            args=pretraining_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            loss="entropy"
            # callbacks=[PrintStepCallback()]
        )
        pretrainer.train()
        torch.save(pretrained_model.state_dict(), '/fs/nexus-scratch/peiran/test_classification_using_AUC_maximization/model_weights.pth')
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load('/fs/nexus-scratch/peiran/test_classification_using_AUC_maximization/model_weights.pth'))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

   

    if model_name_or_path == "gpt2":
        model.config.pad_token_id = tokenizer.pad_token_id



   
   # Train 
    

    training_args = TrainingArguments(
        output_dir=args.model + args.dataset_name,
        learning_rate=args.learning_rate, #1e-3
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay, #originally 0.01
        evaluation_strategy="steps",
        eval_steps=1,
        load_best_model_at_end=True, #,
        lr_scheduler_type = "constant",
        seed=args.seed
    )

    if args.loss == "AUC":
        trainer = AUCTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        p=positive,
        lr_2=args.learning_rate_2
        # callbacks=[PrintStepCallback()]
    )

    else:
        trainer = AUCTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            loss=args.loss
            # callbacks=[PrintStepCallback()]
        )

   
    eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    print("balanced data AUC before train\n %s"% eval)

    trainer.train()
    eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    print("balanced data AUC after train\n %s"% eval)
    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--num_virtual_tokens", default=6, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--learning_rate_2", default=0.5, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--per_device_train_batch_size", default=32, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--positive_rate", default=1e-4, type=float)
    parser.add_argument("--loss", default="entropy", type=str)
    parser.add_argument("--num_pretrain_epochs", default=0.1, type=float)
    parser.add_argument("--pre_weight_decay", default=0.01, type=float)
    parser.add_argument("--pre_learning_rate", default=1e-3, type=float)
    parser.add_argument("--pretrain", default=False, type=bool)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model", default="gpt2", type=str)
    parser.add_argument("--dataset_name", default="sst2", type=str)


    args = parser.parse_args()
    # args.learning_rate_2 = 1/(args.positive_rate*(1 - args.positive_rate))/2
    

    
    print(args) 
    main(args)




