#!/usr/bin/env python3
"""
DeepSpeed Training Wrapper for run_train_luh.py

This wrapper script enables DeepSpeed training by:
1. Adding DeepSpeed import and initialization
2. Modifying TrainingArguments to include DeepSpeed config
3. Calling the original run_train_luh.py logic

Usage:
    # Direct execution with deepspeed launcher
    deepspeed run_deepspeed_wrapper.py [hydra_args]
    
    # Or via bash script
    bash bash_scripts/run_ds.sh [num_gpus]

This eliminates the need to maintain separate DeepSpeed-specific code in run_train_luh.py
"""

import sys
import os
from pathlib import Path

# Filter out DeepSpeed's --local_rank argument before Hydra processes sys.argv
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]

# Add DeepSpeed import
import deepspeed

# Import and execute everything from the original run_train_luh.py
# but replace the main function with our DeepSpeed-enabled version
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Read the original script content
with open(current_dir / "run_train_luh.py", 'r') as f:
    original_content = f.read()

# Extract everything before the @hydra.main decorator
functions_and_imports = original_content.split("@hydra.main")[0]

# Execute all the imports and function definitions
exec(functions_and_imports, globals())

# Environment setup for Hydra config
hydra_cfg_path = os.environ.get("HYDRA_CONFIG", None)
hydra_cfg_dir = str(Path(hydra_cfg_path).parent) if hydra_cfg_path is not None else None
hydra_cfg_name = str(Path(hydra_cfg_path).name) if hydra_cfg_path is not None else None


@hydra.main(
    version_base=None,
    config_path=hydra_cfg_dir,
    config_name=hydra_cfg_name,
)
def main(config):
    # Add default DeepSpeed config if not specified
    if not hasattr(config, 'deepspeed_config') or config.deepspeed_config is None:
        config.deepspeed_config = "configs/ds_config.json"
    
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")
    log.info(f"Using DeepSpeed config: {config.deepspeed_config}")

    # setup huggingface logger
    hf_logger.handlers = []
    for h in log.handlers:
        hf_logger.addHandler(h)

    if config.report_to == "wandb":
        import wandb

        wandb_cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        config_path_hydra = [
            path["path"]
            for path in HydraConfig.get().runtime.config_sources
            if path["schema"] == "file"
        ][0]
        wandb_cfg["HYDRA_CONFIG"] = (
            Path(config_path_hydra) / HydraConfig.get().job.config_name
        )
        os.environ["WANDB_DIR"] = str(Path(output_dir))
        project = os.environ["WANDB_PROJECT"]
        wandb.init(project=project, dir=output_dir, config=wandb_cfg)
        wandb_save_directory(Path(output_dir) / ".hydra")

    hf_logger.info("Init transformers logger.")

    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    set_seed(random_seed)
    np.random.seed(random_seed)

    if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = None
    f_model_init = None
    if config.do_hyperopt:

        def model_init(trial):
            if trial is None:
                return load_model(config)

            log.info(repr(trial))
            model_params = OmegaConf.to_container(config, resolve=True)
            model_params["ue_layer"]["n_layers"] = trial["n_layers"]
            model_params["ue_layer"]["n_heads"] = trial["n_heads"]
            model_params["ue_layer"]["pos_weight"] = trial["pos_weight"]
            omega_model_params = OmegaConf.create(model_params)

            return load_model(omega_model_params)

        f_model_init = model_init

    else:
        log.info("Loading model...")
        model = load_model(config)
        log.info("Done.")
        log.info(repr(model))

    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    log.info("Done.")

    log.info("Loading dataset...")
    tokenized_data = load_data(config, tokenizer)
    log.info("Done.")
    log.info(repr(tokenized_data))

    # Modified TrainingArguments to include DeepSpeed
    train_args = TrainingArguments(
        num_train_epochs=config.training_arguments.num_train_epochs,
        per_device_train_batch_size=config.training_arguments.per_device_train_batch_size,
        per_device_eval_batch_size=config.training_arguments.per_device_train_batch_size,
        gradient_accumulation_steps=config.training_arguments.gradient_accumulation_steps,
        eval_accumulation_steps=getattr(config.training_arguments, 'eval_accumulation_steps', 4),
        learning_rate=config.training_arguments.learning_rate,
        weight_decay=config.training_arguments.weight_decay,
        max_grad_norm=config.training_arguments.max_grad_norm,
        warmup_ratio=config.training_arguments.warmup_ratio,
        lr_scheduler_type=getattr(config.training_arguments, 'lr_scheduler_type', 'linear'),
        # fp16=True,  # Had to comment for Qwen2.5-Math-1.5B, othervise it output nan logits and attentions
        # fp16_full_eval=False,
        load_best_model_at_end=True if (config.do_save_checkpoints and 
                                       getattr(config.training_arguments, 'eval_strategy', 'epoch') != 'no') else False,
        metric_for_best_model=getattr(config.training_arguments, 'metric_for_best_model', 'pr_auc'),
        eval_strategy=getattr(config.training_arguments, 'eval_strategy', 'epoch'),
        logging_strategy=getattr(config.training_arguments, 'logging_strategy', 'epoch'),
        save_strategy="epoch" if config.do_save_checkpoints else "no",
        output_dir=Path(output_dir) / "outputs",
        logging_dir=Path(output_dir) / "transformers_logs",
        report_to=config.report_to if config.report_to else None,
        include_num_input_tokens_seen=getattr(config.training_arguments, 'include_num_input_tokens_seen', True),
        gradient_checkpointing=getattr(config.training_arguments, 'gradient_checkpointing', False),
        dataloader_num_workers=getattr(config.training_arguments, 'dataloader_num_workers', 1),
        remove_unused_columns=getattr(config.training_arguments, 'remove_unused_columns', False),
        # DeepSpeed configuration
        deepspeed=config.deepspeed_config,
        bf16=getattr(config.training_arguments, 'bf16', True),  # Use bfloat16 for better numerical stability with DeepSpeed
        save_total_limit=getattr(config.training_arguments, 'save_total_limit', 3),
    )

    # Print the complete training arguments that will be used
    log.info("===============================================")
    log.info("FINAL TRAINING ARGUMENTS TO BE USED:")
    log.info("===============================================")
    for key, value in train_args.__dict__.items():
        log.info(f"  {key}: {value}")
    log.info("===============================================")

    if model.ue_head.model_type == "claim":
        def dataset_filter(inst):
            return len(inst['claims']) > 0

        tokenized_data = tokenized_data.filter(dataset_filter)
        data_collator = DataCollatorForLanguageModelingWithUncertaintyClaim(tokenizer, mlm=False)
    elif model.ue_head.model_type == "token":
        data_collator = DataCollatorForLanguageModelingWithUncertainty(tokenizer, mlm=False)
    
    callbacks = [LoggerCallback()]
    if (config.do_save_checkpoints and 
        getattr(config.training_arguments, 'eval_strategy', 'epoch') != 'no'):
        # Make early stopping patience configurable with default of 5
        early_stopping_patience = getattr(config.training_arguments, 'early_stopping_patience', 5)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    if model.ue_head.model_type == "claim":
        f_eval = lambda eval_pred_: compute_metrics_claims(tokenized_data["test"], eval_pred_)
    elif model.ue_head.model_type == "token":
        f_eval = lambda eval_pred_: compute_metrics(tokenized_data["test"], eval_pred_)

    trainer = TrainerCustom(
        model=model,
        model_init=f_model_init,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        args=train_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=f_eval,
    )

    if config.do_hyperopt:
        # This option is only for hperparameter optimization using optuna
        # For optimization with wandb, use the wandb sweep feature

        def compute_objective(metrics):
            return metrics["eval_f1"]

        def hp_space(trial):
            return {
                "training_arguments": {
                    "learning_rate": trial.suggest_categorical(
                        "learning_rate", [1e-5, 5e-5, 1e-4]
                    ),
                    "weight_decay": trial.suggest_categorical(
                        "weight_decay", [0.0, 0.01, 0.1, 0.5]
                    ),
                    "warmup_ratio": trial.suggest_categorical(
                        "warmup_ratio", [0.0, 0.1]
                    ),
                    "num_train_epochs": trial.suggest_categorical(
                        "num_train_epochs", [5, 7, 10, 15]
                    ),
                },
                "ue_layer": {
                    "n_layers": trial.suggest_categorical("n_layers", [1, 2]),
                    "n_heads": trial.suggest_categorical("n_heads", [16, 32, 64]),
                    "pos_weight": trial.suggest_categorical(
                        "pos_weight", [4.0, 6.0, 12.0]
                    ),
                },
            }

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=30,
            compute_objective=compute_objective,
        )

        log.info(f"Best metric: {repr(best_trial.objective)}")
        log.info(f"Best hyperparameters: {repr(str(best_trial.hyperparameters))}")
        with open(Path(output_dir) / "best_hyperparameters.json", "w") as f:
            json.dump(best_trial.hyperparameters, f)

    else:
        if config.do_train:
            trainer.model.orig_base_model.config.use_cache = False

            try:
                trainer.train(ignore_keys_for_eval=["logits"])
            except KeyboardInterrupt:
                log.info("Training interrupted.")
                
            log.info("Done with training.")

            if config.do_save_final_model:
                log.info("Saving model...")
                save_path = Path(output_dir) / "model"
                
                # DeepSpeed-aware saving
                if hasattr(trainer, 'deepspeed') and trainer.deepspeed:
                    # Gather parameters for saving when using DeepSpeed
                    with deepspeed.zero.GatheredParameters(trainer.model.ue_head.parameters()):
                        if trainer.is_world_process_zero():
                            trainer.model.ue_head.save(save_path)
                            log.info(f"Saved to: {save_path}")
                            if getattr(config, 'save_dir', None) is not None:
                                trainer.model.ue_head.save(Path(config.save_dir))
                                log.info(f"Saved to: {config.save_dir}.")
                            if getattr(config, 'hf_save_path', None) is not None:
                                trainer.model.ue_head.push_to_hub(config.hf_save_path)
                                log.info(f"Saved to HF: {config.hf_save_path}.")
                else:
                    trainer.model.ue_head.save(save_path)
                    log.info(f"Saved to: {save_path}")
                    if getattr(config, 'save_dir', None) is not None:
                        trainer.model.ue_head.save(Path(config.save_dir))
                        log.info(f"Saved to: {config.save_dir}.")
                    if getattr(config, 'hf_save_path', None) is not None:
                        trainer.model.ue_head.push_to_hub(config.hf_save_path)
                        log.info(f"Saved to HF: {config.hf_save_path}.")

        if config.do_eval:
            log.info("Evaluating...")
            log.info(trainer.evaluate(ignore_keys=["logits"]))
            log.info("Done with evaluation.")

        if config.do_predict:
            log.info("Predicting...")
            predictions = trainer.predict(tokenized_data["test"])
            log.info("Done with prediction.")
            
            save_dataset = Dataset.from_dict({
                "logits" : predictions[0][0], 
                "uncertainty_logits" : predictions[0][1]})
        
            save_path = Path(output_dir) / "predictions"
            log.info(f"Saving predictions to {save_path}")
            save_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()