import argparse
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, default_data_collator
from transformers import AutoModelForCausalLM
from easy_context import Qwen2ForCausalLM_RingAttn
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import math
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
)

os.environ["WANDB_DISABLED"] = "true"

apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()

def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb:
        import wandb

        wandb.login()
    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )
    accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb":{"name":args.output_dir.split("/")[-1]}})
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]
    if "Qwen2" in args.model:
        model = Qwen2ForCausalLM_RingAttn.from_pretrained(
            args.model,
            device_map=accelerator.device,
            torch_dtype=torch.bfloat16,
            rope_theta=args.rope_theta,
            _attn_implementation="flash_attention_2",
        )
        assert args.parallel_mode == "zigzag_ring_attn", "Only support zigzag ring attention for Qwen2 model"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=accelerator.device,
            torch_dtype=torch.bfloat16,
            rope_theta=args.rope_theta,
            _attn_implementation="flash_attention_2",
        )

        assert isinstance(
            model, (transformers.LlamaForCausalLM, transformers.MistralForCausalLM)
        ), "Only support llama and mistral model"
        model_type = (
            "llama" if isinstance(model, transformers.LlamaForCausalLM) else "mistral"
        )
        apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    # remove everything that is not input_ids
    to_remove = [col for col in train_dataset.column_names if col != "input_ids"]
    train_dataset = train_dataset.remove_columns(to_remove)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    print("Dataset Size:", len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=False,
        batch_size=args.batch_size,
    )
    if args.learning_rate != 1e-5:
        accelerator.print(f"Warning: You also need to modify easy_context/accelerate_configs/zero3_offload.json to change the learning rate")
    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
    )
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator) # does not really prepare it
    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    if args.resume_from_checkpoint:
        accelerator.print(
            f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        resume_step = (
            int(training_difference.replace("step_", ""))
        )
        skip = 0
        # train_loader = accelerator.skip_first_batches(train_loader, resume_step*args.gradient_accumulate_every-1)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")
        
    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True)
    for step, batch in enumerate(train_loader):
        if args.resume_from_checkpoint:
            skip += 1
            if skip <= resume_step * args.gradient_accumulate_every:
                accelerator.print(f"Skipping iter {skip}")
                continue
        input_ids = batch["input_ids"][..., : args.seq_length + 1][..., :-1]
        target_ids = batch["input_ids"][..., : args.seq_length + 1][..., 1:]
        position_ids = (
            torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
        )
        # shard the input_ids according to the world size and rank according to zig zag attention

        prepared = prepare_seq_parallel_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            accelerator.process_index,
            accelerator.num_processes,
            accelerator.device,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]

        loss_log = None
        with accelerator.accumulate(model):
            outputs = model(local_input_ids, 
                            position_ids=local_position_ids, 
                            labels=local_target_ids,
                            return_dict=True,
                            )
            loss = outputs.loss

            # logits = model(
            #     local_input_ids,
            #     position_ids=local_position_ids,
            # ).logits
            # loss = loss_func(
            #     logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
            # )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # pay attention here. When any seq parallel algo is turned on. This technically only log the very first chunk's loss
                # and what is the first chunk really depends on how do you shard the sequence
                # for zig zag attention, the first chunk contains the left most and rightmost tokens
                # so you cannot compare the (logged) loss of dist attention and zigzag ring attention.
                # loss_log = {"loss": loss.item(), "ppl": math.exp(loss.item())}

                # we now try gathered loss to verify if ring attention and dist flash attention produce the same loss
                # this may slow down the training
                completed_steps += 1
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                loss_log = {
                    "loss": gathered_loss.item(),
                    "ppl": math.exp(gathered_loss.item()),
                }
                accelerator.log(loss_log, step=completed_steps)

            optim.step()
            scheduler.step()
            optim.zero_grad()
        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            if isinstance(args.checkpointing_steps, int) and completed_steps > 0:
                if completed_steps % args.checkpointing_steps == 0:
                    state_dir = os.path.join(args.output_dir, "states")
                    state_dir = os.path.join(state_dir, f"step_{completed_steps}")
                    os.makedirs(state_dir, exist_ok=True)
                    model_weight_dir = os.path.join(args.output_dir,  f"step_{completed_steps}")
                    accelerator.save_state(state_dir)
                    accelerator.wait_for_everyone()
                    state_dict = accelerator.get_state_dict(model)
                    accelerator.unwrap_model(model).save_pretrained(
                        model_weight_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=state_dict,
                    )
        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()

        state_dict = accelerator.get_state_dict(model)

        accelerator.unwrap_model(model).save_pretrained(
            f"{args.output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--checkpointing-steps", type=int, default=100)
    args.add_argument("--learning-rate", type=float, default=1e-5)
    args.add_argument("--rope-theta", type=float, default=100000)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument(
        "--dataset",
        type=str,
        default="emozilla/pg_books-tokenized-bos-eos-chunked-65536",
    )
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    main(args.parse_args())
