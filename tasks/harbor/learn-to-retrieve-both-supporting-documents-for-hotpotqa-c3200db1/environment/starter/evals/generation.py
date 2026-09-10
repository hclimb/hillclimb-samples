import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import wandb
from tqdm import tqdm
from .base import Evaluator
from inference import generate

class GenerationEvaluator(Evaluator):
    def evaluate(self, model, dataset, **kwargs):
        print(f"Starting Generation Evaluation for {self.cfg.num_examples} examples...")
        
        results = []
        batch_items = []
        
        # Batch size should ideally be a multiple of the number of data-parallel shards
        num_devices = jax.device_count()
        # tp_devices might be in cfg if passed from eval.py, otherwise default to 1
        tp_devices = self.cfg.get("tp_devices", 1) 
        data_shards = max(1, num_devices // tp_devices)
        batch_size = self.cfg.get("batch_size", max(4, data_shards))
        
        dataset.batch_size = batch_size
        
        # Determine iterator based on dataset capabilities
        # TODO: generalize to just being eval_gen_data or something
        if hasattr(dataset, "eval_qa_completion_generator"):
            iterator = dataset.eval_qa_completion_generator()
        else:
            raise NotImplementedError(f"Dataset {type(dataset)} does not provide a known evaluation iterator (eval_data, bior_data, or bios_qa).")

        count = 0
       
        if batch_size % data_shards != 0:
            batch_size = ((batch_size // data_shards) + 1) * data_shards
            print(f"Adjusting batch_size to {batch_size} for sharding consistency.")

        pbar = tqdm(total=self.cfg.num_examples, desc="Generating")
        
        def run_batch(items):
            actual_num_in_batch = len(items)
            
            # Prepare batch
            prompts = []
            ground_truths = []
            
            # If we need to pad the batch to batch_size
            process_items = list(items)
            if len(process_items) < batch_size:
                padding_needed = batch_size - len(process_items)
                process_items.extend([process_items[0]] * padding_needed)

            for bi in process_items:
                prompt_text = ""
                ground_truth = ""
                
                if isinstance(bi, tuple) and len(bi) >= 2:
                    prompt_text = bi[0]
                    ground_truth = str(bi[1])
                elif isinstance(bi, dict):
                    if "question" in bi:
                        prompt_text = bi["question"]
                        ground_truth = bi.get("answer", "")
                    elif "biography" in bi:
                        full_text = bi["biography"]
                        words = full_text.split()
                        split_idx = min(32, len(words) // 2)
                        prompt_text = " ".join(words[:split_idx])
                        ground_truth = " ".join(words[split_idx:])
                    else:
                         full_text = str(bi)
                         prompt_text = full_text[:len(full_text)//2]
                         ground_truth = full_text[len(full_text)//2:]
                else:
                    prompt_text = str(bi)
                    ground_truth = ""

                # Apply prompt template if needed
                if self.cfg.get("prompt_template"):
                    final_prompt = self.cfg.prompt_template.format(text=prompt_text)
                else:
                    final_prompt = prompt_text
                
                prompts.append(final_prompt)
                ground_truths.append(ground_truth)

            # Generate batch
            batch_generated_texts = generate(
                forward=model.forward,
                init_kv=model.init_kv,
                tokenizer=model.tokenizer,
                params=model.weights,
                prompts=prompts,
                chat_template=(self.cfg.get("input_mode", "chat") == "chat"),
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_k=self.cfg.get("top_k", 20),
                top_p=self.cfg.get("top_p", 0.8),
            )

            # Collect results (only for the non-padded items)
            for i in range(actual_num_in_batch):
                results.append({
                    "prompt": prompts[i],
                    "generated": batch_generated_texts[i],
                    "ground_truth": ground_truths[i]
                })

        for item in iterator:
            if count >= self.cfg.num_examples:
                break
            
            batch_items.append(item)
            count += 1
            
            # If batch is full, or it's the last possible item
            if len(batch_items) == batch_size or count == self.cfg.num_examples:
                run_batch(batch_items)
                pbar.update(len(batch_items))
                batch_items = []
        
        # Process any remaining items if iterator ended before reaching num_examples
        if batch_items:
            run_batch(batch_items)
            pbar.update(len(batch_items))
        
        pbar.close()

        # Save results
        if self.cfg.output_file:
            try:
                from hydra.core.hydra_config import HydraConfig
                output_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                output_dir = os.getcwd()

            output_path = os.path.join(output_dir, self.cfg.output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved generations to {output_path}")
            if jax.process_index() == 0 and wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"{wandb.run.id}-eval-{self.key}-results",
                    type="evaluation_results",
                )
                artifact.add_file(output_path)
                wandb.log_artifact(artifact)

        return {"generated_count": count}
