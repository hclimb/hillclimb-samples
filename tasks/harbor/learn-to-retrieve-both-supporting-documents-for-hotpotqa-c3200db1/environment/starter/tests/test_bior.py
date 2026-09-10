

"""Test script to verify BIOR data loader is working correctly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from transformers import AutoTokenizer
from data.bior import BioR
import numpy as np


def visualize_tokens(tokenizer, tokens, mask, title=""):
    """Print tokens with color coding based on mask (trained vs not trained)."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Decode individual tokens
    token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
    mask_list = mask.tolist() if hasattr(mask, 'tolist') else list(mask)
    
    print("\n[Token Analysis]")
    print("-" * 40)
    
    trained_tokens = []
    masked_tokens = []
    
    for i, (tok_id, m) in enumerate(zip(token_list, mask_list)):
        if tok_id == tokenizer.pad_token_id:
            continue  # Skip padding
            
        tok_str = tokenizer.decode([tok_id])
        if m > 0:
            trained_tokens.append((i, tok_id, tok_str))
        else:
            masked_tokens.append((i, tok_id, tok_str))
    
    print(f"\n🔴 MASKED (not trained on) - {len(masked_tokens)} tokens:")
    for i, tok_id, tok_str in masked_tokens[:30]:  # Show first 30
        print(f"  [{i:3d}] {tok_id:6d} -> {repr(tok_str)}")
    if len(masked_tokens) > 30:
        print(f"  ... and {len(masked_tokens) - 30} more")
    
    print(f"\n🟢 TRAINED ON - {len(trained_tokens)} tokens:")
    for i, tok_id, tok_str in trained_tokens[:30]:  # Show first 30
        print(f"  [{i:3d}] {tok_id:6d} -> {repr(tok_str)}")
    if len(trained_tokens) > 30:
        print(f"  ... and {len(trained_tokens) - 30} more")
    
    # Show full decoded text
    print(f"\n[Full Decoded Text]")
    print("-" * 40)
    
    # Masked portion (prefix)
    masked_ids = [tok_id for tok_id, m in zip(token_list, mask_list) if m == 0 and tok_id != tokenizer.pad_token_id]
    trained_ids = [tok_id for tok_id, m in zip(token_list, mask_list) if m > 0 and tok_id != tokenizer.pad_token_id]
    
    print(f"PREFIX (masked): {tokenizer.decode(masked_ids)}")
    print(f"\nTARGET (trained): {tokenizer.decode(trained_ids)}")


def test_bior_samples():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading BioR dataset...")
    dataset = BioR(
        tokenizer=tokenizer,
        bio_interval=1,
        num_qa_per_bio=2,
        bio_limit=0,      # Only 2 biographies
        qa_limit=2,       # Only 2 individuals for QA
        provide_docs=False,
        chat_template=True,
        seq_len=256,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nDataset loaded. Getting first batch...")
    
    # Get first batch
    gen = dataset.generator()
    batch_inputs, batch_masks = next(gen)
    
    # Convert to numpy for easier handling
    inputs_np = np.array(batch_inputs)
    masks_np = np.array(batch_masks)
    
    print(f"\nBatch shape: {inputs_np.shape}")
    print(f"Mask shape: {masks_np.shape}")
    
    # Visualize first few examples
    for i in range(min(4, inputs_np.shape[0])):
        visualize_tokens(
            tokenizer, 
            inputs_np[i], 
            masks_np[i], 
            title=f"Example {i+1}"
        )


def test_bior_with_docs():
    """Test with docs (memory) enabled."""
    print("\n" + "="*60)
    print("TESTING WITH DOCS (MEMORY) ENABLED")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = BioR(
        tokenizer=tokenizer,
        bio_interval=1,
        num_qa_per_bio=1,
        bio_limit=1,
        qa_limit=1,
        provide_docs=True,  # Enable docs
        chat_template=True,
        seq_len=256,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    gen = dataset.generator()
    x, masks = next(gen)
    
    # x is {"batch": inputs, "docs": docs}
    # masks is {"batch_mask": masks, "docs_mask": docs_masks}
    
    print(f"\nBatch inputs shape: {np.array(x['batch']).shape}")
    print(f"Docs shape: {np.array(x['docs']).shape}")
    
    batch_np = np.array(x['batch'])
    docs_np = np.array(x['docs'])
    batch_mask_np = np.array(masks['batch_mask'])
    docs_mask_np = np.array(masks['docs_mask'])
    
    # Show first example
    visualize_tokens(tokenizer, batch_np[0], batch_mask_np[0], "Main Input (QA/Bio)")
    visualize_tokens(tokenizer, docs_np[0], docs_mask_np[0], "Doc (Biography Memory)")


if __name__ == "__main__":
    print("="*60)
    print("BioR Dataset Test")
    print("="*60)
    
    test_bior_samples()
    #test_bior_with_docs()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
