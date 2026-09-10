

import json

def calculate_mlp_params(config):
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    
    gate_proj = hidden_size * intermediate_size
    up_proj = hidden_size * intermediate_size
    down_proj = intermediate_size * hidden_size
    
    total_params = gate_proj + up_proj + down_proj
    return total_params

def calculate_memory_layer_params(model_config, mem_config):
    hidden_size = model_config['hidden_size']
    
    mem_size = mem_config['mem_size']
    num_mem_heads = mem_config['num_mem_heads']
    mem_k_dim = mem_config['mem_k_dim']
    mem_v_dim = mem_config['mem_v_dim']
    
    mem_q_proj = num_mem_heads * mem_k_dim * hidden_size
    mem_o_proj = hidden_size * num_mem_heads * mem_v_dim
    mem_k = mem_size * mem_k_dim
    mem_v = mem_size * mem_v_dim
    
    total_params = mem_q_proj + mem_o_proj + mem_k + mem_v
    return total_params

if __name__ == '__main__':
    # Based on Qwen/Qwen3-0.6B-Base
    model_config = {
        "hidden_size": 1024,
        "intermediate_size": 3072
    }
    
    with open('configs/memory_layer_config.json', 'r') as f:
        mem_config = json.load(f)

    mlp_params = calculate_mlp_params(model_config)
    mem_layer_params = calculate_memory_layer_params(model_config, mem_config)
    
    print(f"MLP Layer Parameters: {mlp_params:,}")
    print(f"Memory Layer Parameters: {mem_layer_params:,}")
