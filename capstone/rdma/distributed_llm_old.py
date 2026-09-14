
import os
import mmap
import struct
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HEADER_SIZE = 28
STRUCT_FMT = '7i' 

IS_VM = '/mnt/weights' in __file__

print(f"--- INITIALIZING {'VM WORKER' if IS_VM else 'HOST MASTER'} NODE ---")

if IS_VM:
    MODEL_NAME = "/mnt/weights"
else:
    MODEL_NAME = "/home/nitk/qwen_weights"

# ==========================================
# 1. LOAD MODEL & TOKENIZER
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", local_files_only=True)

# ==========================================
# 2. NETWORK SLICING (THE ELEGANT FIX)
# ==========================================
# Instead of doing a custom loop, we physically slice the model apart!
if IS_VM:
    # VM handles Layers 24-27. We keep the final Layer Norm intact.
    model.model.layers = torch.nn.ModuleList(model.model.layers[24:])
    print("[PyTorch] Sliced model for VM (Layers 24-27)")
else:
    # Host handles Layers 0-23. 
    model.model.layers = torch.nn.ModuleList(model.model.layers[:24])
    # Disable the final Layer Norm so we send RAW tensors to the VM!
    model.model.norm = torch.nn.Identity() 
    print("[PyTorch] Sliced model for Host (Layers 0-23)")

# ==========================================
# 3. CONNECT TO RDMA SHARED MEMORY
# ==========================================
print("[PyTorch] Connecting to RDMA Shared Memory...")
fd = os.open("/dev/shm/llm_buffer", os.O_RDWR)
buffer = mmap.mmap(fd, 0)

def read_header():
    return list(struct.unpack(STRUCT_FMT, buffer[:HEADER_SIZE]))

def write_header(header_list):
    buffer[:HEADER_SIZE] = struct.pack(STRUCT_FMT, *header_list)

def read_tensor(seq_len):
    tensor_size = seq_len * 3584 * 4
    raw_data = buffer[HEADER_SIZE : HEADER_SIZE + tensor_size]
    arr = np.frombuffer(raw_data, dtype=np.float32).copy()
    
    # Convert back to PyTorch and match the model's native dtype (BFloat16)
    return torch.tensor(arr).reshape(1, seq_len, 3584).to(model.dtype)

def write_tensor(tensor):
    # Convert PyTorch BFloat16 -> PyTorch Float32 -> NumPy Float32
    flat_tensor = tensor.detach().to(torch.float32).numpy().flatten()
    
    raw_data = flat_tensor.tobytes()
    buffer[HEADER_SIZE : HEADER_SIZE + len(raw_data)] = raw_data

# ==========================================
# 4. THE NODE LOGIC
# ==========================================
if IS_VM:
    print("--- WORKER NODE READY ---")
    while True:
        header = read_header()
        
        if header[0] == 1:
            seq_len = header[2]
            print(f"[PyTorch-VM] Received tensor! Sequence length: {seq_len}")
            
            hidden_states = read_tensor(seq_len)
            
            print("[PyTorch-VM] Passing through Layers 24-27...")
            # Let Hugging Face do all the hard work automatically!
            outputs = model.model(inputs_embeds=hidden_states)
            
            # The output is already normalized, just pass to LM Head
            logits = model.lm_head(outputs.last_hidden_state)
            
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            print(f"[PyTorch-VM] Predicted Token ID: {next_token_id}")
            
            header[3] = next_token_id  
            header[0] = 2              
            write_header(header)
            
        time.sleep(0.01)

else:
    print("--- MASTER NODE READY ---")
    while True:
        prompt = input("\nEnter Prompt: ")
        if not prompt: continue
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        
        print(f"[PyTorch-Host] Passing through Host Layers 0-23...")
        
        # Let Hugging Face do all the hard work automatically!
        outputs = model.model(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state
            
        print("[PyTorch-Host] Writing to Shared Memory...")
        write_tensor(hidden_states)
        
        header = read_header()
        header[2] = seq_len  
        header[0] = 1        
        write_header(header)
        
        print("[PyTorch-Host] Waiting for VM...")
        while header[0] != 2: # Waiting for the C-engine to change the flag
            time.sleep(0.001)

        # Grab the token from the designated slot
        predicted_token = header[1]
        print(f"[RESULT] The VM predicted Token: {predicted_token}")
        header[0] = 0
        write_header(header)