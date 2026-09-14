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
IS_VM = '/mnt/weights' in __file__

print(f"--- INITIALIZING {'VM WORKER' if IS_VM else 'HOST MASTER'} NODE ---")

if IS_VM:
    MODEL_NAME = "/mnt/weights"
else:
    MODEL_NAME = "/home/nitk/qwen_weights"

HEADER_SIZE = 28
STRUCT_FMT = '7i' 
MAX_NEW_TOKENS = 100 # How long the AI is allowed to talk

# ==========================================
# 1. LOAD MODEL & TOKENIZER
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", local_files_only=True)

# ==========================================
# 2. NETWORK SLICING
# ==========================================
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
    return torch.tensor(arr).reshape(1, seq_len, 3584).to(model.dtype)

def write_tensor(tensor):
    flat_tensor = tensor.detach().to(torch.float32).numpy().flatten()
    raw_data = flat_tensor.tobytes()
    buffer[HEADER_SIZE : HEADER_SIZE + len(raw_data)] = raw_data

# ==========================================
# 4. THE AUTOREGRESSIVE NODE LOGIC
# ==========================================
if IS_VM:
    print("--- WORKER NODE READY ---")
    while True:
        # Poll RAM until Host sends a tensor (Flag == 1)
        while read_header()[0] != 1:
            time.sleep(0.001)
            
        header = read_header()
        seq_len = header[2]
        
        hidden_states = read_tensor(seq_len)
        
        # Pass through Layers 24-27 and LM Head
        outputs = model.model(inputs_embeds=hidden_states)
        logits = model.lm_head(outputs.last_hidden_state)
        
        # Grab the prediction for the very last token in the sequence
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        
        # Write back to shared memory
        header[3] = next_token_id  # Index 3 is generated_token_id
        header[0] = 2              # Tell VM C-Engine to blast it back
        write_header(header)
        
        # Wait for VM C-Engine to clear the flag before looping
        while read_header()[0] == 2:
            time.sleep(0.001)

else:
    print("--- MASTER NODE READY ---")
    while True:
        prompt = input("\nEnter Prompt: ")
        if not prompt: continue
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print("\nResponse: ", end="", flush=True)
        
        for step in range(MAX_NEW_TOKENS):
            seq_len = input_ids.shape[1]
            
            # Pass through Host Layers 0-23
            outputs = model.model(input_ids=input_ids)
            hidden_states = outputs.last_hidden_state
                
            # Write to Shared Memory
            write_tensor(hidden_states)
            
            header = read_header()
            header[2] = seq_len  
            header[0] = 1        # Tell Host C-Engine to blast the tensor
            write_header(header)
            
            # Poll RAM until Host C-Engine receives the token (Flag == 3)
            # NOTE: If your C-engine uses '2' to mean finished, change this to 2!
            while read_header()[0] != 2: 
                time.sleep(0.001)

            # Grab the token from Index 3
            header = read_header()
            predicted_token = header[3] 
            
            # Decode and print like a typewriter
            word = tokenizer.decode([predicted_token])
            print(word, end="", flush=True)
            
            # Stop if the AI generated the End-Of-Sequence token
            if predicted_token == tokenizer.eos_token_id:
                break
                
            # Append the new token to our context window for the next loop!
            input_ids = torch.cat([input_ids, torch.tensor([[predicted_token]])], dim=-1)
            
            # Reset the flag so we can safely start the next math pass
            header[0] = 0
            write_header(header)
            
        print("\n[Generation Complete]")