# Aegis - SOC

## Datasets

HuggingFace/datasets/
https://huggingface.co/datasets/ ...

AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1   [X]
https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1

AlicanKiraz0/Cybersecurity-Dataset-v1
https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1

AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1
https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1

AlicanKiraz0/All-CVE-Records-Training-Dataset
https://huggingface.co/datasets/AlicanKiraz0/All-CVE-Records-Training-Dataset

tihanyin/CyberMetric
https://huggingface.co/datasets/tihanyin/CyberMetric

tuandunghcmut/combine-llm-security-benchmark  [ gated, I already request access, still pending ]  this one has few  likes in HF and very  few donwloads,  it looks it is not as good as the others which are open ready to train.  [X*]
https://huggingface.co/datasets/tuandunghcmut/combine-llm-security-benchmark

Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset
https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset?utm_source=chatgpt.com

oyildirim/cyberstrike-sft-120k  [+++]
https://huggingface.co/datasets/oyildirim/cyberstrike-sft-120k

---------------------------------------------------

 The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None}.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 55,275 | Num Epochs = 3 | Total steps = 6,912
O^O/ \_/ \    Batch size per device = 12 | Gradient accumulation steps = 2
\        /    Data Parallel GPUs = 1 | Total batch size (12 x 2 x 1) = 24
 "-____-"     Trainable parameters = 161,480,704 of 7,777,097,216 (2.08% trained)
`use_return_dict` is deprecated! Use `return_dict` instead!
Unsloth: Will smartly offload gradients to save VRAM!
Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.

    
      
      
      [ 375/6912 1:03:31 < 31:02:18, 0.06 it/s, Epoch 0.16/3]
    
    
  
 
      Step
      Training Loss
      Validation Loss
    
  
  
    
      200
      1.058180
      1.020077
    
    
      250
      0.987876
      0.992787
    
    
      300
      0.959798
      0.970407
    
    
      350
      0.974118
      0.954969
    
  
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-200/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-250/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-300/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-350/tokenizer_config.json.
-----------------------

 The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None}.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 55,275 | Num Epochs = 3 | Total steps = 6,912
O^O/ \_/ \    Batch size per device = 12 | Gradient accumulation steps = 2
\        /    Data Parallel GPUs = 1 | Total batch size (12 x 2 x 1) = 24
 "-____-"     Trainable parameters = 161,480,704 of 7,777,097,216 (2.08% trained)
`use_return_dict` is deprecated! Use `return_dict` instead!
Unsloth: Will smartly offload gradients to save VRAM!
Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.

    
      
      
      [ 601/6912 2:08:25 < 30:05:09, 0.06 it/s, Epoch 0.26/3]
    
    
  
 
      Step
      Training Loss
      Validation Loss
    
  
  
    
      200
      1.058180
      1.020077
    
    
      250
      0.987876
      0.992787
    
    
      300
      0.959798
      0.970407
    
    
      350
      0.974118
      0.954969
    
    
      400
      0.958650
      0.940252
    
    
      450
      0.968929
      0.930109
    
    
      500
      0.880029
      0.920330
    
    
      550
      0.948989
      0.913324
    
  
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-200/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-250/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-300/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-350/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-400/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-450/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-500/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in /content/drive/MyDrive/Aegis-CyberSec-Guard/checkpoints/checkpoint-550/tokenizer_config.json.

____________________________________

https://www.figma.com/make/84srNgzZbt48YHIH2PYF2r/Cyberpunk-HUD-Console-Design?t=CzyklFevO0xYBqRL-0



