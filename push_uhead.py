from pathlib import Path
from luh import AutoUncertaintyHead
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model (minimal setup)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", 
    torch_dtype="auto", 
    device_map="cpu"
)

# Load UHead using the same method as training script
local_path = "/home/wutianyi/uncertainty4reasoning/checkpoints/uhead_claim_Qwen3-8B_trip_plan_wrapper"
uq_head = AutoUncertaintyHead.from_pretrained(local_path, base_model)

# Push to hub - replace with your desired repo name
repo_id = "awsuineg/uhead-claim-qwen3-8b-trip-plan"
uq_head.push_to_hub(repo_id)

print(f"Successfully pushed to: https://huggingface.co/{repo_id}")