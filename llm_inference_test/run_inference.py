import os
import json
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

# Create output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Load data
data_file = "../data/llm_training_sample.json"
data = []
with open(data_file, "r") as f:
    content = f.read()
    if content.strip().startswith("["):
        data = json.loads(content)
    else:
        for line in content.splitlines():
            if line.strip() and not line.strip() in ("[", "]"):
                line = line.strip()
                if line.endswith(","):
                    line = line[:-1]
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

# Use a small subset for quick testing if needed, or all
# data = data[:10]  # Uncomment to test on first 10

# Initialize Hugging Face API
api = HfApi()
models_info = api.list_models(author="farbodtavakkoli")

# Filter for LLM generative models (excluding Embeddings, Reranker, Classification, Safety)
llm_models = []
for m in models_info:
    model_id = m.modelId
    if "OTel-LLM" in model_id and ("-IT" in model_id or "-Reasoning" in model_id):
        llm_models.append(model_id)

# Helper to extract billion parameters from name
def extract_params(model_id):
    # Example: farbodtavakkoli/OTel-LLM-8.3B-IT
    try:
        parts = model_id.split("-")
        for part in parts:
            if "B" in part and part.replace("B", "").replace(".", "").isdigit() or "M" in part and part.replace("M", "").replace(".", "").isdigit():
                if "B" in part:
                    return float(part.replace("B", ""))
                elif "M" in part:
                    return float(part.replace("M", "")) / 1000
    except Exception:
        pass
    return 0.0

results = []

for model_id in llm_models:
    print(f"\n--- Testing Model: {model_id} ---")
    param_size = extract_params(model_id)

    # 1. Log Metadata
    metadata = {
        "model_id": model_id,
        "parameters_B": param_size,
    }

    try:
        # 2. Pull from Huggingface
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Handle large models with device_map="auto" and fp16 to avoid OOM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 3. Store output when queried on prompts
        model_outputs = []
        total_query_ingestion_time = 0
        total_prompt_processing_time = 0
        total_inferencing_speed = 0
        total_wall_time = 0
        total_performance_score = 0

        for idx, item in enumerate(data):
            prompt = item["prompt"]
            expected_completion = item["completion"]

            wall_start = time.time()

            # Query ingestion (Tokenization)
            ingest_start = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            ingest_end = time.time()
            query_ingestion_time = ingest_end - ingest_start

            # Prompt processing & Generation
            generate_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id
                )
            generate_end = time.time()

            # Extract generated text
            generated_sequence = outputs[0][inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

            wall_end = time.time()
            wall_time = wall_end - wall_start

            # Prompt processing (rough estimate: time to forward pass before first token generation)
            # Since we use `generate`, we approximate prompt processing vs decoding speed.
            # A more accurate way requires hooks, but we'll use total generation time / tokens as inferencing speed.
            num_generated_tokens = len(generated_sequence)
            generation_time = generate_end - generate_start
            inferencing_speed = num_generated_tokens / generation_time if generation_time > 0 else 0

            # Simple performance metric: Does the output contain the expected completion?
            # A more robust metric would be BLEU/ROUGE, but we use a basic inclusion/length heuristic for ranking
            score = 1 if expected_completion.lower() in output_text.lower() else 0

            total_query_ingestion_time += query_ingestion_time
            total_prompt_processing_time += generation_time # Grouping prompt proc and generation for now
            total_inferencing_speed += inferencing_speed
            total_wall_time += wall_time
            total_performance_score += score

            model_outputs.append({
                "prompt_idx": idx,
                "output_text": output_text,
                "expected_completion": expected_completion,
                "score": score,
                "wall_time": wall_time
            })

            if idx >= 9: # Run on 10 examples to save time during this evaluation run
                break

        num_examples = min(len(data), 10)

        # Save output for this model
        safe_model_name = model_id.replace("/", "_")
        with open(f"{output_dir}/{safe_model_name}_outputs.json", "w") as f:
            json.dump(model_outputs, f, indent=2)

        # 4 & 5. Aggregate metrics
        avg_ingestion = total_query_ingestion_time / num_examples
        avg_processing = total_prompt_processing_time / num_examples
        avg_speed = total_inferencing_speed / num_examples
        avg_wall_time = total_wall_time / num_examples
        accuracy = total_performance_score / num_examples

        metadata.update({
            "avg_query_ingestion_s": avg_ingestion,
            "avg_prompt_processing_and_generation_s": avg_processing,
            "avg_inferencing_speed_tok_per_s": avg_speed,
            "avg_wall_time_s": avg_wall_time,
            "performance_score": accuracy,
            "status": "Success"
        })

        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed to evaluate {model_id}: {e}")
        metadata["status"] = f"Failed: {e}"

    results.append(metadata)

# 6. Rank Models
df = pd.DataFrame(results)
if not df.empty and "performance_score" in df.columns:
    # We want high performance, high speed, low wall time
    # Let's create a combined score: (performance_score * 10) + (speed / 10) - (wall_time / 10)
    df["composite_score"] = (df.get("performance_score", 0) * 10) + (df.get("avg_inferencing_speed_tok_per_s", 0) / 10) - (df.get("avg_wall_time_s", 0) / 10)

    df_ranked = df.sort_values(by="composite_score", ascending=False)

    print("\n--- Model Rankings ---")
    print(df_ranked[["model_id", "parameters_B", "performance_score", "avg_inferencing_speed_tok_per_s", "avg_wall_time_s", "composite_score"]])

    best_model = df_ranked.iloc[0]["model_id"]
    print(f"\nBest model based on composite score: {best_model}")
    print("This model is recommended to be worked on next.")

    df_ranked.to_csv(f"{output_dir}/model_rankings.csv", index=False)
else:
    print("No valid results to rank.")
    df.to_csv(f"{output_dir}/model_rankings.csv", index=False)
