from vllm.outputs import RequestOutput


import os
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def main():
    model_id = 'cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit'
    image_dir = "/home/michael/Videos/model_input/frames/5seenwanderung_zweitersee_hd"

    # 1. Initialize vLLM for your RTX 5080
    # On 16GB VRAM, we set gpu_memory_utilization to 0.8 to leave room
    # for the OS and the vision encoder overhead.
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        dtype="float16",
        limit_mm_per_prompt={"image": 1},
        # THIS IS THE KEY FIX:
        allowed_local_media_path=image_dir 
    )

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy search for consistent Yes/No answers
        max_tokens=20
    )

    
    image_files = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    PROMPT = "Answer with yes or no.\n1. Is there at least one person visible?\n2. Is there a lake, river, or large body of water?"

    # vLLM handles batching internally. You don't need a manual loop for 
    # small batches; you can pass the whole list or larger chunks.
    batch_size = 8 
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        # Format the inputs for vLLM Chat
        vllm_inputs = []
        for img_path in batch_files:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"file://{img_path}"}
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }]
            vllm_inputs.append({"prompt": messages})

        # Generate batch results
        outputs = llm.chat(messages=[inp["prompt"] for inp in vllm_inputs], 
                          sampling_params=sampling_params)

        for img_path, output in zip(batch_files, outputs):
            generated_text = output.outputs[0].text
            print(f"\n--- Result for {os.path.basename(img_path)} ---")
            print(generated_text.strip())

if __name__ == "__main__":
    main()