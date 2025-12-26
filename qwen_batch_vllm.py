from vllm.outputs import RequestOutput


import os
import re
import pandas as pd
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

    questions = [
        "Is there at least one person visible?",
        "Is there a lake, river, or large body of water?"
    ]

    PROMPT = "Answer with yes or no. \n" + "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])

    # Store results for dataframe
    results = []

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
            generated_text = output.outputs[0].text.strip()
            print(f"\n--- Result for {os.path.basename(img_path)} ---")
            print(generated_text)
            
            # Parse the answers from the generated text
            answers = parse_answers(generated_text)
            
            # Store results
            results.append({
                'file_name': img_path,
                **{questions[i]: answer for i, answer in enumerate(answers)}
            })
    
    # Create pandas dataframe
    df = pd.DataFrame(results)
    
    # Display the dataframe
    print("\n" + "="*60)
    print("Results DataFrame:")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


def parse_answers(text):
    """
    Parse the generated text to extract answers to the two questions.
    Handles various formats like:
    - "Yes. No."
    - "1. Yes\n2. No"
    - "Yes\nNo"
    - etc.
    """
    text = text.strip().lower()
    
    # Try to find numbered answers (1. ... 2. ...)
    pattern1 = r'1[\.\)]\s*(yes|no)'
    pattern2 = r'2[\.\)]\s*(yes|no)'
    
    match1 = re.search(pattern1, text, re.IGNORECASE)
    match2 = re.search(pattern2, text, re.IGNORECASE)
    
    if match1 and match2:
        answer1 = match1.group(1).capitalize()
        answer2 = match2.group(1).capitalize()
        return answer1, answer2
    
    # Try to find yes/no patterns in sequence
    yes_no_pattern = r'\b(yes|no)\b'
    matches = re.findall(yes_no_pattern, text, re.IGNORECASE)
    
    if len(matches) >= 2:
        return matches[0].capitalize(), matches[1].capitalize()
    elif len(matches) == 1:
        return matches[0].capitalize(), "Unknown"
    else:
        return "Unknown", "Unknown"

if __name__ == "__main__":
    answers = main()
    answers.to_csv("qwen_batch_vllm_results.csv", index=False)