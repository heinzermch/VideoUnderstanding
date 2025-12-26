from vllm.outputs import RequestOutput


import os
import re
import pandas as pd
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def main():
    model_id = 'cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit'
    image_dir = "/home/michael/Videos/model_input/frames/"

    # 1. Initialize vLLM for your RTX 5080
    # On 16GB VRAM, we set gpu_memory_utilization to 0.8 to leave room
    # for the OS and the vision encoder overhead.
    # For multi-folder support, allow the parent directory to access all subfolders
    image_dir_abs = os.path.abspath(image_dir)
    allowed_path = os.path.dirname(image_dir_abs) if os.path.dirname(image_dir_abs) else image_dir_abs
    
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        dtype="float16",
        limit_mm_per_prompt={"image": 1},
        # Allow access to the parent directory to handle subfolders
        allowed_local_media_path=allowed_path
    )

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy search for consistent Yes/No answers
        max_tokens=20
    )

    # Scan all subfolders for images
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images in {image_dir} and subfolders")

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
            answers = parse_answers(generated_text, num_questions=len(questions))
            
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


def parse_answers(text, num_questions=2):
    """
    Parse the generated text to extract answers to the questions.
    Handles various formats like:
    - "Yes. No."
    - "1. Yes\n2. No"
    - "Yes\nNo"
    - etc.
    
    Args:
        text: The generated text from the model
        num_questions: Number of questions to extract answers for
    
    Returns:
        List of answers (capitalized)
    """
    text_lower = text.strip().lower()
    answers = []
    
    # Try to find numbered answers (1. ... 2. ... etc.)
    for i in range(1, num_questions + 1):
        pattern = rf'{i}[\.\)]\s*(yes|no)'
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            answers.append(match.group(1).capitalize())
    
    # If we found all numbered answers, return them
    if len(answers) == num_questions:
        return answers
    
    # Otherwise, try to find yes/no patterns in sequence
    yes_no_pattern = r'\b(yes|no)\b'
    matches = re.findall(yes_no_pattern, text_lower, re.IGNORECASE)
    
    if len(matches) >= num_questions:
        return [m.capitalize() for m in matches[:num_questions]]
    elif len(matches) > 0:
        # Fill remaining with "Unknown"
        result = [m.capitalize() for m in matches]
        result.extend(["Unknown"] * (num_questions - len(matches)))
        return result
    else:
        return ["Unknown"] * num_questions

if __name__ == "__main__":
    answers = main()
    answers.to_csv("qwen_batch_vllm_results.csv", index=False)