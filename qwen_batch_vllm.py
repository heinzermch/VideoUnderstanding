from vllm.outputs import RequestOutput


import os
import time
import pandas as pd
from vllm import LLM, SamplingParams
from output_processing.answer_parsing import parse_answers
from data_processing.image_loading import load_images

QUESTIONS = [
    "Is there at least one person visible?",
    "Is there a lake, river, or large body of water?"
]

def main(only_run_n_examples: int=-1):
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


    image_files = load_images(image_dir, keep_every_nth_image=2)
    
    print(f"Found {len(image_files)} images in {image_dir} and subfolders")

    PROMPT = "Answer with yes or no. \n" + "\n".join([f"{i+1}. {question}" for i, question in enumerate(QUESTIONS)])

    # Store results for dataframe
    results = []

    if only_run_n_examples > 0:
        image_files = image_files[:only_run_n_examples]


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
            answers = parse_answers(generated_text, num_questions=len(QUESTIONS))
            
            # Store results
            results.append({
                'file_name': img_path,
                **{QUESTIONS[i]: answer for i, answer in enumerate(answers)}
            })

        

    
    # Create pandas dataframe
    df = pd.DataFrame(results)
    
    # Display the dataframe
    print("\n" + "="*60)
    print("Results DataFrame:")
    print("="*60)
    print(df.to_string(index=False))
    
    return df




if __name__ == "__main__":
    MAX_EXAMPLES_TO_RUN = 50000
    start_time = time.time()
    answers = main(only_run_n_examples=MAX_EXAMPLES_TO_RUN)
    # Time the execution
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    # Create output directory if it doesn't exist
    os.makedirs("output_files", exist_ok=True)
    # Save the results to a CSV file with time of day and timestamp
    answers.to_csv(f"output_files/qwen_batch_vllm_results_{time.strftime('%Y%m%d_%H%M%S')}_rows_{len(answers)}.csv", index=False)