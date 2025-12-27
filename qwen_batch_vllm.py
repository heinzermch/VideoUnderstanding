from vllm.outputs import RequestOutput


import os
import time
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from output_processing.answer_parsing import parse_answers
from data_processing.image_loading import load_images

QUESTIONS = [
    "Is there at least one person visible?",
    "Is there a lake, river, or large body of water?"
]

def main(model_id: str, image_dir: str, output_dir: str, only_run_n_examples: int=-1):

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

    # Format the inputs for vLLM Chat
    # vLLM handles batching internally, so we can pass all inputs at once
    messages_list = []
    for img_path in image_files:
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
        messages_list.append(messages)

    # Generate results - vLLM will handle batching internally
    outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

    for img_path, output in zip(image_files, outputs):
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save the results to a CSV file with time of day and timestamp
    output_file = os.path.join(output_dir, f"qwen_batch_vllm_results_{time.strftime('%Y%m%d_%H%M%S')}_rows_{len(df)}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen batch vLLM inference on images")
    parser.add_argument(
        "--model-id",
        type=str,
        default='cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit',
        help="Model ID to use for inference (default: cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/michael/Videos/model_input/frames/",
        help="Directory containing images to process (default: /home/michael/Videos/model_input/frames/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_files/frame_level",
        help="Directory to save output CSV files (default: output_files/frame_level)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5000,
        help="Maximum number of images to process (default: 50000, use -1 for all)"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    answers = main(
        model_id=args.model_id,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        only_run_n_examples=args.max_images if args.max_images > 0 else -1
    )
    # Time the execution
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")