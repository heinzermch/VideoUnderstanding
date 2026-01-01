from vllm.outputs import RequestOutput


import os
import time
import argparse
import pandas as pd
import ctypes
import gc
from vllm import LLM, SamplingParams
from output_processing.answer_parsing import parse_answers
from data_processing.image_loading import load_images

QUESTIONS = [
    "Is there at least one person visible?",
    "Is there a lake, river, or large body of water?"
]

def _init_llm(model_id: str, image_dir: str):
    """Initialize vLLM engine with proper configuration."""
    image_dir_abs = os.path.abspath(image_dir)
    allowed_path = os.path.dirname(image_dir_abs) if os.path.dirname(image_dir_abs) else image_dir_abs
    
    return LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        dtype="float16",
        limit_mm_per_prompt={"image": 1},
        # Allow access to the parent directory to handle subfolders
        allowed_local_media_path=allowed_path
    )


def main(model_id: str, image_dir: str, output_dir: str, only_run_n_examples: int=-1, batch_size: int=500, restart_every_n_batches: int=10, chunk_file: str=None, chunk_idx: int=None, base_timestamp: str=None):

    # 1. Initialize vLLM for your RTX 5080
    # On 16GB VRAM, we set gpu_memory_utilization to 0.7 to leave room
    # for the OS and the vision encoder overhead.
    llm = _init_llm(model_id, image_dir)

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy search for consistent Yes/No answers
        max_tokens=20
    )


    # Load image files - either from directory or from chunk file list
    if chunk_file and os.path.exists(chunk_file):
        import json
        print(f"Loading image list from chunk file: {chunk_file}")
        with open(chunk_file, 'r') as f:
            image_files = json.load(f)
        print(f"Loaded {len(image_files)} images from chunk file")
    else:
        image_files = load_images(image_dir, keep_every_nth_image=2)
        print(f"Found {len(image_files)} images in {image_dir} and subfolders")

    PROMPT = "Answer with yes or no. \n" + "\n".join([f"{i+1}. {question}" for i, question in enumerate(QUESTIONS)])

    if only_run_n_examples > 0:
        image_files = image_files[:only_run_n_examples]

    # Process in batches to avoid OOM when dealing with large numbers of images
    # vLLM handles batching internally, but we need to limit the queue size
    BATCH_SIZE = batch_size  # Process images in chunks to avoid memory issues

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file with timestamp (we'll append to it incrementally)
    if base_timestamp:
        timestamp = base_timestamp
        chunk_suffix = f"_chunk_{chunk_idx}" if chunk_idx else ""
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        chunk_suffix = ""
    output_file = os.path.join(output_dir, f"qwen_batch_vllm_results_{timestamp}_rows_{len(image_files)}{chunk_suffix}.csv")
    
    # Process images in batches
    total_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(image_files)} images in {total_batches} batches of up to {BATCH_SIZE} images each")
    print(f"Results will be saved incrementally to: {output_file}")
    
    # Track if we need to write header (first batch only)
    write_header = True
    
    # Create batch indices
    batch_indices = list(range(0, len(image_files), BATCH_SIZE))
    
    for batch_i, batch_idx in enumerate(batch_indices):
        # Periodically restart vLLM engine to free accumulated memory
        if batch_i > 0 and batch_i % restart_every_n_batches == 0:
            print(f"\n🔁 Restarting vLLM engine to free memory (after {batch_i} batches)...")
            
            del llm
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)  # Force system RAM release
            # time.sleep(2)  # Give system time to free memory
            
            llm = _init_llm(model_id, image_dir)
            print("✅ vLLM engine restarted successfully")
        
        batch_files = image_files[batch_idx:batch_idx + BATCH_SIZE]
        current_batch_num = batch_i + 1
        
        print(f"\nProcessing batch {current_batch_num}/{total_batches} ({len(batch_files)} images)...")
        
        # Format the inputs for vLLM Chat for this batch
        messages_list = []
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
            messages_list.append(messages)

        # Generate results for this batch - vLLM will handle batching internally
        outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

        # Process results and save immediately to avoid memory accumulation
        batch_results = []
        for img_path, output in zip(batch_files, outputs):
            generated_text = output.outputs[0].text.strip()
            # Only print first few results to reduce memory usage from print buffers
            if len(batch_results) < 3:
                print(f"  Result for {os.path.basename(img_path)}: {generated_text[:50]}...")
            
            # Parse the answers from the generated text
            answers = parse_answers(generated_text, num_questions=len(QUESTIONS))
            
            # Store results for this batch
            batch_results.append({
                'file_name': img_path,
                **{QUESTIONS[i]: answer for i, answer in enumerate(answers)}
            })
        
        # Save batch results immediately to CSV (append mode)
        batch_df = pd.DataFrame(batch_results)
        batch_df.to_csv(output_file, mode='a', header=write_header, index=False)
        write_header = False  # Don't write header for subsequent batches
        
        # Explicitly clean up large objects to free memory
        del messages_list, outputs, batch_results, batch_df
        gc.collect()                                # Clear Python objects
        ctypes.CDLL('libc.so.6').malloc_trim(0)      # FORCE System RAM release
        
        total_saved = min(batch_idx + BATCH_SIZE, len(image_files))
        print(f"Completed batch {current_batch_num}/{total_batches}. Saved {total_saved} results so far")

    print(f"\nAll batches completed! Results saved to: {output_file}")
    
    # Read back the final CSV to return as DataFrame (optional, for compatibility)
    df = pd.read_csv(output_file)
    print(f"\nTotal results: {len(df)} rows")
    
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
        default=500000,
        help="Maximum number of images to process (default: 50000, use -1 for all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to process per batch (default: 500, reduce if running out of memory)"
    )
    parser.add_argument(
        "--restart-every-n-batches",
        type=int,
        default=5,
        help="Restart vLLM engine every N batches to free memory (default: 10, set to 0 to disable)"
    )
    parser.add_argument(
        "--chunk-file",
        type=str,
        default=None,
        help="JSON file containing list of image files to process (for chunk processing)"
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=None,
        help="Chunk index (for chunk processing)"
    )
    parser.add_argument(
        "--base-timestamp",
        type=str,
        default=None,
        help="Base timestamp for output files (for chunk processing)"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    answers = main(
        model_id=args.model_id,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        only_run_n_examples=args.max_images if args.max_images > 0 else -1,
        batch_size=args.batch_size,
        restart_every_n_batches=args.restart_every_n_batches if args.restart_every_n_batches > 0 else float('inf'),
        chunk_file=args.chunk_file,
        chunk_idx=args.chunk_idx,
        base_timestamp=args.base_timestamp
    )
    # Time the execution
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")