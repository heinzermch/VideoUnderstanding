import csv
import os
from PIL import Image, ImageDraw, ImageFont


def visualize_nonzero_answers(csv_file_path, output_dir="nonzero"):
    """
    Read CSV file, filter rows where answer is not "No",
    and write questions and answers on the corresponding images.
    
    Args:
        csv_file_path: Path to the CSV file
        output_dir: Directory to save the annotated images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        questions = reader.fieldnames[1:]  # Skip 'file_name' column
        
        processed_count = 0
        skipped_count = 0
        
        for row in reader:
            file_name = row['file_name']
            answers = [row[q] for q in questions]
            
            # Check if any answer is not "No"
            if any(answer.strip().lower() != 'no' for answer in answers):
                try:
                    # Load the image
                    if not os.path.exists(file_name):
                        print(f"Warning: Image not found: {file_name}")
                        skipped_count += 1
                        continue
                    
                    img = Image.open(file_name)
                    # Convert to RGBA if needed for overlay
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    draw = ImageDraw.Draw(img)
                    
                    # Try to load a font, fallback to default if not available
                    font_size = max(20, min(img.width, img.height) // 30)
                    font = None
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
                        except:
                            try:
                                font = ImageFont.load_default()
                            except:
                                font = None
                    
                    # Prepare text to write
                    y_offset = 10
                    line_height = font_size + 15
                    
                    # Write each question and answer
                    for question, answer in zip(questions, answers):
                        text = f"{question}: {answer}"
                        # Get text dimensions
                        if font:
                            bbox = draw.textbbox((0, 0), text, font=font)
                        else:
                            bbox = draw.textbbox((0, 0), text)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Draw semi-transparent background rectangle
                        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        overlay_draw.rectangle(
                            [(5, y_offset - 2), (text_width + 15, y_offset + text_height + 2)],
                            fill=(0, 0, 0, 180)
                        )
                        img = Image.alpha_composite(img, overlay)
                        draw = ImageDraw.Draw(img)
                        
                        # Draw text in white
                        if font:
                            draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
                        else:
                            draw.text((10, y_offset), text, fill=(255, 255, 255))
                        y_offset += line_height
                    
                    # Convert back to RGB for saving
                    img = img.convert('RGB')
                    
                    # Preserve folder structure: extract subfolder from original path
                    # e.g., /home/michael/Videos/model_input/frames/5seenwanderung_letztersee_hd/00000.png
                    # -> extract "5seenwanderung_letztersee_hd"
                    file_dir = os.path.dirname(file_name)
                    subfolder = os.path.basename(file_dir)  # Get the subfolder name
                    
                    # Create the subfolder in output directory
                    output_subfolder = os.path.join(output_dir, subfolder)
                    os.makedirs(output_subfolder, exist_ok=True)
                    
                    # Save the image preserving the filename
                    output_path = os.path.join(output_subfolder, os.path.basename(file_name))
                    img.save(output_path)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} images...")
                        
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    skipped_count += 1
                    continue
    
    print(f"\nCompleted!")
    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Output directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # Default CSV file - you can change this
    csv_file = "output_files/qwen_batch_vllm_results_20251226_185030_rows_1711.csv"
    output_dir = "/home/michael/Videos/model_output"
    visualize_nonzero_answers(csv_file, output_dir=output_dir)

