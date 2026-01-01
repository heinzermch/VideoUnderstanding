import os


def load_images(image_dir, keep_every_nth_image=1):
    # Scan all subfolders for images
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        # Filter image files first
        image_files_in_dir = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, file in enumerate(image_files_in_dir):
            # Try to use numeric filename for filtering, fall back to enumeration index
            try:
                file_number = int(file.split('.')[0])
            except ValueError:
                # If filename is not numeric, use enumeration index instead
                file_number = i
            
            if file_number % keep_every_nth_image == 0:
                image_files.append(os.path.join(root, file))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {image_dir} and subfolders")
    return image_files