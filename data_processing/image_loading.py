import os


def load_images(image_dir, keep_every_nth_image=1):
    # Scan all subfolders for images
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for i, file in enumerate(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_number = int(file.split('.')[0])
                if file_number % keep_every_nth_image == 0:
                    image_files.append(os.path.join(root, file))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {image_dir} and subfolders")
    return image_files