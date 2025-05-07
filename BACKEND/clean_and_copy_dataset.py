import os
from PIL import Image, UnidentifiedImageError
import shutil

def clean_and_copy_dataset(original_dir, clean_dir='cleaned_data', corrupt_dir='corrupt'):
    clean_output_dir = os.path.join(clean_dir, os.path.basename(original_dir))
    corrupt_output_dir = os.path.join(corrupt_dir, os.path.basename(original_dir))

    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(corrupt_output_dir, exist_ok=True)

    total_checked = 0
    total_moved = 0
    total_copied = 0

    for class_name in os.listdir(original_dir):
        class_path = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        clean_class_path = os.path.join(clean_output_dir, class_name)
        corrupt_class_path = os.path.join(corrupt_output_dir, class_name)
        os.makedirs(clean_class_path, exist_ok=True)
        os.makedirs(corrupt_class_path, exist_ok=True)

        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            total_checked += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Copy clean image
                shutil.copy(file_path, os.path.join(clean_class_path, file_name))
                total_copied += 1
            except (UnidentifiedImageError, OSError):
                # Move corrupt image
                print(f"[Corrupt] Moving: {file_path}")
                shutil.move(file_path, os.path.join(corrupt_class_path, file_name))
                total_moved += 1

    print(f"\n✅ Finished.\nTotal Images Checked: {total_checked}\nClean Copied: {total_copied}\nCorrupt Moved: {total_moved}")
    print(f"Clean dataset saved to: {clean_output_dir}")
    print(f"Corrupt images saved to: {corrupt_output_dir}")


# 👇 Run it like this
clean_and_copy_dataset("data/train")
# Optionally do the same for validation
clean_and_copy_dataset("data/validation")
# clean_and_copy_dataset("data/validation")
