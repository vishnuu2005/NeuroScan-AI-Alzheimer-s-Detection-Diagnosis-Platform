import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms


def augment_and_balance(data_dir, target_dir, image_size=128):
    """
    Augment and balance classes by generating extra images for minority classes.

    Args:
        data_dir (str): Original dataset path (subfolders per class)
        target_dir (str): Folder to save augmented/balanced images
        image_size (int): Resize images to this size
    """
    os.makedirs(target_dir, exist_ok=True)
    classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]

    # Count images per class
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
    max_count = max(class_counts.values())
    print(f"Original class counts: {class_counts}, balancing to {max_count} per class.")

    # Define augmentation transforms
    augment_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((image_size, image_size))
    ])

    for cls in classes:
        src_dir = os.path.join(data_dir, cls)
        dst_dir = os.path.join(target_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)

        images = os.listdir(src_dir)
        count = len(images)

        # Copy original images
        for img_name in images:
            img_path = os.path.join(src_dir, img_name)
            Image.open(img_path).save(os.path.join(dst_dir, img_name))

        # Generate augmented images until class reaches max_count
        while count < max_count:
            img_name = random.choice(images)
            img_path = os.path.join(src_dir, img_name)
            img = Image.open(img_path)

            # Apply augmentation
            img_aug = augment_transforms(img)

            # Slight random zoom/crop
            width, height = img_aug.size
            scale = random.uniform(0.9, 1.1)  # 90% - 110%
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_aug = img_aug.resize((new_width, new_height), Image.BILINEAR)
            left = max(0, (new_width - width) // 2)
            top = max(0, (new_height - height) // 2)
            img_aug = img_aug.crop((left, top, left + width, top + height))

            # Save augmented image
            img_aug.save(os.path.join(dst_dir, f"aug_{count}.jpg"))
            count += 1

    print(f"Augmentation complete. Augmented dataset saved to: {target_dir}")


# ------------------------
# Run script directly
# ------------------------
if __name__ == "__main__":
    train_dir = r"D:/AlzeigmersChatBot/AlzeigmersChatBot/data/train"
    augmented_dir = r"D:/AlzeigmersChatBot/AlzeigmersChatBot/data/train_aug"
    augment_and_balance(train_dir, augmented_dir, image_size=128)