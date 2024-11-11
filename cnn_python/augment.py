import argparse
from pathlib import Path
from torchvision import transforms
from PIL import Image

FINAL_IMG_SIZE = 224

def parse_args():
    parser = argparse.ArgumentParser(prog='Augment images')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str )
    parser.add_argument('transform_type', type=str, choices=['pad', 'aug'])
    parser.add_argument('augmentations_count', type=int)
    return parser.parse_args()

def correctExifRotation(img: Image.Image) -> Image.Image:
    w, h = img.size
    if (w > h):
        img = img.rotate(90, expand=True)
    return img

def square(img: Image.Image) -> Image.Image:
    crop = transforms.CenterCrop(min(img.size))
    return crop(img)

def get_augmentation_transforms():
    return transforms.Compose([
        transforms.Lambda(lambd=correctExifRotation),
        transforms.Lambda(lambd=square),
        transforms.Resize(size=(256, 256)),
        transforms.RandomRotation(degrees=25, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((FINAL_IMG_SIZE, FINAL_IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    ])

def get_padding_transforms():
    return transforms.Compose([
            transforms.Lambda(lambd=correctExifRotation),
            transforms.Lambda(lambda img: transforms.Pad(padding=((img.size[1] - img.size[0]) // 2, 0),
                                                         fill=0, padding_mode='constant')(img)),
            transforms.Pad(padding=100, fill=0, padding_mode='constant')
        ])

def augment_images(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if (args.transform_type == 'aug'):
        transform = get_augmentation_transforms()
    elif (args.transform_type == 'pad'):
        transform = get_padding_transforms()
    else: exit(1)
    
    class_folders = [d for d in input_path.iterdir() if d.is_dir()]

    total = 0
    for in_folder in class_folders:
        out_folder = output_path / in_folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        idx = 0
        for img_path in in_folder.glob("*.jpg"):
            print(f"Processing: {img_path}")
            img = Image.open(img_path).convert('RGB')
            if img is None or img.size[0] < FINAL_IMG_SIZE or img.size[1] < FINAL_IMG_SIZE:
                print(f"Skipping: {img_path}")
                continue
            
            for a in range(args.augmentations_count):
                t_img = transform(img)
                output_filename = f"{out_folder.name}_{idx}_{args.transform_type}_{a}.jpg"
                t_img.save(out_folder / output_filename)
                total += 1
                
            idx += 1
    print(f"Total number of images created: {total}")
    
def main():
    args = parse_args()
    augment_images(args)

if __name__ == "__main__":
    main()