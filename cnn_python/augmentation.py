import argparse
import random
from pathlib import Path
import cv2
from cv2.typing import MatLike


def resizing(image: MatLike, width: int, height: int) -> MatLike:
  return cv2.resize(image, (width, height))

def rotate_random(image: MatLike, max_rotation_deg: float) -> MatLike:
  height, width = image.shape[:2]
  center = (width // 2, height // 2)
  
  degree = random.uniform(-max_rotation_deg, max_rotation_deg)
  rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
  
  return cv2.warpAffine(image, 
                        rotation_matrix, 
                        (width, height),
                        borderMode=cv2.BORDER_REPLICATE)

def crop_random(image: MatLike, max_crop: float) -> MatLike:
  height, width = image.shape[:2]
  crop_scale    = 1.0 - random.uniform(0, max_crop)
  crop_width    = int(crop_scale * width)
  crop_height   = int(crop_scale * height)
  start_x       = random.randint(0, width - crop_width)
  start_y       = random.randint(0, height - crop_height)
  
  cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
  
  return cv2.resize(cropped_image, (width, height))

def flip_vertical_random(image: MatLike, chance: float) -> MatLike:
  return cv2.flip(image, 0) if random.random() < chance else image
  
def flip_horizontal_random(image: MatLike, chance: float) -> MatLike:
  return cv2.flip(image, 1) if random.random() < chance else image

def brightness_random(image: MatLike, max_brightness_change: float) -> MatLike:
  brightness_change = 1.0 + random.uniform(-max_brightness_change, max_brightness_change)
  return cv2.convertScaleAbs(src=image, 
                             alpha=brightness_change, 
                             beta=0)

def contrast_random(image: MatLike, max_contrast_change: float) -> MatLike:
  contrast_change = 1.0 + random.uniform(-max_contrast_change, max_contrast_change)
  return cv2.convertScaleAbs(src=image, 
                             alpha=contrast_change, 
                             beta=128 * (1 - contrast_change))

def gaussian_blur_random(image: MatLike, max_blur: float) -> MatLike:
  blur = random.uniform(0, max_blur)
  return cv2.GaussianBlur(src=image, 
                          ksize=(3, 3), 
                          sigmaX=blur)

def augment_images(input_dir_path: Path, output_dir_path: Path, n=15) -> int:
  # variables for the augmentation
  height, width            = 512, 384
  max_rotation_degree      = 40
  max_crop_deviation       = 0.15
  max_brightness_deviation = 0.3
  max_contrast_deviation   = 0.3
  max_blur                 = 3.0
  
  class_dir = output_dir_path / input_dir_path.name
  class_dir.mkdir(parents=True, exist_ok=True)
  
  idx = 0
  for input_image_path in input_dir_path.glob("*.jpg"):
    # load images as grayscale
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None or original_image.shape[0] < height or original_image.shape[1] < width:
      print(f"Skipping: {input_image_path}")
      continue

    print(f"Augmenting: {input_image_path}")
    
    # resizing
    resized_image = resizing(original_image, width, height)
    
    # create random augmentations
    for i in range(n):
      augmented = resized_image
      augmented = rotate_random(augmented, max_rotation_degree)
      augmented = crop_random(augmented, max_crop_deviation)
      augmented = brightness_random(augmented, max_brightness_deviation)
      augmented = contrast_random(augmented, max_contrast_deviation)
      augmented = gaussian_blur_random(augmented, max_blur)
      
      # saving image, keeping folder structure
      output_filename = class_dir / f"{input_image_path.parent.name}_{idx}.jpg"
      idx += 1
      cv2.imwrite(filename=str(output_filename), img=augmented)

  print(f"Count of augmented images saved: {idx}")
  return idx

def main():
  parser = argparse.ArgumentParser(prog='Image Augmentation',
                                   description='Augment images for the dataset: ...')
  
  # command line arguments
  parser.add_argument('input_dir', type=str, help='Input images directory')
  parser.add_argument('output_dir', type=str, help='Output images directory')
  parser.add_argument('number_of_augmentations', type=int, help='Number of different augmentations per images')
  args = parser.parse_args()
  
  # validating provided folders
  input_dir_path = Path(args.input_dir)
  if not input_dir_path.is_dir():
    print("Provide valid input images directory!")
    exit(1)
  
  output_dir_path = Path(args.output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)
  
  # augmentation per folders
  classes_folders = [d for d in input_dir_path.iterdir() if d.is_dir()]
  if not classes_folders:
    print("No class-directories in input path!")
    exit(1)
  
  total = 0
  for f in classes_folders:
    total += augment_images(f, output_dir_path, args.number_of_augmentations)
  
  print(f"Total images created: {total}")


if __name__ == '__main__':
  main()