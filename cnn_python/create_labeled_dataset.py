import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms.functional import to_pil_image
import argparse
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class CustomDataset(Dataset):
  def __init__(self, pre_formatted_images_folder: Path, output_path: Path):
    self.src_img_paths = list(pre_formatted_images_folder.glob("*.jpg"))
    self.output_path = output_path
    self.label_map = {"tok_7"       : 0,  "tok_8"       : 1,  "tok_9"       : 2,  "tok_10"      : 3,
                      "tok_also"    : 4,  "tok_felso"   : 5,  "tok_kiraly"  : 6,  "tok_asz"     : 7,
                      "makk_7"      : 8,  "makk_8"      : 9,  "makk_9"      : 10, "makk_10"     : 11,
                      "makk_also"   : 12, "makk_felso"  : 13, "makk_kiraly" : 14, "makk_asz"    : 15,
                      "sziv_7"      : 16, "sziv_8"      : 17, "sziv_9"      : 18, "sziv_10"     : 19,
                      "sziv_also"   : 20, "sziv_felso"  : 21, "sziv_kiraly" : 22, "sziv_asz"    : 23,
                      "zold_7"      : 24, "zold_8"      : 25, "zold_9"      : 26, "zold_10"     : 27,
                      "zold_also"   : 28, "zold_felso"  : 29, "zold_kiraly" : 30, "zold_asz"    : 31 }
    self.label_map_key_n = { val : key for key, val in self.label_map.items() }
    
  def __len__(self):
    return len(self.src_img_paths)
  
  def __getitem__(self, idx):
    img_path = self.src_img_paths[idx]
    # uint8 tensor -> normalize to 0-1
    img_tensor = io.decode_image(img_path).float() / 255.0
    # standardization
    # img_tensor = (img_tensor.float() / 255.0 - 0.5) / 0.5
    # print(img_tensor.shape) # -> [channels 1, height 512, width 384]
    label_key = "_".join(self.src_img_paths[idx].stem.split("_", 2)[:2])
    label = self.label_map[label_key]
    return img_tensor, label

  def verify_saved_dataset(self):
    print(f"Verifying dataset: {self.output_path}")

    saved_dataset = torch.load(self.output_path)
    img_tensors = saved_dataset['img_tensors']
    labels = saved_dataset['labels']
    
    print("Displaying some images with their labels:")
    for idx in [0, self.__len__() // 2 , -1]:
        img = to_pil_image(img_tensors[idx])
        label = labels[idx].item()
        str_label = self.label_map_key_n[label]
        print(f"Current label: {label} : {str_label}.")
        img.show()

    print(f"Tensor count: {img_tensors.shape[0]}. Label count: {labels.shape[0]}.")
    print(f"Tensor [N, C, H, W] shape: {img_tensors.shape}. Label shape: {labels.shape}")
    print(f"Tensor type: {img_tensors.dtype}. Label type: {labels.dtype}")
    print(f"Tensor max val: {img_tensors.min()}.Image tensor min val: {img_tensors.max()}")


  def save_dataset(self):
    img_tensors = []
    labels = []
    # Dataset class is iterable by default:
    for img_tensor, label in self:
      img_tensors.append(img_tensor)
      labels.append(torch.tensor(label))
    
    img_tensors = torch.stack(img_tensors)
    labels = torch.stack(labels)
    torch.save( {'img_tensors': img_tensors, 'labels': labels}, self.output_path )
    print(f"Dataset saved: {self.output_path}")
    self.verify_saved_dataset()
      

def main():
  parser = argparse.ArgumentParser(prog='creating labeled_dataset.pt')
  
  # command line arguments
  parser.add_argument('input_dir', type=str, help='Input pre-formatted images directory')
  parser.add_argument('output_dir', type=str, help='Output labeled dataset directory')
  #parser.add_argument('batch_size', type=int, help='Batch size for loading the images')
  args = parser.parse_args()
  
  # validating provided folders
  input_dir_path = Path(args.input_dir)
  if not input_dir_path.is_dir():
    print("Provide valid input images directory!")
    exit()
  
  output_dir_path = Path(args.output_dir)
  if not output_dir_path.is_dir():
    print("Provide valid output images directory!")
    exit()
    
  output_file = output_dir_path / "labeled_dataset.pt"
  
  d = CustomDataset(pre_formatted_images_folder=input_dir_path, output_path=output_file)
  d.save_dataset()


if __name__ == '__main__':
  main()
  