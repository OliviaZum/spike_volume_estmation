import os
from PIL import Image
from torchvision import transforms as v2
import torchvision.transforms.v2.functional as F

class BottomBiasedCrop:
    def __init__(self, size, vertical_shift_ratio=0):
        self.size = size
        self.shift_ratio = vertical_shift_ratio  # 0 = center, 1 = bottom edge

    def __call__(self, img):
        h, w = F.get_dimensions(img)[1:]
        th, tw = self.size, self.size
        # vertical bias: move crop window down by a fraction
        y_center = int(h // 2 + self.shift_ratio * (h // 2))
        y1 = max(0, min(h - th, y_center - th // 2))
        x1 = max(0, (w - tw) // 2)
        return F.crop(img, y1, x1, th, tw)


image_transform = v2.Compose([
    v2.Resize(256, antialias=True),
    BottomBiasedCrop(256, vertical_shift_ratio=0), 
])

# Paths (modify these)
input_folder = "/projects/zumstego/1_datasets/field_images_no_bar" 
output_folder = "/projects/zumstego/1_datasets/field_images_test" 

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

number = 0
# Process all images in the input folder
for filename in os.listdir(input_folder):
    print(number)
    if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):  
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename) 

        # Open and transform image
        img = Image.open(input_path).convert("RGB")
        transformed_img = image_transform(img)

        # Save to output folder
        transformed_img.save(output_path)
        number +=1

print(f"All images have been transformed and saved in {output_folder}")

