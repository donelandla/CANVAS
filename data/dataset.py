import os
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, image_root, hsv_preprocess, clip_preprocess, 
                    extensions={'.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'}):        
        self.image_root = image_root
        self.clip_preprocess = clip_preprocess
        self.hsv_preprocess = hsv_preprocess
        self.extensions = extensions
        self.samples = []
        for label, subdir in enumerate(['real', 'fake']):
            class_dir = os.path.join(self.image_root, subdir)
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[-1].lower() in extensions:
                    self.samples.append({
                        'path': os.path.join(class_dir, fname),
                        'label': 1 if subdir == 'fake' else 0
                    })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample['path']).convert('RGB')

        hsv_input = self.hsv_preprocess(image)
        clip_input = self.clip_preprocess(image)

        return hsv_input, clip_input, sample['label']


class TestDataset(Dataset):
    def __init__(self, image_root, hsv_preprocess, clip_preprocess, 
                 extensions={'.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'}):
        self.image_root = image_root
        self.clip_preprocess = clip_preprocess
        self.hsv_preprocess = hsv_preprocess
        self.extensions = extensions
        self.samples = []
        for model_name in os.listdir(self.image_root):
            model_dir = os.path.join(self.image_root, model_name)
            if not os.path.isdir(model_dir):
                continue

            for root, dirs, files in os.walk(model_dir):
                dirname = os.path.basename(root).lower()
                if "real" in dirname:
                    label = 0
                elif "fake" in dirname:
                    label = 1
                else:
                    continue 
                for fname in files:
                    if os.path.splitext(fname)[-1].lower() in extensions:
                        path = os.path.join(root, fname)
                        try:
                            with Image.open(path) as img:
                                img.verify()
                            self.samples.append({
                                'path': path,
                                'label': label,
                                'model_name': model_name
                            })
                        except (OSError, IOError):
                            print(f"Skipped corrupted image: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path, label, model_name = sample['path'], sample['label'], sample['model_name']
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, IOError):
            print(f"Skipped corrupted image: {path}")
            return self.__getitem__((idx + 1) % len(self.samples))

        clip_img = self.clip_preprocess(image)
        hsv_img = self.hsv_preprocess(image)

        return {
            'clip_img': clip_img,
            'hsv_img': hsv_img,
            'label': label,
            'model_name': model_name
        }
        