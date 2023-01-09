import random
import torch
from torchvision.datasets import CocoCaptions, VisionDataset

import os, sys
sys.path.append(os.getcwd())

from ldm.data.tokenizers_factory import create_tokenizer
from ldm.data.coco_transforms import create_transforms
from tokenizers.processors import TemplateProcessing
from ldm.util import instantiate_from_config

class CocoTrainValid(VisionDataset):
    splits = {'valid', 'train'}
    def __init__(self, root, split, image_resolution, transform_type=None, is_eval=False, tokenizer_config=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        
        assert transform_type in {"dalle", "dalle-vqvae", "clip", "clip-dvae", "none", "imagenet_train", "imagenet_val"}
        transform = create_transforms(transform_type, image_resolution, split, is_eval)
        super().__init__(root, transform=transform)

        self.split = split
        self.tokenizer = instantiate_from_config(tokenizer_config)

        if split == "valid":
            self.dataset = CocoCaptions(root=f'{self.root}/images/val2014', annFile=f'{self.root}/annotations/captions_val2014.json')
        else:
            self.dataset = CocoCaptions(root=f'{self.root}/images/train2014', annFile=f'{self.root}/annotations/captions_train2014.json')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]

        if self.transform:
            img = self.transform(img)

        # text = ' '.join(text)  # text is a list of sentences. Concat them.
        if self.split == 'train':
            rnd_txt = random.randint(0, len(text)-1)
            text = text[rnd_txt]
        else:
            text = text[0]
        
        output = self.tokenizer.get_tokens(text)
        ids, mask = output["token"], output["mask"]
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)
        ids = ids.squeeze(0)
        # mask = mask.squeeze()

        return {
            "image": img,
            "caption": ids,
            "raw_text": text,
            "mask": mask,
        }

class CocoPureImageTrainValid(VisionDataset):
    splits = {'valid', 'train'}
    def __init__(self, root, split, image_resolution, transform_type=None, is_eval=False):
        assert split in self.splits, f'{split} is not in {self.splits}'
        
        assert transform_type in {"dalle", "dalle-vqvae", "clip", "clip-dvae", "none", "imagenet_train", "imagenet_val"}
        transform = create_transforms(transform_type, image_resolution, split, is_eval)
        super().__init__(root, transform=transform)

        self.split = split

        if split == "valid":
            self.dataset = CocoCaptions(root=f'{self.root}/images/val2014', annFile=f'{self.root}/annotations/captions_val2014.json')
        else:
            self.dataset = CocoCaptions(root=f'{self.root}/images/train2014', annFile=f'{self.root}/annotations/captions_train2014.json')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
        }

if __name__ == "__main__":
    dataset = CocoTrainValid(
        root="data/mscoco", split="valid", image_resolution=256, 
        transform_type="imagenet_val", is_eval=False,
        tokenizer_config={
            "target": "modules.clip_text_encoder.my_tokenizer.my_tokenize.Tokenize",
            "params": {
                "context_length": 77,
                "add_start_and_end": True,
                "with_mask": True,
                "pad_value": 0,
                "clip_embedding": False,
                "tokenizer_config": {
                     'target': 'modules.clip_text_encoder.clip.simple_tokenizer.SimpleTokenizer',
                     'params':{
                        'end_idx': 49152 # 16384 fo DALL-E
                        },
                }
            }
        },
        )

    data = dataset.__getitem__(0)
    print(data)
    print(dataset.__len__())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data["caption"])
        print(data["raw_text"])
        
        import torchvision
        torchvision.utils.save_image(data["image"], "image.png", normalize=True)
        exit()