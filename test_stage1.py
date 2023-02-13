import argparse

import torch
import torchvision
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

from ldm.data.coco import CocoTrainValid 

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset_type", type=str, default="coco_val")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    if opt.dataset_type == "coco_val":
        dset = CocoTrainValid(root="data/mscoco", split="valid", image_resolution=256, transform_type="imagenet_val", is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    elif opt.dataset_type == "coco_train":
        dset = CocoTrainValid(root="data/mscoco", split="train", image_resolution=256, transform_type="imagenet_val", is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)

    with torch.no_grad():
        for i,data in enumerate(dloader):
            image = data["image"].float().cuda()
            image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

            dec, diff = model(image)

            real_image = torch.clamp(image * 0.5 + 0.5, 0, 1)
            rec_image = torch.clamp(dec * 0.5 + 0.5, 0, 1)
            torchvision.utils.save_image(real_image, "real.png", normalize=False)
            torchvision.utils.save_image(rec_image, "rec.png", normalize=False)
            
            exit()