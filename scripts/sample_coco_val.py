import torch
import torchvision
import argparse
import os, sys
import numpy as np
import pytz
import datetime
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())

from ldm.data.coco import CocoTrainValid
from ldm.util import instantiate_from_config

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--sample_num", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_ddim", type=bool, default=True)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    return parser

if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    save_path = opt.model_path.replace(".ckpt", "") + "_" + now
    os.makedirs(save_path, exist_ok=True)
    
    val_dataset = CocoTrainValid(root="data/mscoco", split="valid", image_resolution=256, transform_type="imagenet_val", is_eval=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    
    config = OmegaConf.load(opt.yaml_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.model_path)["state_dict"], strict=False)
    model.cuda()
    model.eval()
    
    sampled_num = 0
    for i, data in tqdm(enumerate(val_dataloader)):
        c = data["caption"]
        c = model.get_learned_conditioning(c)
        with model.ema_scope("Plotting"):
            samples, z_denoise_row = model.sample_log(
                cond=c, batch_size=opt.batch_size, ddim=opt.use_ddim, ddim_steps=opt.ddim_steps, eta=opt.ddim_eta
            )
            x_samples = model.decode_first_stage(samples)
            
            for j in range(opt.batch_size):
                x = x_samples[j]
                x = x.detach().cpu()
                x = torch.clamp(x, -1., 1.)
                x = (x + 1.) / 2.
                x = x.permute(1, 2, 0).numpy()
                x = (255 * x).astype(np.uint8)
                x = Image.fromarray(x)
                if not x.mode == "RGB":
                    x = x.convert("RGB")
                x.save("{}/samples_{}_{}.png".format(save_path, i, j))
                
                sampled_num += 1
                if sampled_num >= opt.sample_num:
                    exit()