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
import pickle

sys.path.append(os.getcwd())

from ldm.data.coco import CocoTrainValid
from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--sample_num", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--use_ddim", type=bool, default=True)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--plms", action='store_true', help="use plms sampling",)
    parser.add_argument("--dpm_solver", action='store_true', help="use dpm_solver sampling",)
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across samples ",)
    parser.add_argument("--scale", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
    
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space",)
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space",)
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor",)
    return parser

if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    save_image_path = opt.model_path.replace(".ckpt", "") + "_" + now + "/images"
    save_pickle_path = opt.model_path.replace(".ckpt", "") + "_" + now + "/pickles"
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_pickle_path, exist_ok=True)
    with open(opt.model_path.replace(".ckpt", "") + "_" + now + "/sample_config.txt", "w") as f:
        f.write(str(opt))
    
    val_dataset = CocoTrainValid(root="data/mscoco", split="valid", image_resolution=256, transform_type="imagenet_val", is_eval=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    
    config = OmegaConf.load(opt.yaml_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.model_path)["state_dict"], strict=False)
    model.cuda()
    model.eval()
    
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    start_code = None
    sampled_num = 0
    if opt.sample_num % opt.batch_size == 0:
        total_batch = opt.sample_num // opt.batch_size
    else:
        total_batch = opt.sample_num // opt.batch_size + 1
    for i, data in tqdm(enumerate(val_dataloader)):
        c = data["caption"]
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.batch_size * [""])
                c = model.get_learned_conditioning(c)
                
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                
                save_pickle(os.path.join(save_pickle_path, 'samples_({}_{}).pkl'.format(i, total_batch)), x_samples_ddim.cpu().numpy(),)
                for j in range(opt.batch_size):
                    x = x_samples_ddim[j].detach().cpu()
                    x = x.permute(1, 2, 0).numpy()
                    x = (255 * x).astype(np.uint8)
                    x = Image.fromarray(x)
                    if not x.mode == "RGB":
                        x = x.convert("RGB")
                    x.save("{}/samples_{}_{}.png".format(save_image_path, i, j))
                    
                    sampled_num += 1
                    if sampled_num >= opt.sample_num:
                        exit()