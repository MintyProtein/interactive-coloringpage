import os
import random
import argparse
from omegaconf import OmegaConf
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
from patchify import patchify, unpatchify
from src.postprocessor.model import UNetPostprocessor
from src.postprocessor.data import PostprocessorDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore


if __name__=='__main__':
    device = torch.device("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default='')
    args = parser.parse_args()
    
    ###### Load Config ######
    cfg = OmegaConf.load(args.train_config)
    run = wandb.init(config=OmegaConf.to_container(cfg))
    cfg = wandb.config
    seed_everything(cfg.seed)
    
    ###### Prepare Data ######
    checkpoint_path = cfg.model_checkpoint + "/postprocessor.pt"
    match cfg.patch_size:
        case 32: img_dir = cfg.dataset_dir + "/patches_32/"
        case 48: img_dir = cfg.dataset_dir + "/patches_48/"
        case 96: img_dir = cfg.dataset_dir + "/patches_96/"
    
    img_paths = np.array(glob.glob(f"{img_dir}/*"))
    train_paths, val_paths = train_test_split(img_paths, test_size=0.2, random_state=cfg.seed)

    train_dataset = PostprocessorDataset(train_paths, max_holes=cfg.max_holes)
    val_dataset = PostprocessorDataset(val_paths, max_holes=0)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=4)
    
    sample_paths = glob.glob('sample_images/*')
    sample_images = []
    for path in sample_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(cfg.img_w, cfg.img_h))
        sample_images.append(img)


    ###### Prepare Model ######
    match cfg.loss:
        case 'MSE': criterion = nn.MSELoss()
        case 'BCE': criterion = nn.BCEWithLogitsLoss()
    
    model = UNetPostprocessor(in_channels=cfg.in_channels,
                 out_channels=cfg.out_channels,
                 width=cfg.model_width,
                 depth=cfg.model_depth,
                 kernel_size=cfg.kernel_size, 
                 patch_size=cfg.patch_size,
                 device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), cfg.lr)

    ###### Train! ######
    total_train_losses = []
    total_val_losses = []
    best_val = 999
    for epoch in range(cfg.epochs):
        train_losses = []
        val_losses = []
        model.train()
        for x, y in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        with torch.no_grad():
            model.eval()
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            
            ###### Sample Images ######
            sample_outputs = []
            for image in sample_images:
                sample_outputs.append(wandb.Image(model.inference(image)))
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "sample_images": sample_outputs
            })
            
            if val_loss < best_val:
                torch.save(model.state_dict(), checkpoint_path)
                best_bal = val_loss
