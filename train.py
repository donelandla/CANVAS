from tqdm import tqdm 
import argparse
from ruamel.yaml import YAML
import os
from pathlib import Path
import datetime
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from collections import defaultdict
from data import create_dataset, create_loader
from models.canvas import CANVAS


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lambda(epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return 1.0 - (epoch / warmup_epochs)
    return 0.0


def trainer(config, args, model, train_loader, val_loader, optimizer, scheduler, scheduler_warmup, device):
    start_time = time.time()
    total_epochs = config['total_epochs']
    warmup_epochs = config['warmup_epochs']
    best = 0
    best_epoch = 0
    start_epoch = 0

    for epoch in range(start_epoch, total_epochs):
        current_time = datetime.datetime.now()
        print(f"{current_time:%Y-%m-%d %H:%M:%S} : Train Epoch: [{epoch + 1:03d}]")

        # train
        model.train()
        train_loss = []

        for i, (hsv_input, clip_input, targets) in enumerate(train_loader):
            hsv_input = hsv_input.to(device)
            clip_input = clip_input.to(device)
            targets = targets.to(device).float().unsqueeze(-1)
            con_loss, cla_loss = model(hsv_input, clip_input, targets, train=True)
            lambda_weight = get_lambda(epoch, total_epochs, warmup_epochs)
            loss = lambda_weight * con_loss + (1 - lambda_weight) * cla_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler.step()

            train_loss.append(loss.item())

        mean_loss = sum(train_loss) / len(train_loss)
        current_time = datetime.datetime.now()
        print(f"{current_time:%Y-%m-%d %H:%M:%S} : \
            [ Train | {epoch + 1:03d}/{total_epochs:03d} ] \
            mean_loss = {mean_loss:.5f}")

        # eval
        model.eval()
        val_acc = []
        y_pred_proba = []
        y_true = []

        for hsv_input, clip_input, targets in val_loader:
            hsv_input = hsv_input.to(device)
            clip_input = clip_input.to(device)
            targets = targets.to(device).float().unsqueeze(-1)
            
            with torch.no_grad():
                pred = model(hsv_input, clip_input, targets, train=False)
                y_pred_proba.extend(pred.squeeze(-1).cpu().numpy())
                y_true.extend(targets.squeeze(-1).cpu().numpy())
            
            pred_class = (pred > 0.5).long()
            accuracy = (pred_class == targets.long()).sum() / targets.size(0)
            val_acc.append(accuracy.item())
        
        mean_acc = sum(val_acc) / len(val_acc)
        current_time = datetime.datetime.now()
        print(f"{current_time:%Y-%m-%d %H:%M:%S} : \
              [ Evaluation | {epoch + 1:03d}/{total_epochs:03d} ] \
              mean_acc = {mean_acc:.5f}")

        if mean_acc >= best:
            best = mean_acc
            best_epoch = epoch
            torch.save({'model': model.state_dict()}, os.path.join(
                args.output_dir, 'checkpoint_best.pth'))

    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    print(
        f"Training time: {total_time}, Best epoch: {best_epoch + 1:03d}, Best acc: {best:.5f}")
    return


def classify(model, test_loader, device):
    model.eval()
    model_results = defaultdict(lambda: {"acc": [], "y_true": [], "y_pred": []})

    data_loader = tqdm(test_loader, desc="Testing", unit="batch")

    for batch in data_loader:
        hsv_input = batch['hsv_img'].to(device)
        clip_input = batch['clip_img'].to(device)
        targets = batch['label'].to(device).float().unsqueeze(-1)
        model_names = batch['model_name']
        with torch.no_grad():
            pred = model(hsv_input, clip_input, targets, train=False)
            pred_class = (pred > 0.5).long()

        for i, model_name in enumerate(model_names):
            acc = (pred_class[i] == targets[i].long()).item()
            model_results[model_name]["acc"].append(acc)
            model_results[model_name]["y_true"].append(targets[i].item())
            model_results[model_name]["y_pred"].append(pred[i].item())

    current_time = datetime.datetime.now()
    all_acc = []
    all_ap = []

    for model_name, res in model_results.items():
        mean_acc = sum(res["acc"]) / len(res["acc"])
        ap = average_precision_score(res["y_true"], res["y_pred"])
        all_acc.append(mean_acc)
        all_ap.append(ap)
        print(f"{current_time:%Y-%m-%d %H:%M:%S} : \
              [ Test | {model_name} ] mean_acc = {mean_acc:.4f}, AP = {ap:.4f}")
    
    mean_acc_all = sum(all_acc) / len(all_acc)
    mean_ap_all = sum(all_ap) / len(all_ap)
    print(f"{current_time:%Y-%m-%d %H:%M:%S} : \
          [ Mean | All Models ] mean_acc = {mean_acc_all:.4f}, AP = {mean_ap_all:.4f}")
    return

def main(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    same_seed(args.seed)

    print("Creating dataset")
    datasets = create_dataset(config)
    train_loader, val_loader, test_loader = create_loader(datasets,
                                                          batch_size=[
                                                              config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                                          is_trains=[True, False, False])
            
    print("Creating model")
    model = CANVAS()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    model = model.to(device)

    params = []
    for name, p in model.named_parameters():
        if "clip" in name: 
            p.requires_grad = False
        else:
            params.append(p) 
    
    optimizer = torch.optim.AdamW(params, lr=config['optimizer']['lr'],  
        weight_decay=config['optimizer']['weight_decay'])
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, 
        total_iters=config['warmup_epochs'] * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(config['total_epochs'] - config['warmup_epochs']) * len(train_loader),
        eta_min=config['min_lr'])

    if args.test is False:
        print("Start training")
        trainer(config, args, model, train_loader, val_loader, optimizer, scheduler, scheduler_warmup, device)
    else:
        print("Start testing")
        classify(model, test_loader, device)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/configs_cross.yaml')
    parser.add_argument('--output_dir', default='/data/lirunjie/code/output/')
    parser.add_argument('--checkpoint', default=False)
    parser.add_argument('--test', default=False)
    parser.add_argument('--seed', default=412, type=int)
    args = parser.parse_args()

    yaml = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    # 将配置文件保存到output_dir文件夹下
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config_cross.yaml'), 'w'))

    main(args, config)
