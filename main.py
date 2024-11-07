import argparse, os, yaml, shutil
from pprint import PrettyPrinter
from dotmap import DotMap
import wandb
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from utils.solver import _lr_scheduler, EarlyStopping
from modules.video_clip import VideoCLIP
from dataset.oad_dataset import ActionDataset
from utils.saving import save_epoch, save_best
from utils.evaluation import evaluate

def get_config():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--log_time", type=str, default=None, help="Current time for logging purposes")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.full_load(f)
  
    config['working_dir'] = os.path.join("./exp", config['name'], args.log_time)
    # Log config
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(config['working_dir']))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)
    
    config = DotMap(config)

    # Set the working directory
    Path(config.working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, config.working_dir)
    shutil.copy("main.py", config.working_dir)

    # Set wandb
    wandb.init(project="TOAD",
                name="{}_{}".format(config.name, args.log_time),
                config=config)

    return config


def main():
    wandb.require("core")
    config = get_config()

    # Set the seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("===== WARNING =====")
        print("Running on CPU")
        print("==================")
        wandb.alert("Running on CPU")
        import sys; sys.exit(1)

    train_ds = ActionDataset(config, "train")
    test_ds = ActionDataset(config, "test")
    
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, drop_last=True)


    model = VideoCLIP(config).to(device)


    start_epoch = 0
    best = 0.0
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            del checkpoint
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(config.pretrain))

    optimizer = optim.AdamW(params= model.parameters(),
                            betas=(0.9, 0.999), lr=config.solver.lr, eps=1e-8,
                            weight_decay=config.solver.weight_decay) 
    
    lr_scheduler = _lr_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler()

    criterion = torch.nn.CrossEntropyLoss()
    fut_criterion = torch.nn.CrossEntropyLoss() if config.data.future_steps > 0 else None

    metric = "mAP" if config.data.dataset == "THUMOS14" else "mcAP"

    if config.eval:
        print("===========evaluate===========")
        res = evaluate(model, test_loader, device)[metric]
        print(f"{metric}: {res}")
        wandb.log({metric: res})
        wandb.log({f"best_{metric}": res})
        wandb.finish()
        return

    for epoch in range(start_epoch, config.solver.epochs):
        model.train()
        autocast = torch.cuda.amp.autocast
        for i, (rgb, label, fut_label) in enumerate(tqdm(train_loader)):
            if i % config.logging.freq == 0:
                wandb.log({"epoch": epoch, "batch": i})
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))

            rgb = rgb.to(device)
            label = label.to(device)

            with autocast():
                logits, fut_logits = model(rgb)
                actual_loss = criterion(logits, label)
                if config.data.future_steps > 0:
                    fut_loss = fut_criterion(fut_logits, fut_label.to(device))
                    loss = actual_loss + model.future_relevance * fut_loss
                else:
                    loss = actual_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


            if i % config.logging.freq == 0:
                wandb.log({"future_relevance":model.future_relevance.item(),"loss": loss.item(), "actual_loss": actual_loss.item(), "fut_loss": fut_loss.item() if config.data.future_steps > 0 else 0.0,"lr": optimizer.param_groups[0]['lr']})
                print(f"[{epoch}/{config.solver.epochs}] Loss: {loss.item()} ({actual_loss.item()} | {fut_loss.item() if config.data.future_steps > 0 else 0.0})")



        if (epoch+1) % config.solver.eval_freq == 0:
            print(f"[{epoch}/{config.solver.epochs}] Saving epoch...")
            save_epoch(epoch, model, optimizer, config.working_dir)
            res = evaluate(model, test_loader, device)[metric]
            wandb.log({metric: res})

            if res > best:
                best = res
                save_best(config.working_dir, epoch)
                wandb.run.summary["best_epoch"] = epoch
                print(f"[{epoch}/{config.solver.epochs}] New best {metric}: {best}")
            wandb.log({f"best_{metric}": best})

    wandb.finish()

if __name__ == '__main__':
    main()

