import torch, os
import shutil

def save_epoch(epoch, model, optimizer, work_dir):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                os.path.join(work_dir, "last_epoch.pt"))

def save_best(working_dir, epoch):
    shutil.copy(os.path.join(working_dir, "last_epoch.pt"), os.path.join(working_dir, f'best_epoch_{epoch}.pt'))