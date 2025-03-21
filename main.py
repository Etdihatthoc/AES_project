import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random
import logging
import os
import numpy as np
import wandb
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataloader import SpeakingDataset, collate_fn
from model import WhisperScoreModel
from torch.utils.data import Subset
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Đăng nhập wandb
wandb.login(
    key='072fb112587c6b4507f5ec59e575d234c3e22649',
    relogin=True
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_mel = 0
    running_loss = 0.0
    with torch.no_grad():
        for mels, scores in data_loader:
            mels = mels.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            outputs = model(mels)
            loss = criterion(outputs, scores)
            running_loss += loss.item() * mels.size(0)
            total_mel += mels.size(0)
    avg_loss = running_loss / total_mel
    return avg_loss

def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_epochs, checkpoint_path, log_file, early_stop_patience, warmup_epochs,project,
                         config):
    # Setup logger
    logging.basicConfig(level=logging.INFO, 
                        filename=log_file, 
                        filemode='w', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Start training...")
    print("Start training...")

    model.to(device)
    model.train()

    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch

    # Warmup scheduler: tăng dần LR trong giai đoạn warmup
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps))
    
    # Plateau scheduler: giảm LR khi loss validation không cải thiện
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.1, patience=3)
    
    best_loss = float('inf')
    no_improve_epochs = 0  
    global_step = 0

    # Initialize wandb run
    wandb.init(project=project, config=config)
    wandb.watch(model, log="all")
    
    for epoch in range(num_epochs):
        total_mel = 0
        running_loss = 0.0
        for i, (mels, scores) in enumerate(train_loader):
            mels = mels.to(device)
            scores = scores.to(device)
            
            outputs = model(mels)
            loss = criterion(outputs, scores)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            # Warmup LR update
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            
            running_loss += loss.item() * mels.size(0)
            total_mel += mels.size(0)
            
            # Log every 10 steps
            if (i + 1) % 10 == 0:
                lr_group0 = optimizer.param_groups[0]['lr']
                lr_group1 = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None

                log_message = (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{steps_per_epoch}], "
                               f"Loss: {loss.item():.4f}, LR_group0: {lr_group0:.6f}" +
                               (f", LR_group1: {lr_group1:.6f}" if lr_group1 is not None else ""))
                logger.info(log_message)
                print(log_message)
                wandb.log({
                    "train_step_loss": loss.item(),
                    "lr": lr_group0,
                    "global_step": global_step
                })

        train_loss = running_loss / total_mel
        train_epoch_msg = f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {train_loss:.4f}"
        logger.info(train_epoch_msg)
        print(train_epoch_msg)
        
        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        val_epoch_msg = f"Epoch [{epoch+1}/{num_epochs}] validation completed. Average Loss: {val_loss:.4f}"
        logger.info(val_epoch_msg)
        print(val_epoch_msg)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        if epoch >= warmup_epochs:
            plateau_scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1} with validation loss {val_loss:.4f}")
            wandb.run.summary["best_val_loss"] = val_loss
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epoch(s)")
            if no_improve_epochs >= early_stop_patience:
                logger.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break
                
    wandb.finish()
    return model
def main():
    # Đọc config từ file YAML
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed_everything(42)
    
    # Cập nhật config từ YAML
    csv_file = config["csv_file"]
    model_size = config["model_size"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    sample_rate = config["sample_rate"]
    early_stop_patience = config["early_stop_patience"]
    warmup_epochs = config["warmup_epochs"]
    checkpoint_path = config["checkpoint_path"]
    log_file = config["log_file"]
    num_workers = config["num_workers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Initialize dataset
    dataset = SpeakingDataset(csv_file=csv_file, sample_rate=sample_rate)
    print("Số video trong dataset:", len(dataset))
    
    # Stratified split dựa trên cột 'pronunciation'
    
    df = dataset.df
    labels = df['pronunciation']
    indices = np.arange(len(dataset))

    # Tách 70% cho train và 30% cho temp
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    # Tách temp thành validation (1/3) và test (2/3)
    temp_labels = labels.iloc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2/3,
        stratify=temp_labels,
        random_state=42
    )

    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    print("Số video trong train:", len(train_set))
    print("Số video trong validation:", len(val_set))
    print("Số video trong test:", len(test_set))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    # Initialize model
    model = WhisperScoreModel(model_size=model_size)
    print("hidden_dim:", model.fc[0].in_features)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load("/mnt/disk1/SonDinh/Project/ckpt/ckpt_pronunciation/ckpt_en_AdamW_L1_tiny_3fc_data_2_continu.pth", map_location=device)

    #Nạp state_dict của model từ checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Sử dụng hai param groups: encoder và fc
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config["encoder_lr"]},
        {'params': model.fc.parameters(), 'lr': learning_rate}
    ])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    criterion = nn.SmoothL1Loss().to(device)
    
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                         num_epochs=num_epochs, 
                         checkpoint_path=checkpoint_path, 
                         log_file=log_file, 
                         early_stop_patience=early_stop_patience, 
                         warmup_epochs=warmup_epochs,
                         project=config["project"],
                         config=config
                         )
    
    # Test Evaluation Phase
    logging.basicConfig(level=logging.INFO, 
                        filename=log_file, 
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info(f"Final Test Loss L1 smooth: {test_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()
