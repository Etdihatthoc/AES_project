import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import random
import logging
import os
import numpy as np
import wandb
import yaml
from sklearn.model_selection import train_test_split
from dataloader import SpeakingDatasetWav2Vec2, collate_fn
from model_new import MultimodalWav2VecScoreModel
from transformers import get_linear_schedule_with_warmup, Wav2Vec2Processor
from CELoss import SoftLabelCrossEntropyLoss
from tqdm.auto import tqdm
import gc 


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

wandb.login(key='b6bf189f51b29501771e7a3294635dfee6d75021', relogin=True)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def compute_class_weights(labels, num_classes=21):
    """
    Tính weight cho từng lớp dựa vào số lượng mẫu của lớp đó.
    Sử dụng công thức: weight[class] = total_samples / (num_classes * count[class])
    """
    print("Labels:", labels)
    counts = np.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    total = np.sum(counts)
    weights = total / (num_classes * counts)
    weights[np.isinf(weights)] = 0.0
    return torch.tensor(weights, dtype=torch.float)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_samples = 0
    running_loss = 0.0
    running_mae = 0.0
    with torch.no_grad():
        for audios, labels, texts in tqdm(val_loader): 
            with torch.amp.autocast('cuda'):
                logits = model(audios, texts)
                loss = criterion(logits, labels.to(device))
                
            running_loss += loss.item() * audios.size(0)
            total_samples += audios.size(0)
            
            preds = torch.argmax(logits, dim=1)
            preds_scores = preds.to(device).float() * 0.5
            true_scores = labels.to(device).float() * 0.5
            mae = torch.abs(preds_scores - true_scores).mean().item()
            running_mae += mae * audios.size(0)
            
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_mae = running_mae / total_samples if total_samples > 0 else 0
    return avg_loss, avg_mae

def train_model(model,  train_loader, val_loader, optimizer, criterion, device, 
                num_epochs, checkpoint_path, log_file, early_stop_patience, warmup_epochs, project, config):
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
    total_training_steps = num_epochs * steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_epochs * steps_per_epoch,
                    num_training_steps=total_training_steps
                )

    best_loss = float('inf')
    no_improve_epochs = 0  
    global_step = 0
    ckpt_dict = {}

    # Resume from epoch 10
    wandb.init(project=project, config=config)
    wandb.watch(model, log="all")
    
    scaler = torch.amp.GradScaler('cuda')

    # Các list lưu các metric qua từng epoch để vẽ biểu đồ
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    val_mae_list = []


    for epoch in range(num_epochs):
        total_samples = 0
        running_loss = 0.0
        for i, (mels, labels, texts) in enumerate(train_loader):
            model.train()
            with torch.amp.autocast('cuda'):
                optimizer.zero_grad()
                logits = model(mels, texts)
                loss = criterion(logits, labels.to(device))
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            scheduler.step()  # Cập nhật learning rate sau mỗi bước
            
            running_loss += loss.item() * mels.size(0)
            total_samples += mels.size(0)
            
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

        train_loss = running_loss / total_samples
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {train_loss:.4f}")
        
        val_loss, val_mae = evaluate(model,  val_loader, criterion, device)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] validation completed. Avg Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] validation completed. Avg Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        
        # Log metrics từng epoch
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae
        })
        
        # Cập nhật các list metric để vẽ biểu đồ
        epoch_list.append(epoch+1)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)
        
        # Sử dụng wandb.plot.line_series() để vẽ biểu đồ với 3 đường:
        wandb.log({
            "metrics_plot": wandb.plot.line_series(
                xs = epoch_list,
                ys = [train_loss_list, val_loss_list, val_mae_list],
                keys = ["train_loss", "val_loss", "val_mae"],
                title = f"Training Metrics Up to Epoch {epoch+1}",
                xname = "Epoch"
            )
        })
        
        if val_mae < best_loss:
            best_loss = val_mae
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_mae,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1} with validation loss {val_mae:.4f}")
            wandb.run.summary["best_val_mae"] = val_mae
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epoch(s)")
            if no_improve_epochs >= early_stop_patience:
                logger.info("Early stopping triggered.")
                print("Early stopping triggered.")
                ckpt_dict[str(epoch+1)] = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_mae,
                }
                torch.save(ckpt_dict, "SaveCKPTClassificationFLuency.pth")
                break

        ckpt_dict[str(epoch+1)] = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_mae,
        }
        torch.save(ckpt_dict, "SaveCKPTClassificationFLuency.pth")
        logger.info(f"Checkpoint for epoch {epoch+1} saved in SaveCKPTClassificationFLuency.pth")
                
    wandb.finish()
    return model


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed_everything(42)
    
    train_file = config["train_file"]
    val_file = config["val_file"]
    test_file = config["test_file"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    audio_encoder_id = config["audio_encoder_id"]
    learning_rate = config["learning_rate"]
    encoder_lr = config["encoder_lr"]
    sample_rate = config["sample_rate"]
    early_stop_patience = config["early_stop_patience"]
    warmup_epochs = config["warmup_epochs"]
    checkpoint_path = config["checkpoint_path"]
    log_file = config["log_file"]
    num_workers = config["num_workers"]
    istrain = config.get("istrain", True)
    load_pretrained = config.get("load_pretrained", False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    processor = Wav2Vec2Processor.from_pretrained(audio_encoder_id)
    # Khởi tạo dataset
    train_dataset = SpeakingDatasetWav2Vec2(csv_file = train_file, processor=processor, sample_rate=sample_rate, is_train=True)
    val_dataset = SpeakingDatasetWav2Vec2(csv_file = val_file, processor=processor, sample_rate=sample_rate, is_train=False)
    test_dataset = SpeakingDatasetWav2Vec2(csv_file = test_file, processor=processor, sample_rate=sample_rate, is_train=False)
    
        
    
    print("Số video trong train:", len(train_dataset))
    print("Số video trong validation:", len(val_dataset))
    print("Số video trong test:", len(test_dataset))
    
    # Thống kê số lượng mẫu cho từng nhãn
    train_df = train_dataset.df
    val_df = val_dataset.df
    
    print("Số lượng mẫu cho từng nhãn trong train:")
    print(train_df['fluency'].value_counts().sort_index())
    print("\nSố lượng mẫu cho từng nhãn trong validation:")
    print(val_df['fluency'].value_counts().sort_index())
    
    # Tính class weights cho tập train
    labels= train_df['fluency'].apply(lambda s: int(round(s / 0.5)))
    train_scores = labels.values
    class_weights = compute_class_weights(train_scores, num_classes=21)
    print("Class Weights:", class_weights)
    
    train_sampler = WeightedRandomSampler(np.ones(len(train_dataset)), num_samples=len(train_dataset), replacement=True)
    val_sampler = WeightedRandomSampler(np.ones(len(val_dataset)), num_samples=len(val_dataset), replacement=True)
    test_sampler = WeightedRandomSampler(np.ones(len(test_dataset)), num_samples=len(test_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler = val_sampler, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler = test_sampler, collate_fn=collate_fn, num_workers=num_workers)
    
    model = MultimodalWav2VecScoreModel(audio_encoder_id = audio_encoder_id, device = device)
    print("Model created. Fusion dimension =", model.fc[0].in_features)
    
    if load_pretrained:
        ckpt = torch.load(config['pretrained_path'])
        model.load_state_dict(ckpt['model_state_dict'])
        print("Checkpoint loaded from:", config['pretrained_path'])
        
        del ckpt 
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleared cache after loading checkpoint.")
    
    # Phân nhóm tham số
    encoder_params = list(model.audio_encoder.parameters()) + list(model.text_encoder.parameters())
    other_params = [param for name, param in model.named_parameters() 
                    if not (name.startswith("audio_encoder") or name.startswith("text_encoder"))]

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 1e-2},
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 1e-2}
    ])
    
    # Sử dụng CrossEntropyLoss với class weights
    criterion = SoftLabelCrossEntropyLoss(num_classes=21,class_weight=class_weights.to(device)).to(device)
    
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                         num_epochs=num_epochs, 
                         checkpoint_path=checkpoint_path, 
                         log_file=log_file, 
                         early_stop_patience=early_stop_patience, 
                         warmup_epochs=warmup_epochs,
                         project=config["project"],
                         config=config)
    
    logging.basicConfig(level=logging.INFO, 
                        filename=log_file, 
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    logger.info(f"Final Test - CrossEntropy Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
    print(f"Final Test - CrossEntropy Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")

if __name__ == '__main__':
    main()
