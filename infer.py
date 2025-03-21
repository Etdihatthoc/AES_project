
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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_and_plot(model, data_loader, criterion, device, combined_plot_path='plot_pronunciation_non_round.png'):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_scores = []
    
    with torch.no_grad():
        for mels, scores in data_loader:
            mels = mels.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            outputs = model(mels)
            loss = criterion(outputs, scores)
            running_loss += loss.item()
            # Lưu kết quả đầu ra và nhãn thật
            all_outputs.append(outputs.detach().cpu())
            all_scores.append(scores.detach().cpu())
            
    avg_loss = running_loss / len(data_loader.dataset)
    
    # Ghép các tensor lại thành 1 tensor duy nhất, chuyển sang numpy để vẽ biểu đồ
    all_outputs1 = torch.cat(all_outputs, dim=0).numpy()
    
    all_outputs = np.round(all_outputs1 * 2) / 2.0
    
    #all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    
    # Giả sử outputs và scores là các giá trị scalar (1D arrays)
    # Sắp xếp theo ground truth (all_scores)
    sort_indices = np.argsort(all_scores.flatten())
    sorted_scores = all_scores.flatten()[sort_indices]
    sorted_outputs = all_outputs.flatten()[sort_indices]
    
    # Tạo figure với 2 subplots trong 1 file
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Biểu đồ đầu tiên: Giá trị dự đoán (outputs) sau khi sắp xếp
    axs[0].plot(sorted_outputs, label='Predicted', marker='o', linestyle='-', markersize=3)
    axs[0].set_title('Predicted Outputs (Sorted by Ground Truth)')
    axs[0].set_xlabel('Sample Index (Sorted)')
    axs[0].set_ylabel('Output Value')
    axs[0].legend()
    
    # Biểu đồ thứ hai: Giá trị nhãn thật (scores) sau khi sắp xếp
    axs[1].plot(sorted_scores, label='Ground Truth', color='orange', marker='o', linestyle='-', markersize=3)
    axs[1].set_title('Ground Truth Scores (Sorted)')
    axs[1].set_xlabel('Sample Index (Sorted)')
    axs[1].set_ylabel('Score')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.show()
    
    return avg_loss, sorted_outputs, sorted_scores

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
    
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    # Initialize model
    model = WhisperScoreModel(model_size=model_size)
    print("hidden_dim:", model.fc[0].in_features)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load("/mnt/disk1/SonDinh/Project/ckpt/ckpt_pronunciation/ckpt_en_AdamW_L1_tiny_3fc_rawdata.pth", map_location=device)

    #Nạp state_dict của model từ checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    
    criterion1 = nn.SmoothL1Loss().to(device)
    criterion2 = nn.L1Loss().to(device)
    criterion3 = nn.MSELoss().to(device)
    #                      )
    
    # Test Evaluation Phase
    logging.basicConfig(level=logging.INFO, 
                        filename=log_file, 
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    avg_loss, all_outputs, all_scores = evaluate_and_plot(model, test_loader, criterion1, device, combined_plot_path='plot_pronunciation_non_round.png')
    logger.info(f"Final Test Loss L1 smooth: {avg_loss:.4f}")
    print(f"Final Test Loss: {avg_loss:.4f}")
    
    avg_loss, all_outputs, all_scores = evaluate_and_plot(model, test_loader, criterion2, device, combined_plot_path='plot_pronunciation_non_round.png')
    logger.info(f"Final Test Loss L1: {avg_loss:.4f}")
    print(f"Final Test Loss: {avg_loss:.4f}")
    
    avg_loss, all_outputs, all_scores = evaluate_and_plot(model, test_loader, criterion3, device, combined_plot_path='plot_pronunciation_non_round.png')
    logger.info(f"Final Test Loss MSE: {avg_loss:.4f}")
    print(f"Final Test Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()

