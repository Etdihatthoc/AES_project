import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random
import logging
import os
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from transformers import Wav2Vec2Processor
from dataloader import SpeakingDatasetWav2Vec2, ChunkedSpeakingDataset, collate_fn  # Sử dụng collate_fn_eval cho infer (mỗi sample là list các chunk)
# from model import MultimodalWhisperScoreModel  # Sử dụng model này theo yêu cầu
from model_new import MultimodalWav2VecScoreModel

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


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for audios, scores, texts in tqdm(data_loader):
        
            # audios = audios.to(device, non_blocking=True)
            # scores = scores.to(device, non_blocking=True)
            outputs = model(audios, texts)
            
            preds = torch.argmax(outputs, dim=1)
            preds_scores = preds.float() * 0.5
            true_scores = scores.to(device).float() * 0.5
            
            for pred, true in zip(preds_scores, true_scores):
                predictions.append(pred.item())
                ground_truths.append(true.item())
            
    return predictions, ground_truths


def save_results_csv(predictions, ground_truths, output_csv_path='results.csv'):
    """
    Lưu kết quả dự đoán và ground truth vào file CSV theo mẫu:
      - GroundTruth, Predicted, score_rounded (làm tròn dự đoán đến bội số 0.5)
      - error_range = GroundTruth - score_rounded
    """
    df = pd.DataFrame({
        'GroundTruth': ground_truths,
        'Predicted': predictions
    })
    df['score_rounded'] = np.round(df['Predicted'] * 2) / 2.0
    df['error_range'] = df['GroundTruth'] - df['score_rounded']
    
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

def weighted_average_checkpoint(ckpt_path, loss_threshold=0.4):
    """
    Tính trung bình trọng số của các checkpoint có val_loss < loss_threshold.

    Args:
        ckpt_path (str): Đường dẫn file checkpoint đã lưu (chứa dictionary với key là epoch).
        loss_threshold (float): Ngưỡng loss để chọn checkpoint (mặc định 0.4).

    Returns:
        avg_state_dict (dict): State dict trung bình trọng số.
        selected_epochs (list): Danh sách các epoch được sử dụng trong tính toán.
    """
    ckpt_dict = torch.load(ckpt_path, map_location='cpu')
    selected_ckpts = {}
    for epoch, ckpt in ckpt_dict.items():
        # Kiểm tra ckpt có phải là dictionary và có key 'loss'
        if isinstance(ckpt, dict) and 'loss' in ckpt and ckpt['loss'] < loss_threshold:
            selected_ckpts[epoch] = ckpt

    if not selected_ckpts:
        raise ValueError("Không tìm thấy checkpoint nào với val_loss < {}".format(loss_threshold))

    total_weight = 0.0
    avg_state_dict = {}
    selected_epochs = []
    
    for epoch, ckpt in selected_ckpts.items():
        loss = ckpt['loss']
        # Định nghĩa trọng số: 1/loss
        weight = 1.0 / loss
        total_weight += weight
        selected_epochs.append(epoch)
        state_dict = ckpt['model_state_dict']
        for key, param in state_dict.items():
            # Chuyển sang float để tính toán
            param = param.float()
            if key not in avg_state_dict:
                avg_state_dict[key] = weight * param.clone()
            else:
                avg_state_dict[key] += weight * param.clone()
    
    # Chia cho tổng trọng số để tính trung bình trọng số
    for key in avg_state_dict:
        avg_state_dict[key] /= total_weight

    return avg_state_dict, selected_epochs

def main():
    # Đọc config từ file YAML
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    seed_everything(42)
    
    # Lấy các tham số từ config
    csv_file = config["csv_file"]
    batch_size = config["batch_size"]
    audio_encoder_id = config["audio_encoder_id"]
    sample_rate = config["sample_rate"]
    checkpoint_path = config["checkpoint_path"]
    log_file = config["log_file"]
    num_workers = config["num_workers"]
    
    # Cấu hình logger: ghi log vào file và console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    logger = logging.getLogger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("Device:", device)
    logger.info(f"Device: {device}")
    
    timestamp_folder = 'audio_chunks'
    processor = Wav2Vec2Processor.from_pretrained(audio_encoder_id)
    # Load dataset
    # Tạo dataset cho train với augment, và cho val/test không augment
    # Khởi tạo dataset
    train_dataset = SpeakingDatasetWav2Vec2(csv_file = csv_file, processor=processor, sample_rate=sample_rate, is_train=True)
    val_dataset = SpeakingDatasetWav2Vec2(csv_file = csv_file, processor=processor, sample_rate=sample_rate, is_train=False)
    test_dataset = SpeakingDatasetWav2Vec2(csv_file = csv_file, processor=processor, sample_rate=sample_rate, is_train=False)
    
    print("Số video trong dataset:", len(train_dataset))
    
    # Chia dữ liệu theo stratify dựa trên nhãn pronunciation (giả sử các nhãn đã được làm tròn theo bước 0.5)
    df = train_dataset.df
    labels = df['pronunciation']
    indices = np.arange(len(train_dataset))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    temp_labels = labels.iloc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2/3,
        stratify=temp_labels,
        random_state=42
    )

    test_set = Subset(test_dataset, test_idx)
    
    print("Số video trong test:", len(test_set))
    logger.info(f"Số video trong test: {len(test_set)}")
    
    # Tạo DataLoader cho tập test sử dụng collate_fn_eval
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    # Khởi tạo model và đưa vào device
    model = MultimodalWav2VecScoreModel(audio_encoder_id=audio_encoder_id, device=device)
    print("hidden_dim:", model.fc[0].in_features)
    model.to_device(device)
    
    # # load ckpt base on SWA
    # avg_state_dict, selected_epochs = weighted_average_checkpoint("/home/user01/aiotlab/sondinh/AES_project/multimodal_classification/SaveCKPT_classification.pth", loss_threshold=0.75)
    # model.load_state_dict(avg_state_dict)
    # print("Weighted average checkpoint loaded from epochs:", selected_epochs)
    
    
    checkpoint_path = "SaveCKPT_classification.pth"

    # Load toàn bộ dictionary checkpoint
    ckpt_dict = torch.load(checkpoint_path)

    # Lấy model_state_dict của epoch 10
    epoch = 10 
    if str(epoch) in ckpt_dict:
        model_state_dict_epoch = ckpt_dict[str(epoch)]['model_state_dict']
        print(f"Đã load model_state_dict của epoch {epoch}")
    
    model.load_state_dict(model_state_dict_epoch)
    print("Checkpoint loaded from:", checkpoint_path)
    
    # # Load checkpoint từ checkpoint_path
    # ckpt = torch.load(checkpoint_path)
    # model.load_state_dict(ckpt['model_state_dict'])
    # print("Checkpoint loaded from:", checkpoint_path)
    # logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    # Evaluate model
    predictions, ground_truths = evaluate(model, test_loader, device)
    
    # Tính các metric
    preds_np = np.array(predictions)
    truths_np = np.array(ground_truths)
    
    smooth_l1_loss_fn = nn.SmoothL1Loss(reduction='mean')
    smooth_l1_loss = smooth_l1_loss_fn(torch.tensor(preds_np), torch.tensor(truths_np)).item()
    
    l1_loss_fn = nn.L1Loss(reduction='mean')
    l1_loss = l1_loss_fn(torch.tensor(preds_np), torch.tensor(truths_np)).item()
    
    mse_loss_fn = nn.MSELoss(reduction='mean')
    mse_loss = mse_loss_fn(torch.tensor(preds_np), torch.tensor(truths_np)).item()
    
    r2 = r2_score(truths_np, preds_np)
    
    # Ghi các metric vào file log và in ra console
    logger.info("Evaluation Metrics on Test Set:")
    logger.info(f"Smooth L1 Loss: {smooth_l1_loss:.4f}")
    logger.info(f"L1 Loss : {l1_loss:.4f}")
    logger.info(f"MSE Loss: {mse_loss:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")
    
    print("Evaluation Metrics on Test Set:")
    print(f"Smooth L1 Loss: {smooth_l1_loss:.4f}")
    print(f"L1 Loss: {l1_loss:.4f}")
    print(f"MSE Loss: {mse_loss:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Lưu kết quả dự đoán và ground truth ra file CSV
    save_results_csv(predictions, ground_truths, output_csv_path='results_pronunciation.csv')


if __name__ == '__main__':
    main()
