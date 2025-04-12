import torch
import torch.utils.data as data
import pandas as pd
import librosa
import whisper
import random
import nlpaug.augmenter.audio as naa
import numpy as np

def fixed_chunk_audio(audio, sr, num_chunks=10, chunk_length_sec=30):
    """
    Cắt audio thành đúng num_chunks với độ dài mỗi chunk = chunk_length_sec.
    Nếu audio quá ngắn, pad thêm zeros.
    Nếu audio đủ dài, lấy các chunk cách đều trên audio (có thể overlap).
    """
    chunk_samples = int(chunk_length_sec * sr)
    audio_length = len(audio)
    if audio_length < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - audio_length), mode='constant')
        audio_length = len(audio)
    # Tính các chỉ số bắt đầu để chia thành num_chunks đều nhau
    if num_chunks == 1:
        starts = [0]
    else:
        max_start = audio_length - chunk_samples
        starts = np.linspace(0, max_start, num_chunks, dtype=int)
    chunks = []
    for start in starts:
        end = start + chunk_samples
        chunk = audio[start:end]
        chunks.append(chunk)
    return chunks

def pad_or_trim_tensor(tensor, length=3000):
    """
    Pad hoặc cắt tensor theo chiều time (chỉ số thứ 1) để đảm bảo có độ dài cố định.
    Input tensor có shape (80, T).
    """
    current_len = tensor.shape[1]
    if current_len < length:
        pad = torch.zeros(tensor.shape[0], length - current_len, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    else:
        return tensor[:, :length]

class SpeakingDataset(data.Dataset):
    def __init__(self, csv_file, sample_rate=16000, chunk_length_sec=30, target_length=3000, num_chunks=10, is_train=True):
        """
        csv_file: Đường dẫn file CSV chứa các cột 'absolute_path', 'pronunciation' và 'text'
        sample_rate: Tốc độ mẫu của audio (16kHz)
        chunk_length_sec: Độ dài của mỗi chunk (giây)
        target_length: Số frame mong muốn của mel spectrogram (ví dụ: 3000)
        num_chunks: Số chunk cố định mỗi sample
        is_train: Nếu True thì áp dụng augment data, ngược lại không augment.
        """
        augment_csv = "augmented_data.csv"
        df_aug = pd.read_csv(augment_csv)
        df_raw = pd.read_csv(csv_file)
        
        self.df = pd.concat([df_raw, df_aug], ignore_index=True)
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        self.target_length = target_length
        self.num_chunks = num_chunks
        self.is_train = is_train
        
        # Khởi tạo các augmenter (áp dụng với xác suất 50% nếu is_train)
        self.noise_aug = naa.NoiseAug(name='NoiseAug')
        self.speed_aug = naa.SpeedAug(name='SpeedAug', factor=(0.9, 1.1))
        self.pitch_aug = naa.PitchAug(sampling_rate=sample_rate, name='PitchAug', factor=(0.7, 0.9))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['absolute_path'].replace('/mnt/son_usb/DATA_Vocal', '/media/gpus/Data/DATA_Vocal')
        score = row['pronunciation']  # điểm theo thang 0-10, với bước nhảy 0.5
        transcript = row['text'] 

        # Tính label cho bài toán classification: chuyển score thành index (0 -> 0, 0.5 -> 1, ..., 10 -> 20)
        label = int(round(score / 0.5))

        # 1. Load audio với sample_rate cho trước
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Áp dụng augment nếu đang train
        if self.is_train:
            if random.random() < 0.7:
                audio = self.noise_aug.augment(audio)[0]
            if random.random() < 0.7:
                audio = self.speed_aug.augment(audio)[0]
            if random.random() < 0.7:
                audio = self.pitch_aug.augment(audio)[0]
            
        # 2. Cắt audio thành đúng num_chunks cố định
        audio_chunks = fixed_chunk_audio(audio, sr, num_chunks=self.num_chunks, chunk_length_sec=self.chunk_length_sec)
        
        # 3. Với mỗi chunk: tính log-mel spectrogram và pad/trim về target_length
        mel_chunks = []
        for chunk in audio_chunks:
            min_length = 400
            if len(chunk) < min_length:
                pad_length = min_length - len(chunk)
                chunk = np.pad(chunk, (0, pad_length), mode='constant')
            mel_chunk = whisper.log_mel_spectrogram(chunk)  # shape: (80, T_chunk)
            final_mel = pad_or_trim_tensor(mel_chunk, length=self.target_length)  # shape: (80, target_length)
            mel_chunks.append(final_mel)
            
        # Stack các mel của chunk thành tensor có shape (num_chunks, 80, target_length)
        mel_tensor = torch.stack(mel_chunks)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mel_tensor, label_tensor, transcript

def collate_fn(batch):
    """
    Collate function cho training/evaluation.
    Batch là list các tuple: (mel_tensor, label_tensor, transcript)
    Trả về:
      - mels_tensor: Tensor shape (batch_size, num_chunks, 80, target_length)
      - labels_tensor: Tensor shape (batch_size,)
      - texts_list: List các string, độ dài = batch_size
    """
    mels, labels, texts = zip(*batch)
    mels_tensor = torch.stack(mels, dim=0)
    labels_tensor = torch.stack(labels, dim=0)
    texts_list = list(texts)
    return mels_tensor, labels_tensor, texts_list
