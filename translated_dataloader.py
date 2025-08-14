import torch
import torch.utils.data as data
import pandas as pd
import librosa
import random
import nlpaug.augmenter.audio as naa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import os
import json

# model = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

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

class TranslatedSpeakingDatasetWav2Vec2(data.Dataset):
    def __init__(self, csv_file, processor, sample_rate=16000, chunk_length_sec=30, target_length=3000, num_chunks=10, is_train=True):
        """
        csv_file: Đường dẫn file CSV chứa các cột 'absolute_path', 'fluency' và 'text'
        sample_rate: Tốc độ mẫu của audio (16kHz)
        chunk_length_sec: Độ dài của mỗi chunk (giây)
        target_length: Số frame mong muốn của mel spectrogram (ví dụ: 3000)
        num_chunks: Số chunk cố định mỗi sample
        is_train: Nếu True thì áp dụng augment data, ngược lại không augment.
        """
        df_aug = pd.read_csv(csv_file)
        df_raw = pd.read_csv(csv_file)
        
        if is_train:
            self.df = pd.concat([df_raw, df_aug], ignore_index=True)
        else: 
            self.df = df_raw
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        self.target_length = target_length
        self.num_chunks = num_chunks
        self.is_train = is_train
        self.processor = processor
        
        # Khởi tạo các augmenter (áp dụng với xác suất 50% nếu is_train)
        self.noise_aug = naa.NoiseAug(name='NoiseAug')
        self.speed_aug = naa.SpeedAug(name='SpeedAug', factor=(0.9, 1.1))
        self.pitch_aug = naa.PitchAug(sampling_rate=sample_rate, name='PitchAug', factor=(0.7, 0.9))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['absolute_path'].replace('/mnt/son_usb/DATA_Vocal', '/media/gpus/Data/DATA_Vocal')
        score = row['grammar']  # điểm theo thang 0-10, với bước nhảy 0.5
        transcript = row['text']
        translated_script = row['translated_text']  
    
        # Tính label cho bài toán classification: chuyển score thành index (0 -> 0, 0.5 -> 1, ..., 10 -> 20)
        label = int(round(score / 0.5))
    
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
    
        # Apply augmentation if training
        if self.is_train:
            if random.random() < 0.7:
                audio = self.noise_aug.augment(audio)[0]
            if random.random() < 0.7:
                audio = self.speed_aug.augment(audio)[0]
            if random.random() < 0.7:
                audio = self.pitch_aug.augment(audio)[0]
        
        # Cut into chunks
        audio_chunks = fixed_chunk_audio(audio, sr, num_chunks=self.num_chunks, chunk_length_sec=self.chunk_length_sec)
        
        for i in range(len(audio_chunks)):
            # print (f"Shape of audio chunk {i}: {audio_chunks[i].shape}")
            inputs = self.processor(audio_chunks[i], sampling_rate=self.sample_rate, return_tensors="pt")
            audio_chunks[i] = inputs.input_values.squeeze(0)  # Chuyển về tensor 1 chiều (80, T)
            # audio_chunks[i] = pad_or_trim_tensor(audio_chunks[i], length=self.target_length)
            # print (f"Shape of audio chunk {i} after padding: {audio_chunks[i].shape}")
    
        # Pad chunks to the same length (needed for batching!)
        chunk_samples = int(self.chunk_length_sec * self.sample_rate)
        padded_chunks = []
        for chunk in audio_chunks:
            min_length = 400 
            if len(chunk) < chunk_samples:
                pad_length = chunk_samples - len(chunk)
                chunk = np.pad(chunk, (0, pad_length), mode='constant')
            else:
                chunk = chunk[:chunk_samples]
            padded_chunks.append(chunk.to(torch.float32))  # Wav2Vec2 expects float32
    
        # Stack into tensor: shape (num_chunks, chunk_samples)
        audio_tensor = torch.stack(padded_chunks)
    
        label_tensor = torch.tensor(label, dtype=torch.long)

        return audio_tensor, label_tensor, transcript, translated_script


def translated_collate_fn(batch):
    all_chunks, labels, texts, translated_script = zip(*batch)  # all_chunks is list of list[tensor]

    # Stack chunks for each sample separately
    padded_audios = []
    for chunk_list in all_chunks:
        # padded = [pad_or_trim_tensor(chunk.unsqueeze(0)).squeeze(0) for chunk in chunk_list]
        padded = [chunk for chunk in chunk_list]
        padded_audios.append(torch.stack(padded))  # shape: (num_chunks, T)

    # Now stack batch
    audios_tensor = torch.stack(padded_audios)  # shape: (batch_size, num_chunks, T)
    labels_tensor = torch.stack(labels, dim = 0)
    return audios_tensor, labels_tensor, list(texts), list(translated_script)



def main():
    # Example usage
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    dataset = TranslatedSpeakingDatasetWav2Vec2(csv_file='new_full_train_greaterthan3_removenoise_fixspellingerror_translated.csv',
                                     processor=processor)
    
    
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=translated_collate_fn)

    for batch in dataloader:
        audio_tensor, label_tensor, texts_list, translated_texts_list = batch
        print(label_tensor)
        # print(f"Audio tensor shape: {audio_tensor.shape}")
        break 

if __name__ == "__main__":
    main()