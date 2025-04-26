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

class SpeakingDatasetWav2Vec2(data.Dataset):
    def __init__(self, csv_file, processor, sample_rate=16000, chunk_length_sec=30, target_length=3000, num_chunks=10, is_train=True):
        """
        csv_file: Đường dẫn file CSV chứa các cột 'absolute_path', 'pronunciation' và 'text'
        sample_rate: Tốc độ mẫu của audio (16kHz)
        chunk_length_sec: Độ dài của mỗi chunk (giây)
        target_length: Số frame mong muốn của mel spectrogram (ví dụ: 3000)
        num_chunks: Số chunk cố định mỗi sample
        is_train: Nếu True thì áp dụng augment data, ngược lại không augment.
        """
        df_aug = pd.read_csv(csv_file)
        df_raw = pd.read_csv(csv_file)
        
        self.df = pd.concat([df_raw, df_aug], ignore_index=True)
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
        score = row['pronunciation']  # điểm theo thang 0-10, với bước nhảy 0.5
        transcript = row['text']
    
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

        return audio_tensor, label_tensor, transcript


# def collate_fn(batch):
#     """
#     Collate function cho training/evaluation.
#     Batch là list các tuple: (mel_tensor, label_tensor, transcript)
#     Trả về:
#       - mels_tensor: Tensor shape (batch_size, num_chunks, 80, target_length)
#       - labels_tensor: Tensor shape (batch_size,)
#       - texts_list: List các string, độ dài = batch_size
#     """
#     audios, labels, texts = zip(*batch)
#     audios_tensor = torch.stack(audios, dim=0)
#     labels_tensor = torch.stack(labels, dim=0)
#     texts_list = list(texts)
#     return audios_tensor, labels_tensor, texts_list

def collate_fn(batch):
    all_chunks, labels, texts = zip(*batch)  # all_chunks is list of list[tensor]

    # Stack chunks for each sample separately
    padded_audios = []
    for chunk_list in all_chunks:
        # padded = [pad_or_trim_tensor(chunk.unsqueeze(0)).squeeze(0) for chunk in chunk_list]
        padded = [chunk for chunk in chunk_list]
        padded_audios.append(torch.stack(padded))  # shape: (num_chunks, T)

    # Now stack batch
    audios_tensor = torch.stack(padded_audios)  # shape: (batch_size, num_chunks, T)
    labels_tensor = torch.stack(labels, dim = 0)
    return audios_tensor, labels_tensor, list(texts)


class ChunkedSpeakingDataset(data.Dataset):
    def __init__(self, 
                 csv_file: str, 
                 timestamp_folder: str,
                 processor,  # Wav2Vec2Processor
                 is_train: bool = True,
                 sample_rate: int = 16000, 
                 max_input_length: int = 480000,  # 30s * 16000Hz
                 max_label_length: int = 256):
        self.df_info = pd.read_csv(csv_file)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length
        self.timestamp_folder = timestamp_folder
        self.is_train = is_train
        
        # Khởi tạo các augmenter (áp dụng với xác suất 50% nếu is_train)
        self.noise_aug = naa.NoiseAug(name='NoiseAug')
        self.speed_aug = naa.SpeedAug(name='SpeedAug', factor=(0.9, 1.1))
        self.pitch_aug = naa.PitchAug(sampling_rate=sample_rate, name='PitchAug', factor=(0.7, 0.9))

        self._build_chunked_dataset()

    def _build_chunked_dataset(self):
        self.df = []
        for idx, row in self.df_info.iterrows():
            audio_path = row['absolute_path'].replace('/mnt/son_usb/DATA_Vocal', '/media/gpus/Data/DATA_Vocal')
            score = row['pronunciation']  # điểm theo thang 0-10, với bước nhảy 0.5
            score = int(round(score / 0.5))
            audio_id = os.path.splitext(os.path.basename(audio_path))[0]
            json_path = os.path.join(self.timestamp_folder, f"{audio_id}.json")

            if not os.path.exists(json_path):
                print(f"Missing JSON for {audio_id}")
                continue

            with open(json_path, 'r') as f:
                chunks = json.load(f)

            for chunk in chunks:
                # if not chunk["transcript_text"].strip():
                #     continue  # skip empty transcript
                self.df.append({
                    "audio_path": audio_path,
                    "pronunciation": score,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "transcript": chunk["transcript_text"]
                }) 
        self.df = pd.DataFrame(self.df, columns=["audio_path", "pronunciation", "start_time", "end_time", "transcript"])       


    def __len__(self):
        return len(self.df)

    def _pad_or_truncate(self, tensor: torch.Tensor, max_length: int, pad_value: int = 0):
        length = tensor.shape[-1]
        if length > max_length:
            return tensor[..., :max_length]
        else:
            pad_size = max_length - length
            return torch.nn.functional.pad(tensor, (0, pad_size), value=pad_value)

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        audio_path = entry["audio_path"]
        score_label = entry["pronunciation"]
        start = entry["start_time"]
        end = entry["end_time"]
        transcript = entry["transcript"]

        # Load audio and extract chunk
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        audio_chunk = audio[start_sample:end_sample]
        
        # Apply augmentation if training
        if self.is_train:
            if random.random() < 0.7:
                audio_chunk = self.noise_aug.augment(audio_chunk)[0]
            if random.random() < 0.7:
                audio_chunk = self.speed_aug.augment(audio_chunk)[0]
            if random.random() < 0.7:
                audio_chunk = self.pitch_aug.augment(audio_chunk)[0]

        # Feature extraction and padding
        audio_tensor = self.processor.feature_extractor(
            audio_chunk,
            sampling_rate=self.sample_rate,
            return_tensors='pt'
        ).input_values
        audio_tensor = audio_tensor.to(torch.float32)  # Wav2Vec2 expects float32
        

        audio_tensor = self._pad_or_truncate(audio_tensor, self.max_input_length)
        
        score_label = torch.tensor(score_label, dtype=torch.long)

        # # Process transcript
        # label_ids = self.processor.tokenizer(
        #     transcript.lower(),
        #     return_tensors='pt'
        # ).input_ids.squeeze(0)

        # label_ids = self._pad_or_truncate(label_ids, self.max_label_length, pad_value=self.processor.tokenizer.pad_token_id)
        # label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return audio_tensor, score_label, transcript

def main():
    # Example usage
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    dataset = SpeakingDatasetWav2Vec2(csv_file='/mnt/disk1/quangminh/wav2vec2_finetune/output (1).csv',
                                     processor=processor)
    # score = dataset.df['pronunciation']
    # print(score.value_counts())
    # print(dataset.df_info['pronunciation'].unique())
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        audio_tensor, label_tensor, texts_list = batch
        # print(audio_tensor, label_tensor, texts_list)
        print(f"Audio tensor shape: {audio_tensor.shape}")
        break 

if __name__ == "__main__":
    main()