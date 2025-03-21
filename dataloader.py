import torch
import torch.utils.data as data
import pandas as pd
import librosa
import whisper
def chunk_audio(audio, sr, chunk_length_sec=30, hop_length_sec=20):
    """
    Cắt audio thành nhiều đoạn xen kẽ, mỗi đoạn có độ dài tối đa chunk_length_sec giây,
    với bước trượt (hop) là hop_length_sec giây.
    Ví dụ: Với chunk_length_sec=30, hop_length_sec=20, ta có các chunk: 0-30s, 20-50s, 40-70s,...
    Trả về danh sách các mảng audio numpy.
    """
    samples_per_chunk = int(chunk_length_sec * sr)
    step_samples = int(hop_length_sec * sr)
    chunks = []
    for start in range(0, len(audio), step_samples):
        end = start + samples_per_chunk
        chunk = audio[start:end]
        chunks.append(chunk)
    return chunks

def pad_or_trim_tensor(tensor, length=3000):
    """
    Pad hoặc trim tensor theo chiều time (chỉ số thứ 1) để đảm bảo có độ dài cố định.
    Input tensor có shape (80, T).
    """
    current_len = tensor.shape[1]
    if current_len < length:
        pad = torch.zeros(tensor.shape[0], length - current_len, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    else:
        return tensor[:, :length]

class SpeakingDataset(data.Dataset):
    def __init__(self, csv_file, sample_rate=16000, chunk_length_sec=30, target_length=3000, hop_length_sec=20):
        """
        csv_file: Đường dẫn file CSV chứa cột 'absolute_path' và 'pronunciation'
        sample_rate: Tốc độ mẫu của audio (16kHz)
        chunk_length_sec: Độ dài của mỗi chunk (giây)
        target_length: Số frame mong muốn của mel spectrogram (ví dụ: 3000)
        hop_length_sec: Bước trượt giữa các overlapping chunk (giây)
        """
        self.df = pd.read_csv(csv_file)
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        self.target_length = target_length
        self.hop_length_sec = hop_length_sec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['absolute_path']
        score = row['pronunciation']

        # 1. Load toàn bộ audio với sample_rate cho trước
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 2. Tạo overlapping chunks: ví dụ 0-30s, 20-50s, ...
        audio_chunks = chunk_audio(audio, sr, self.chunk_length_sec, self.hop_length_sec)
        
        # 3. Với mỗi chunk, tính log-mel spectrogram, sau đó pad/trim về độ dài cố định
        sample_list = []
        for chunk in audio_chunks:
            mel_chunk = whisper.log_mel_spectrogram(chunk)  # shape: (80, T_chunk)
            final_mel = pad_or_trim_tensor(mel_chunk, length=self.target_length)  # shape: (80, target_length)
            sample_list.append((final_mel, torch.tensor(score, dtype=torch.float)))
        
        # Mỗi video có thể tạo ra nhiều sample (mỗi sample là 1 overlapping chunk)
        return sample_list
def collate_fn(batch):
    """
    Vì mỗi item của dataset trả về là một list các sample (mỗi sample: (mel, score)),
    hàm collate này sẽ flatten list đó lại trước khi stack các tensor.
    """
    flat_list = []
    for sample_list in batch:
        flat_list.extend(sample_list)
    mels, scores = zip(*flat_list)
    mels_tensor = torch.stack(mels)       # (total_samples, 80, target_length)
    scores_tensor = torch.stack(scores)
    return mels_tensor, scores_tensor
