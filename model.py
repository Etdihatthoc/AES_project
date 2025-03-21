import torch
import torch.nn as nn
import whisper

class WhisperScoreModel(nn.Module):
    def __init__(self, model_size='tiny'):
        """
        model_size: Kích thước của model Whisper (tiny, base, small, ...)
        """
        super(WhisperScoreModel, self).__init__()
        # Load model Whisper và lấy encoder
        whisper_model = whisper.load_model(model_size)
        self.encoder = whisper_model.encoder
        
        # Lấy kích thước của embedding từ encoder
        hidden_dim = whisper_model.dims.n_audio_state
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, mel):
        """
        mel: Tensor có shape (batch, 80, time)
        """
        # Đầu ra của encoder có shape (batch, time, hidden_dim)
        encoder_out = self.encoder(mel)
        # Mean pooling theo chiều time
        pooled = encoder_out.mean(dim=1)
        # Dự đoán score từ vector pooled qua 3 fully connected layers
        score = self.fc(pooled)
        return score.squeeze(-1)
