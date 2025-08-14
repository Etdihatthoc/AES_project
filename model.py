import torch
import torch.nn as nn
import whisper
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel

class MultimodalWhisperScoreModel(nn.Module):
    def __init__(self, whisper_model_size='tiny.en', text_model_name='bert-base-uncased', d_fuse=256, num_classes=21):
        """
        whisper_model_size: Kích thước của model Whisper (tiny, base, small, ...)
        text_model_name: Tên của DistilBERT model từ HuggingFace
        d_fuse: Kích thước không gian chung sau khi chiếu (fusion dimension)
        num_classes: Số lớp của bài toán classification (mặc định 21, tương ứng với 0, 0.5, ..., 10)
        """
        super(MultimodalWhisperScoreModel, self).__init__()
        
        # --- Audio Encoder ---
        whisper_model = whisper.load_model(whisper_model_size)
        self.audio_encoder = whisper_model.encoder
        self.audio_hidden_dim = whisper_model.dims.n_audio_state  # Ví dụ: 384 đối với tiny
        
        # --- Text Encoder ---
        self.text_tokenizer = BertTokenizer.from_pretrained(text_model_name)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size  # Ví dụ: 768
        
        # --- Projection sang không gian chung ---
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)
        self.audio_norm = nn.LayerNorm(d_fuse)  # LayerNorm sau audio projection
        #self.audio_dropout = nn.Dropout(p=0.2)
        
        self.text_proj = nn.Linear(self.text_hidden_dim, d_fuse)
        self.text_norm = nn.LayerNorm(d_fuse)   # LayerNorm sau text projection
        #self.text_dropout = nn.Dropout(p=0.2)
        
        # --- Cross-Modal Attention ---
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_fuse)  # LayerNorm sau cross-attention
        #self.attn_dropout  = nn.Dropout(p=0.2)
        
        # --- Prediction (Fully Connected layers) ---
        # Sau cross attention, trung bình theo chiều token của text để thu được vector đại diện sample.
        self.fc = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.LayerNorm(d_fuse),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(d_fuse, d_fuse),
            nn.LayerNorm(d_fuse),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(d_fuse, num_classes)
        )
    
    def forward(self, mels, text):
        """
        Args:
            mels: Tensor có shape (batch, num_chunks, 80, target_length)
            text: List các chuỗi văn bản (batch_size strings)
        Returns:
            logits: Tensor dự đoán logits cho các lớp (batch, num_classes)
        """
        batch_size, num_chunks, n_mel, t_len = mels.shape
        # Reshape để xử lý từng chunk riêng biệt
        mels = mels.view(batch_size * num_chunks, n_mel, t_len)  # (batch*num_chunks, 80, target_length)
        
        # --- Audio Branch ---
        audio_encoder_out = self.audio_encoder(mels)  # (batch*num_chunks, T_audio, audio_hidden_dim)
        audio_features = audio_encoder_out.mean(dim=1)  # Mean pooling theo thời gian: (batch*num_chunks, audio_hidden_dim)
        audio_features = audio_features.view(batch_size, num_chunks, self.audio_hidden_dim)
        audio_features = self.audio_proj(audio_features)  # (batch, num_chunks, d_fuse)
        audio_features = self.audio_norm(audio_features)
        #audio_features = self.audio_dropout(audio_features)
        
        # --- Text Branch ---
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(audio_features.device) for k, v in inputs.items()}
        text_outputs = self.text_encoder(**inputs)
        text_features = text_outputs.last_hidden_state  # (batch, seq_len, text_hidden_dim)
        text_features = self.text_proj(text_features)      # (batch, seq_len, d_fuse)
        text_features = self.text_norm(text_features)
        #text_features = self.text_dropout(text_features)
        
        # --- Cross-Modal Attention ---
        fusion_output, _ = self.cross_attn(query=text_features, key=audio_features, value=audio_features)
        fusion_output = self.attn_norm(fusion_output)
        #fusion_output = self.attn_dropout(fusion_output)
        fused_vector = fusion_output.mean(dim=1)  # (batch, d_fuse)
        
        # --- Prediction ---
        logits = self.fc(fused_vector)  # (batch, num_classes)
        return logits

if __name__ == "__main__":
    # Ví dụ sử dụng model:
    dummy_mels = torch.randn(1, 10, 80, 3000)  # batch=1, 10 chunks, 80 x 3000
    dummy_text = ["This is a sample sentence."]
    
    model = MultimodalWhisperScoreModel(whisper_model_size='small.en', text_model_name='bert-base-uncased')
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Using device:", device)
    model.to(device)
    dummy_mels = dummy_mels.to(device)
    
    logits = model(dummy_mels, dummy_text)
    print("Predicted logits:", logits)