import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel, WhisperConfig, WhisperModel 

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
        config = WhisperConfig.from_pretrained(f"openai/whisper-{whisper_model_size}")
        config.encoder_layerdrop = 0.1
        config.dropout = 0.1
        config.attention_dropout = 0.1
        config.activation_dropout = 0.1
        config.apply_spec_augment = True
        
        whisper_model = WhisperModel.from_pretrained(f"openai/whisper-{whisper_model_size}", config=config)
        self.audio_encoder = whisper_model.encoder
        self.audio_hidden_dim = whisper_model.config.d_model  # Ví dụ: 384 đối với tiny
        
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
        
        # --- Audio2TextAttention ---
        self.A2T = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.a2t_norm = nn.LayerNorm(d_fuse)  # LayerNorm sau cross-attention
        #self.attn_dropout  = nn.Dropout(p=0.2)
        
        # --- Text2AudioAttention ---
        self.T2A = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.t2a_norm = nn.LayerNorm(d_fuse)  # LayerNorm sau cross-attention
        #self.attn_dropout  = nn.Dropout(p=0.2)
        
        # --- AudioseflAttention ---
        self.audiosefl = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.audio_norm = nn.LayerNorm(d_fuse)  # LayerNorm sau cross-attention
        #self.attn_dropout  = nn.Dropout(p=0.2)
        
        # --- Prediction (Fully Connected layers) ---
        # Sau cross attention, trung bình theo chiều token của text để thu được vector đại diện sample.
        self.fc = nn.Sequential(
            nn.Linear(3*d_fuse, 2*d_fuse),
            nn.LayerNorm(2*d_fuse),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(2*d_fuse, d_fuse),
            nn.LayerNorm(d_fuse),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
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
        audio_encoder_out = self.audio_encoder(mels)
        audio_features = audio_encoder_out.last_hidden_state.mean(dim=1)  # Mean pooling theo thời gian
        audio_features = audio_features.view(batch_size, num_chunks, self.audio_hidden_dim)
        audio_features = self.audio_proj(audio_features)  # (batch, num_chunks, d_fuse)
        audio_features = self.audio_norm(audio_features)
        
        # --- Text Branch ---
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(audio_features.device) for k, v in inputs.items()}
        text_outputs = self.text_encoder(**inputs)
        text_features = text_outputs.last_hidden_state  # (batch, seq_len, text_hidden_dim)
        text_features = self.text_proj(text_features)      # (batch, seq_len, d_fuse)
        text_features = self.text_norm(text_features)
       
        
        # --- Audio2TextAttention ---
        a2t_output, _ = self.A2T(query=audio_features, key=text_features, value=text_features)
        a2t_output = self.a2t_norm(a2t_output)
        a2t_vector = a2t_output.mean(dim=1)  # (batch, d_fuse)
        
        # --- Text2AudioAttention ---
        t2a_output, _ = self.T2A(query=text_features, key=audio_features, value=audio_features)
        t2a_output = self.t2a_norm(t2a_output)
        t2a_vector = t2a_output.mean(dim=1)  # (batch, d_fuse)
        
        # --- AudioseflAttention ---
        self_output, _ = self.audiosefl(query=audio_features, key=audio_features, value=audio_features)
        self_output = self.audio_norm(self_output)
        audio_vector = self_output.mean(dim=1)  # (batch, d_fuse)
        
        fused_vector = torch.cat([audio_vector, a2t_vector, t2a_vector], dim=1)
        
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