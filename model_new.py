import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, Wav2Vec2Model, DistilBertTokenizer, DistilBertModel

class MultimodalWav2VecScoreModel(nn.Module):
    def __init__(self, audio_encoder_id = "facebook/wav2vec2-base-960h", text_model_name='bert-base-uncased', d_fuse=256, num_classes=21):
        """
        whisper_model_size: Kích thước của model Whisper (tiny, base, small, ...)
        text_model_name: Tên của DistilBERT model từ HuggingFace
        d_fuse: Kích thước không gian chung sau khi chiếu (fusion dimension)
        num_classes: Số lớp của bài toán classification (mặc định 21, tương ứng với 0, 0.5, ..., 10)
        """
        super(MultimodalWav2VecScoreModel, self).__init__()
        
        # --- Audio Encoder ---
        wav2vec = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        self.audio_encoder = wav2vec
        self.audio_hidden_dim = wav2vec.config.output_hidden_size  # Ví dụ: 768
        
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
        batch_size, num_chunks, waveform_len = mels.shape
        mels = mels.view(batch_size * num_chunks, waveform_len)
        # --- Audio Branch ---
        audio_encoder_out = self.audio_encoder(
            input_values=mels
        )
        audio_features = audio_encoder_out.last_hidden_state.mean(dim=1)  # Mean pooling theo thời gian
        audio_features = audio_features.view(batch_size, num_chunks, self.audio_hidden_dim)
        audio_features = self.audio_proj(audio_features)  # (batch, num_chunks, d_fuse)
        audio_features = self.audio_norm(audio_features)
        
        
        # --- Text Branch ---
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(audio_features.device) for k, v in inputs.items()}
        text_outputs = self.text_encoder(**inputs)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
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

def main(): 
    # Example usage
    model = MultimodalWav2VecScoreModel()
    
    # # Dummy data
    # batch_size = 1
    # num_chunks = 1
    # waveform_len = 480000  # Ví dụ: 1 giây âm thanh với tần số mẫu 16kHz
    # mels = torch.randn(batch_size, num_chunks, waveform_len)
    # text = ["Second of all, have I ever travelled alone? No, I've travelled with my friends. Sometimes I've travelled with my family and with my good friends."]
    # # Forward pass
    # logits = model(mels, text)
    # print(nn.functional.softmax(logits))   # Expected output: (batch_size, num_classes)
    from dataloader import ChunkedSpeakingDataset, collate_fn
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2Processor
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    dataset = ChunkedSpeakingDataset(csv_file='/mnt/disk1/quangminh/wav2vec2_finetune/output (1).csv',
                                     timestamp_folder='audio_chunks',
                                     processor=processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        audio_tensor, label_tensor, texts_list = batch
        print(audio_tensor, label_tensor, texts_list)
        logits = model(audio_tensor, texts_list)
        print(nn.functional.softmax(logits))
        break 

if __name__ == "__main__":
    main()