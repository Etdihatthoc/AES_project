import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel, BertTokenizer, AutoModel, AutoTokenizer, T5ForConditionalGeneration
import gc 


class FusionTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(FusionTransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        return x


class CascadeWav2VecScoreModel(nn.Module):
    def __init__(self, 
                 audio_encoder_id="facebook/wav2vec2-base-960h", 
                 text_model_name="bert-base-uncased", 
                 translation_model_id="VietAI/envit5-translation",
                 translated_text_model_name="vinai/phobert-base",
                 d_fuse=256, 
                 num_classes=21, 
                 max_len_text=1024, 
                 max_len_audio=100,
                 device='cpu'):
        
        super(CascadeWav2VecScoreModel, self).__init__()
        self.device = device
        self.d_fuse = d_fuse

        # --- Encoders ---
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        self.audio_hidden_dim = self.audio_encoder.config.output_hidden_size
        
        self.text_tokenizer = BertTokenizer.from_pretrained(text_model_name)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size
        
        self.translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_id)
        self.translation_encoder = T5ForConditionalGeneration.from_pretrained(translation_model_id)
        
        self.translated_text_tokenizer = AutoTokenizer.from_pretrained(translated_text_model_name)
        self.translated_text_encoder = AutoModel.from_pretrained(translated_text_model_name)
        self.translated_text_hidden_dim = self.translated_text_encoder.config.hidden_size

        # --- Projection Layers ---
        self.text_proj = nn.Linear(self.text_hidden_dim, d_fuse)
        self.translated_proj = nn.Linear(self.translated_text_hidden_dim, d_fuse)
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)

        # --- Positional Embeddings ---
        self.pos_embed_text = nn.Embedding(max_len_text, d_fuse)
        self.pos_embed_fusion = nn.Embedding(max_len_text + max_len_audio, d_fuse)

        # --- Fusion Transformer Blocks ---
        self.text_fusion_block = FusionTransformerBlock(d_model=d_fuse, num_heads=4)
        self.cross_modal_fusion_block = FusionTransformerBlock(d_model=d_fuse, num_heads=4)

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.ReLU(),
            nn.Linear(d_fuse, num_classes)
        )
        
        self.to(device)
    
    def to(self, device):
        super().to(device)
        self.audio_encoder.to(device)
        self.text_encoder.to(device)
        self.translation_encoder.to(device)
        self.translated_text_encoder.to(device)
        self.text_proj.to(device)
        self.translated_proj.to(device)
        self.audio_proj.to(device)
        self.pos_embed_text.to(device)
        self.pos_embed_fusion.to(device)
        self.text_fusion_block.to(device)
        self.cross_modal_fusion_block.to(device)
        self.classifier.to(device)

    def add_positional_encoding(self, x, pos_embed_layer):
        """
        Add positional encoding.
        x: (B, T, d_fuse)
        pos_embed_layer: nn.Embedding for position IDs
        """
        B, T, _ = x.size()
        pos_ids = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, T)
        pos_embeds = pos_embed_layer(pos_ids)  # (B, T, d_fuse)
        return x + pos_embeds

    def forward(self, audio, text, translated_text = None):
        """
        Args:
            audio: Tensor có shape (batch, num_chunks, target_length)
            text: List các chuỗi văn bản (batch_size strings)
        Returns:
            logits: Tensor dự đoán logits cho các lớp (batch, num_classes)
        """
        # ---- AUDIO ----
        batch_size, num_chunks, waveform_len = audio.shape
        device = self.device
        audio_encoder_out = []
        for i in range(num_chunks):
            inp = audio[:, i, :].to(device) 
            out = self.audio_encoder(input_values=inp).last_hidden_state
            audio_encoder_out.append(out.mean(dim=1).detach().cpu())  # (batch, 1, audio_hidden_dim)
            
            del inp, out # remove unused variables 
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        audio_feats = torch.stack(audio_encoder_out, dim=1).to(device)  # (batch, num_chunks, audio_hidden_dim)
       
        audio_feats = audio_feats.view(batch_size, num_chunks, self.audio_hidden_dim)
        audio_feats = self.audio_proj(audio_feats)  # (batch, num_chunks, d_fuse)
        # audio_feats = self.audio_norm(audio_feats)
        # ---- TEXT ENCODING ----
        text_inputs = self.text_tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=128)
        text_features = self.text_encoder(input_ids=text_inputs['input_ids'].to(self.device), 
                                       attention_mask=text_inputs['attention_mask'].to(self.device)).last_hidden_state  # (B, T1, d_text)
        
        text_feats = self.text_proj(text_features)  # (B, T1, d_fuse)
        text_feats = self.add_positional_encoding(text_feats, self.pos_embed_text)  # (B, d_fuse)
        
        if translated_text is not None:
            # ---- TRANSLATED TEXT ENCODING ----
            translated_text_inputs = self.translated_text_tokenizer(text = translated_text, padding=True, truncation=True, return_tensors='pt', max_length=128)
            translated_feats = self.translated_text_encoder(input_ids=translated_text_inputs['input_ids'].to(self.device),
                                                            attention_mask=translated_text_inputs['attention_mask'].to(self.device)).last_hidden_state  # (B, T2, d_translated_text)
            translated_feats = self.translated_proj(translated_feats)  # (B, T2, d_fuse)
            translated_feats = self.add_positional_encoding(translated_feats, self.pos_embed_text)  # (B, T2, d_fuse)       
            
            # ---- TEXT FUSION ----
            text_concat = torch.cat([text_feats, translated_feats], dim=1)  # (B, T_text_total, d)
            text_concat = self.add_positional_encoding(text_concat, self.pos_embed_text)
            fused_text = self.text_fusion_block(text_concat)  # (B, T_text_total, d)
        
        else: 
            # If no translated text, just use text features
            fused_text = text_feats

        # ---- CROSS-MODAL FUSION ----
        fusion_input = torch.cat([fused_text, audio_feats], dim=1)  # (B, T_total, d)
        fusion_input = self.add_positional_encoding(fusion_input, self.pos_embed_fusion)
        final_fusion = self.cross_modal_fusion_block(fusion_input)  # (B, T_total, d)

        # ---- POOLING & CLASSIFICATION ----
        pooled = final_fusion[:, 0]  # Use first token ([CLS] or first token)
        out = self.classifier(pooled)
        return out


def main(): 
    # Example usage
    model = CascadeWav2VecScoreModel(device = 'cuda')

    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2Processor
    from translated_dataloader import TranslatedSpeakingDatasetWav2Vec2, translated_collate_fn
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    dataset = TranslatedSpeakingDatasetWav2Vec2(csv_file='new_full_train_greaterthan3_removenoise_fixspellingerror_translated.csv',
                                     processor=processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=translated_collate_fn)

    for batch in dataloader:
        audio_tensor, label_tensor, texts_list, translated_texts_list = batch
        logits = model(audio_tensor, texts_list)
        print(f"Logits shape: {logits.shape}")
        break 

if __name__ == "__main__":
    main()