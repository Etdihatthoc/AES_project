import argparse

def get_default_config():
    config = {
        'csv_file': 'data.csv',              # Đường dẫn file CSV chứa cột audio_path và score
        'model_size': 'tiny',                # Kích thước model Whisper (tiny, base, small, ...)
        'batch_size': 2,                     # Batch size cho training
        'num_epochs': 10,                    # Số epoch training
        'learning_rate': 1e-4,               # Learning rate cho optimizer
        'sample_rate': 16000,                # Tốc độ mẫu của audio (16kHz)
    }
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Speaking Scoring Model using Whisper Encoder.')
    parser.add_argument('--csv_file', type=str, default='data.csv', help='Đường dẫn tới file CSV chứa audio_path và score.')
    parser.add_argument('--model_size', type=str, default='tiny', help='Kích thước của model Whisper (tiny, base, small, ...).')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size dùng cho training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Số epoch training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate cho optimizer.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Tốc độ mẫu của audio.')
    #parser.add_argument('--checkpoint', type=str, default="ckpt/ckpt_fluency/ckpt_en_AdamW_L1_small.pth")
    args = parser.parse_args()
    return args
