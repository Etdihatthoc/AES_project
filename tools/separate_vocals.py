import os
import tempfile
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
import logging
from audio_separator.separator import Separator
import tempfile
tempfile.tempdir = "/mnt/disk1/SonDinh/Project/tools/tmp"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def setup_logger(log_file='processing_6000_8402.log'):
    """Thiết lập logger ghi log quá trình xử lý."""
    logger = logging.getLogger("audio_separator")
    logger.setLevel(logging.INFO)
    # Tạo file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    # Thêm handler vào logger
    logger.addHandler(fh)
    return logger

def construct_output_path(input_path: str) -> str:
    """
    Tạo đường dẫn output mới bằng cách thay '/DATA/' thành '/DATA_Vocal/'
    và đổi đuôi file sang .mp3.
    """
    output_path = input_path.replace("/mnt/disk3/Sondinh/DATA", "/media/son_usb/DATA_Vocal")
    base, _ = os.path.splitext(output_path)
    output_path = base + ".mp3"
    return output_path

def ensure_dir_exists(file_path: str):
    """Tạo thư mục chứa file nếu chưa tồn tại."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def separate_vocals(separator, input_audio: str, output_audio: str = None) -> str:
    """
    Tách giọng từ file input_audio sử dụng mô hình BS-RoFormer đã load.
    Nếu output_audio được chỉ định, file kết quả sẽ được copy đến đó.
    Trả về đường dẫn file vocal.
    """
    # Tách giọng: đối với BS-RoFormer, kết quả trả về 2 file (index 1: vocals, index 0: background)
    outs = separator.separate(input_audio)
    
    if len(outs) == 2:
        vocals_relative = outs[1]
        instrument_relative = outs[0]
    else:
        raise Exception("Đầu ra từ separation không như mong đợi!")
    
    # Tạo đường dẫn file vocals (trong thư mục tạm)
    vocals_file = os.path.join(tempfile.gettempdir(), vocals_relative)
    instrument_file = os.path.join(tempfile.gettempdir(), instrument_relative)
    
    if os.path.exists(instrument_file):
        os.remove(instrument_file)
        
    # Nếu output_audio được chỉ định, copy file vocal tới đó
    if output_audio:
        ensure_dir_exists(output_audio)
        shutil.move(vocals_file, output_audio)
        return output_audio
    else:
        return vocals_file

def process_csv(csv_file: str, model_file: str, logger):
    """
    Đọc file CSV, xử lý lần lượt các file audio theo cột 'absolute_path',
    lưu file vocal theo đường dẫn mới và cập nhật lại CSV.
    """
    # Đọc CSV
    df = pd.read_csv(csv_file)
    total_files = len(df)
    
    # Khởi tạo separator và load model BS-RoFormer một lần
    output_dir = tempfile.gettempdir()
    separator = Separator(output_dir=output_dir, output_format="mp3")
    logger.info("Loading model từ %s", model_file)
    separator.load_model(model_filename=model_file)
    logger.info("Model loaded thành công.")
    
    # Duyệt qua từng dòng CSV với tqdm để theo dõi tiến trình
    for i, idx in enumerate(tqdm(df.index, desc="Processing audio files"), start=1):
        input_path = df.at[idx, 'absolute_path']
        output_path = construct_output_path(input_path)
        try:
            # Tách giọng và lưu file vocal
            result = separate_vocals(separator, input_path, output_audio=output_path)
            # Cập nhật lại cột absolute_path trong CSV
            df.at[idx, 'absolute_path'] = result
            logger.info("%d/%d: Đã xử lý %s -> %s", i, total_files, input_path, result)
        except Exception as e:
            logger.error("%d/%d: Lỗi khi xử lý %s. Error: %s", i, total_files, input_path, str(e))
    
    # Lưu CSV đã cập nhật (tạo file mới)
    updated_csv = os.path.splitext(csv_file)[0] + "_updated.csv"
    df.to_csv(updated_csv, index=False)
    logger.info("Quá trình xử lý hoàn thành. CSV được lưu tại %s", updated_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xử lý nhiều file audio từ CSV dùng BS-RoFormer để tách giọng."
    )
    parser.add_argument("csv_file", help="Đường dẫn đến file CSV chứa cột 'absolute_path'", default="/media/son_usb/DATA_Vocal/missing_files.csv")
    parser.add_argument("model_file", help="Đường dẫn đến file mô hình BS-RoFormer (.ckpt)", default="/mnt/disk1/SonDinh/Project/tools/model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    args = parser.parse_args()
    
    logger = setup_logger()
    process_csv(args.csv_file, args.model_file, logger)
