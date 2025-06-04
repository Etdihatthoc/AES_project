import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score

# Load the CSV file (đảm bảo đường dẫn file là chính xác)
data = pd.read_csv('result_fluency_on_fixtest1.csv')

# Tạo mảng nhãn từ 1 đến 10 với bước 0.5 (sử dụng trong classification)
numeric_labels = np.arange(1, 10.5, 0.5)
labels = list(map(str, numeric_labels))

# Chuyển cột GroundTruth và score_rounded sang kiểu chuỗi để xử lý phân loại
y_true_class = data['GroundTruth'].astype(str)
y_pred_class = data['score_rounded'].astype(str)

# Tính Confusion Matrix với các nhãn đã định nghĩa
cm = confusion_matrix(y_true_class, y_pred_class, labels=labels)
print("Confusion Matrix:")
print(cm)

# Vẽ và lưu ảnh Confusion Matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('fluency_confusion_matrix_newloss_test_fixtestset.png')
plt.close()

# Tính các chỉ số regression cho toàn bộ dữ liệu (nếu cần)
y_true_reg = data['GroundTruth']
y_pred_reg = data['score_rounded']

overall_mae = mean_absolute_error(y_true_reg, y_pred_reg)
overall_mse = mean_squared_error(y_true_reg, y_pred_reg)
overall_r2  = r2_score(y_true_reg, y_pred_reg)

# Tạo báo cáo phân loại
class_report = classification_report(y_true_class, y_pred_class, labels=labels)
print("Classification Report:")
print(class_report)

# Định nghĩa các khoảng giá trị cho GroundTruth
intervals = {
    "1-3.5": (1, 3.5),
    "4-7.5": (4, 7.5),
    "8-10": (8, 10)
}

# Tạo nội dung log với các chỉ số regression cho từng khoảng
log_content = f"Overall Regression Metrics:\nL1 (MAE): {overall_mae}\nMSE: {overall_mse}\nR²: {overall_r2}\n\n"

for key, (lower, upper) in intervals.items():
    # Lọc dữ liệu trong khoảng [lower, upper]
    subset = data[(data['GroundTruth'] >= lower) & (data['GroundTruth'] <= upper)]
    
    log_content += f"Regression Metrics for interval {key}:\n"
    if subset.empty:
        log_content += "  No data available\n\n"
    else:
        y_true_subset = subset['GroundTruth']
        y_pred_subset = subset['score_rounded']
        mae = mean_absolute_error(y_true_subset, y_pred_subset)
        mse = mean_squared_error(y_true_subset, y_pred_subset)
        r2  = r2_score(y_true_subset, y_pred_subset)
        log_content += f"  L1 (MAE): {mae}\n  MSE: {mse}\n  R²: {r2}\n\n"

log_content += "Classification Report:\n" + class_report

# Ghi nội dung log vào file
with open("fluency_results_test_fixset_newloss_new.log", "a", encoding="utf-8") as log_file:
    log_file.write(log_content)

print("Kết quả đã được lưu vào file 'fluency_results_test_fixset_newloss' và hình Confusion Matrix thành 'fluency_confusion_matrix_newloss_test_fixtestset.png'.")
