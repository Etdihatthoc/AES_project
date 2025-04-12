import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, left_weight=0.15, right_weight=0.15, correct_weight=0.7, class_weight=None):
        """
        num_classes: số lớp dự đoán
        left_weight: trọng số cho lớp bên trái (nếu tồn tại)
        right_weight: trọng số cho lớp bên phải (nếu tồn tại)
        correct_weight: trọng số cho lớp đúng
        class_weight: tensor trọng số cho từng lớp, có shape (num_classes,)
                      Nếu không có, mặc định sẽ không áp dụng trọng số cho mẫu.
        """
        super(SoftLabelCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.correct_weight = correct_weight
        self.class_weight = class_weight  
        
    def forward(self, logits, target):
        """
        logits: đầu ra của model, shape (batch_size, num_classes)
        target: tensor chứa nhãn dưới dạng chỉ số lớp, shape (batch_size,)
        """
        batch_size = target.size(0)
        # Tạo soft label theo dạng one-hot chuyển sang phân phối mềm
        soft_targets = torch.zeros(batch_size, self.num_classes, device=logits.device)
        soft_targets.scatter_(1, target.unsqueeze(1), self.correct_weight)
        
        # Gán trọng số cho các lớp lân cận nếu có
        for i in range(batch_size):
            t = target[i].item()
            if t > 0:  # lớp bên trái
                soft_targets[i, t - 1] = self.left_weight
            if t < self.num_classes - 1:  # lớp bên phải
                soft_targets[i, t + 1] = self.right_weight
                
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        log_prob = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_prob).sum(dim=1)
        
        if self.class_weight is not None:
            weight = self.class_weight.to(logits.device)  
            sample_weights = weight[target] 
            loss = loss * sample_weights

        return loss.mean()
def main():
    
    num_classes = 6
    target = torch.tensor([3])  # Chọn target là lớp 3

    criterion = SoftLabelCrossEntropyLoss(num_classes=num_classes,
                                          left_weight=0.15,
                                          right_weight=0.15,
                                          correct_weight=0.7,
                                          class_weight=None)
    

    logits_good = torch.tensor([[ -1.0, -1.0, 0.5, 2.0, 0.1, -1.0]])
    

    logits_bad = torch.tensor([[2.0, 0.5, -1.0, -1.0, -1.0, 0.0]])
    

    loss_good = criterion(logits_good, target)
    loss_bad = criterion(logits_bad, target)
    
    # In ra kết quả
    print("Good Prediction Logits:\n", logits_good)
    print("Loss (Good Prediction):", loss_good.item())
    
    print("\nBad Prediction Logits:\n", logits_bad)
    print("Loss (Bad Prediction):", loss_bad.item())
    
    criterion = nn.CrossEntropyLoss(weight=None,label_smoothing=0.2)
    
    loss_good = criterion(logits_good, target)
    loss_bad = criterion(logits_bad, target)
    print("CEL thường ==============")
    print("Good Prediction Logits:\n", logits_good)
    print("Loss (Good Prediction):", loss_good.item())
    
    print("\nBad Prediction Logits:\n", logits_bad)
    print("Loss (Bad Prediction):", loss_bad.item())
    

if __name__ == '__main__':
    main()