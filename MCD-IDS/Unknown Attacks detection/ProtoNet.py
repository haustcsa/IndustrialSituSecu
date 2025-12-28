import torch
import torch.nn as nn
import torch.nn.functional
from sklearn.metrics import classification_report, confusion_matrix


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, x):

        features = self.encoder(x)

        logits = self.classifier(features)
        return logits

    def compute_loss_and_metrics(self, logits, y_true, class_labels=None):
        loss = nn.CrossEntropyLoss()(logits, y_true)

        _, y_pred = torch.max(logits, dim=1)

        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        metrics = classification_report(
            y_true_np, y_pred_np, target_names=class_labels, output_dict=True, zero_division=0
        )
        cf_matrix = confusion_matrix(y_true_np, y_pred_np)
        return loss, metrics, cf_matrix


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 一个简单的卷积神经网络编码器
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),  # 展平
            nn.Linear(64 * 7 * 7, output_dim),  # 输出到特征维度
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


# 测试
if __name__ == "__main__":

    input_data = torch.rand(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))

    encoder = Encoder(input_dim=1, output_dim=128)  # 特征输出为 128 维
    model = ClassificationModel(encoder=encoder, num_classes=10)  # 10 个类别

    logits = model(input_data)

    # 计算损失和指标
    loss, metrics, cf_matrix = model.compute_loss_and_metrics(logits, labels, class_labels=[f"Class_{i}" for i in range(10)])

    print("Loss:", loss.item())
    print("Classification Metrics:", metrics)
    print("Confusion Matrix:")
    print(cf_matrix)



