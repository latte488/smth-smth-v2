import torch
from torch import nn

rnn_units = 1024

class Model(nn.Module):
    def __init__(self, column_units):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rnn = nn.LSTM(16 * (32 // 2) * (32 // 2), rnn_units, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rnn_units, column_units),
        )   

    def forward(self, x):
        x = x[0]
        b, c, t, h, w = 0, 1, 2, 3, 4
        x = x.permute(b, t, c, h, w).clone()
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        x, _ = self.rnn(x)
        self.h = x.detach()

        b, t, f = x.shape
        x = torch.stack([self.classifier(x[:, i, :]) for i in range(t)])
        return x

if __name__ == '__main__':
    model = CLSTM()
    inputs = torch.randn(8, 10, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
