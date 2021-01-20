import torch
from torch import nn
import selected_dropout

rnn_units = 2048

class Model(nn.Module):
    def __init__(self, column_units, score, d_rate, inv):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rnn = nn.RNNCell(16 * (32 // 2) * (32 // 2), rnn_units)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rnn_units, column_units),
        )

        mask = selected_dropout.dropout_mask(score, d_rate, inv)
        self.register_buffer('mask', mask)

    def forward(self, x):
        x = x[0]
        device = x.device
        b, c, t, h, w = 0, 1, 2, 3, 4
        x = x.permute(b, t, c, h, w).clone()
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        hx = torch.zeros(b, rnn_units).to(device)
        hxs = torch.zeros(b, t, rnn_units).to(device)
        for i in range(t):
            hx = self.rnn(x[:, i, :], hx)
            hx = hx * mask
            hxs[:, i, :] = hx
       
        self.h = hxs.detach()

        b, t, f = hxs.shape
        x = torch.stack([self.classifier(hxs[:, i, :]) for i in range(t)])
        return x

if __name__ == '__main__':
    model = CLSTM()
    inputs = torch.randn(8, 10, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
