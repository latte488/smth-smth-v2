import torch
from torch import nn

if __name__ != "__main__":
    from . import rc
    from . import resnet

class Model(nn.Module):
    def __init__(self, column_units):
        super(Model, self).__init__()
        self.cnn = resnet.resnet50(pretrained=True)
        self.rnn = rc.ESN(2048 * 3 * 3, 1024, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, column_units),
        )
        self.echo_state = None

    def forward(self, x):
        b, c, t, h, w = 0, 1, 2, 3, 4
        x = x.permute(b, t, c, h, w).clone()
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)

        # Do not let the pretrained model calculate gradients during training.
        if self.training:
            self.eval()
            with torch.no_grad():
                x = self.cnn(x)
                x = x.detach()
            self.train()
        else:
            x = self.cnn(x)
            x = x.detach()

        x = x.view(b, t, -1)
        x = self.rnn(x)

        self.h = x.data

        x = self.classifier(x[:, -1, :])
        return x

if __name__ == "__main__":
    import rc
    import resnet
    from multi_column import MultiColumn

    num_classes = 174
    input_tensor = torch.autograd.Variable(torch.rand(8, 3, 72, 84, 84))
    input_tensor2 = torch.autograd.Variable(torch.rand(1, 3, 72, 84, 84))
    # create model
    input_tensor = [input_tensor.cuda()]
    input_tensor2 = [input_tensor2.cuda()]
    print(" > Creating model ... !")
    model = MultiColumn(num_classes, Model, 512)

    # multi GPU setting
    model = nn.DataParallel(model, [0]).cuda()


    output = model(input_tensor)
    print(output.shape)

    output = model(input_tensor2)
    print(output.shape)
