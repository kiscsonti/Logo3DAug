from torch import nn


class ConvNet(nn.Module):

    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.hidden_size = args.hidden_size
        self.n_layers = 4
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(((224//(2**self.n_layers))**2) * self.hidden_size, len(args.classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
