import torch

class SimpleMLP(torch.nn.Module):
    def __init__(self, input_n, hidden_n, output_n):
        super(SimpleMLP, self).__init__()

        self.l1 = torch.nn.Linear(input_n, hidden_n)
        self.relu = torch.nn.ReLU()

        self.l2 = torch.nn.Linear(hidden_n, output_n)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x