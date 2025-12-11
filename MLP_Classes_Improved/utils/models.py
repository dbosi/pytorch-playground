import torch

class ImprovedMLP(torch.nn.Module):
    def __init__(self, input_n: int, hidden_1_n: int, hidden_2_n: int, output_n: int):
        super(ImprovedMLP, self).__init__()

        self.l1 = torch.nn.Linear(input_n, hidden_1_n)
        torch.nn.init.kaiming_normal_(self.l1.weight, nonlinearity="relu")
        self.relu_1 = torch.nn.ReLU()


        self.l2 = torch.nn.Linear(hidden_1_n, hidden_2_n)
        torch.nn.init.kaiming_normal_(self.l2.weight, nonlinearity="relu")
        self.relu_2 = torch.nn.ReLU()


        self.l3 = torch.nn.Linear(hidden_2_n, output_n)
        torch.nn.init.kaiming_normal_(self.l3.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.l1(x)
        x = self.relu_1(x)

        x = self.l2(x)
        x = self.relu_2(x)

        x = self.l3(x)

        return x
