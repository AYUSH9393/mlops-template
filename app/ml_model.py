import torch
import torch.nn as nn

# Simple logistic regression model for binary classification
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_inputs):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Training a dummy example
if __name__ == "__main__":
    X = torch.tensor([[0.1, 0.2], [0.9, 0.8], [0.5, 0.4], [0.7, 0.6]])
    y = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    model = LogisticRegressionModel(2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training complete. Final loss:", loss.item())

    # Save model
    torch.save(model.state_dict(), "model.pth")
