# %%
import torch
import torch.nn as nn

print(torch.version.cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
x = torch.randn(2, 3, device=device)
print(x.dtype, x.shape)

# %%
y = torch.randn(2, 3, requires_grad=True, device=device)
z = y + x + 2
print(z.grad_fn)
z = z ** 2
print(z.grad_fn)

mean = z.mean()
mean.backward()

print('dz/dx')
print(x.grad) # No grad because requires_grad isn't set to True

print('dz/dy')
print(y.grad)

# %%

y_no_graph = y.detach()
# Also,
# y.requires_grad_(False)
# with torch.no_grad():

# %%

# f = <USD/UAH rate> * x + b
x = torch.tensor(
    [1, 1.5, 0.3, 0.8, 1.4, 1.1, 13, 15, 11, 6.9, 8.1],
    device=device,
)
y = torch.tensor(
    [41.31, 62.6308, 11.3310, 33.7887, 59.1990, 45.7684, 537.7081, 618.6777, 453.9877, 284.8704, 333.1389],
    device=device,
)

weights = torch.zeros(1, device=device, requires_grad=True)

def forward(_x) -> torch.Tensor:
    return weights * _x


def loss(y_actual: torch.Tensor, y_pred: torch.Tensor):
    # MSE
    return ((y_actual - y_pred) ** 2).mean()

# %%

# Non-trained pass
test = 5.1
first_result = forward(test).item() # 0
print(f'Test: f({test}) = {first_result}')

# Training

weights.grad.zero_()
with torch.no_grad():
    weights[0] = 0

epochs = 1_100
learning_rate = 1e-4

for epoch in range(1, epochs + 1):
    y_pred = forward(x)
    l = loss(y, y_pred)
    l.backward()

    with torch.no_grad():
        weights -= learning_rate * weights.grad

    weights.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {l}, weights: {weights.item()}")

print("Predict USD/UAH rate: ")
test = [5.1, 10.4, 1.0, -11]
result = [forward(sample).item() for sample in test]
print(f'Test: f({test}) = {result}')

# %%

# Using built-ins

X = torch.tensor(
    [[1], [1.5], [0.3], [0.8], [1.4], [1.1], [13], [15], [11], [6.9], [8.1]],
    device=device,
)
Y = torch.tensor(
    [[41.31], [62.6308], [11.3310], [33.7887], [59.1990], [45.7684], [537.7081], [618.6777], [453.9877], [284.8704], [333.1389]],
    device=device,
)

n_samples, n_features = X.shape
print(f"Samples={n_samples}, features={n_features}")

class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim, device=device)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(n_features, n_features)

test = torch.tensor([-1.0], device=device)
first_result = model(test).item()
print(f'Test: f({test}) = {first_result}')

# Train

epochs, learning_rate = 50_000, 1e-4
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    y_pred = model(X)

    l = loss(Y, y_pred)
    l.backward()

    # update weights and clear them
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        w, b = model.parameters()
        print(f"Epoch {epoch} loss: {l}, weights: {w[0][0].item()}, bias: {b.item()}")

# %%

print("Predict USD/UAH rate: ")
test = torch.tensor([[5.1], [10.4], [1.0], [-11]], device=device)
result = [model(sample).item() for sample in test]
print(f'Test: f({test.cpu().numpy().flatten()}) = {result}')
