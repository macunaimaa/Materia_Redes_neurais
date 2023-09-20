import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kstest
import numpy as np
from Macky_glas import x
from modelos import MLP, LSTM, GRU

lr = 0.001
num_epochs = 150
batch_size = 1
samples = len(x)
input_size = 20
hidden_size = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_sliding_windows(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(sequences), np.array(targets)

x = np.array(x).reshape(-1, 1)

x_train = np.array(x[:int(0.9*samples)])
X_train, y_train = create_sliding_windows(x_train, input_size)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

x_test = np.array(x[int(0.9*samples)-input_size:])
X_test, y_test = create_sliding_windows(x_test, input_size)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

test_data = x[int(-0.1*len(x)-input_size):]


def EQM(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)


def EQMN1(y_pred, y_true):
    return EQM(y_pred, y_true).item() / EQM(torch.mean(y_true), y_true).item()


def EQMN2(y_pred, y_true, last_value):
    return EQM(y_pred, y_true).item() / EQM(last_value, y_true).item()

def indicators(predicted, real, last):
    print(f"EQM:   {EQM(torch.tensor(predicted)  , torch.tensor(real))}")
    print(f"EQMN1: {EQMN1(torch.tensor(predicted), torch.tensor(real))}")
    print(f"EQMN2: {EQMN2(torch.tensor(predicted), torch.tensor(real), torch.tensor(last))}")
    print(kstest(real, predicted))

def plot_predictions(predictions, targets):
    plt.plot(predictions, label='Predictions')
    plt.plot(targets, label='Real')
    plt.legend()
    plt.show()

def train(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    losses = []
    pbar = tqdm(range(num_epochs), desc=f"Training {model.__class__.__name__}", ncols=80)
    for epoch in pbar:
        predicted = []
        real = []
        last_value = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            input = data.squeeze(2)
            # forward
            scores = model(input)
            loss = criterion(scores, targets)
            
            real += targets.squeeze(1).cpu().detach().numpy().tolist()
            predicted += scores.cpu().squeeze(1).detach().numpy().tolist()
            last_value += data[:, -1, 0].cpu().detach().numpy().tolist()

            # backward
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss': loss.item()})
    return losses, predicted, real, last_value

# def test(model, train_data):
#     model.eval()
#     test_data = train_data[-input_size:].copy()
#     predicted = []
#     last_value = []
#     with torch.no_grad():
#         for i in range(int(samples*0.1)):
#             x = torch.tensor(test_data[-input_size:]).unsqueeze(0).float().to(device=device)
#             output = model(x)
#             predicted.append(output.item())
#             test_data.append(output.item())
#             last_value += list(x[:, -1].cpu().detach().numpy())

#     return predicted, last_value

def test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  
        predictions = []
        last_value = []
        for value, _ in test_loader:
            value = value.squeeze(2).to(device=device)
            print(value)
            prediction = model(value)
            predictions.append(prediction.item())
            last_value += value[:, -1].cpu().detach().numpy().tolist()
    return predictions, last_value

def save_metrics_csv(loss, predicted,real, model_name):
    with open(f"data/{model_name}.csv", 'w') as f:
        f.write(f"loss,predicted,real\n")
        for i in range(len(loss)):
            f.write(f"{loss[i]},{predicted[i]},{real[i]}\n")

def save_predictions_imgs(predicted, real, model_name):
    plt.plot(predicted, label='Predictions')
    plt.plot(real, label='Real')
    plt.legend()
    plt.savefig(f"Imagens/{model_name}_professor.png")
    plt.close()


def prepare_dataloader(data):
    x=[]
    y=[]
    for i in range(len(data)-input_size):
        x.append(data[i:i+input_size])
        y.append(data[i+input_size])
    x = torch.tensor(np.array(x)).float()

    y = torch.tensor(np.array(y)).float()

    dataset = torch.utils.data.TensorDataset(x, y)
    DataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return DataLoader

def Model(network):
    model = network(input_size=input_size, hidden_size=hidden_size , output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

def treino_teste(network, data, X_test=X_test):
    train_data = data[:int(samples*0.9)]
    train_data = np.array(train_data)
    data_loader = prepare_dataloader(train_data)
    test_loader = prepare_dataloader(x_test)
    test_real = data[int(samples*0.9):]
    test_real = torch.tensor(test_real)
    test_real = test_real.squeeze(1)
    test_real = np.array(test_real)

    model, optimizer, criterion = Model(network)
    losses, predicted_treino, real_treino, last_value_treino = train(model, data_loader, optimizer, criterion, num_epochs, device)
    torch.save(model.state_dict(), f"Modelos/{network.__name__}.pt")
    predicted_teste, last_value_teste = test(model, test_loader)

    print('Indicadores de treino:')
    indicators(predicted_treino, real_treino, last_value_treino)

    print('Indicadores de teste:')
    indicators(predicted_teste, test_real, last_value_teste)
    save_predictions_imgs(predicted_teste, test_real, network.__name__)
    save_metrics_csv(losses, predicted_teste, test_real, network.__name__)

    return losses, predicted_teste

def plot_n_save_losses_from_csv(model_name):
    with open(f"data/{model_name}.csv", 'r') as f:
        f.readline()
        losses = []
        for line in f:
            losses.append(float(line.split(',')[0]))
        plt.plot(losses)
        plt.savefig(f"graficos_losses/{model_name}_loss.png")
        plt.close()
    
if __name__ == '__main__':
    data = np.array(x)

    #mlp_losses, mlp_predicted = treino_teste(MLP, data)
    #lstm_losses, lstm_predicted = treino_teste(LSTM, x)
    #gru_losses, gru_predicted = treino_teste(GRU, x)

    plot_n_save_losses_from_csv('MLP')
    plot_n_save_losses_from_csv('LSTM')
    plot_n_save_losses_from_csv('GRU')