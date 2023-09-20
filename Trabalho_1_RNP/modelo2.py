
#!unzip /content/drive/MyDrive/Sports.zip -d /content/Dataset_sports
#comentar apos extraido caso precise rodar denovo

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

import torch
import torch.nn as nn

class Classifier(nn.Module):

    def block(self, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1)
            ,nn.BatchNorm2d(out_channels),nn.ReLU()
            )

    def __init__(self, num_classes=100):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            self.block(3, 64, 11, 3, 0), nn.MaxPool2d(2),
            self.block(64, 256, 5, 2, 2), nn.MaxPool2d(2),
            self.block(256, 384, 3, 1, 1), nn.MaxPool2d(2),
            nn.Dropout(0.5),
            self.block(384, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.net(x)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.sigma = nn.Parameter(torch.randn(1))
        self.mu = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        out_channels, _, kernel_size, _ = self.weight.size()

        out_height = (height + 2 * self.padding - kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_size) // self.stride + 1

        padding = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        output = torch.zeros(batch_size, out_channels, out_height, out_width)

        for i in range(0, height + 2 * self.padding - kernel_size + 1, self.stride):
            for j in range(0, width + 2 * self.padding - kernel_size + 1, self.stride):
                patch = padding[:, :, i:i + kernel_size, j:j + kernel_size]

                conv = torch.sum(patch.unsqueeze(1) * self.weight, dim=(2, 3))

                if self.bias is not None:
                    conv += self.bias.unsqueeze(0)

                output[:, :, i // self.stride, j // self.stride] = conv

        return output

def operation_conv2d(Conv2d):
    mu= Conv2d.mu
    sigma= Conv2d.sigma
    bias= Conv2d.bias
    p = 1/math.sqrt(2*math.pi*bias*sigma**2)
    q = -(Conv2d-mu)**2
    u = 2*sigma**2
    return p* -1*math.exp(q/u) + bias

class Classifier_custom(nn.Module):

    def block(self, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            operation_conv2d(Conv2d(in_channels, out_channels, kernel_size, strides, padding)),
            nn.ReLU(),

            operation_conv2d(Conv2d(out_channels, out_channels, kernel_size=1)),
            nn.BatchNorm2d(out_channels),nn.ReLU(),

            operation_conv2d(Conv2d(out_channels, out_channels, kernel_size=1))
            ,nn.BatchNorm2d(out_channels),nn.ReLU()
            )

    def __init__(self, num_classes=100):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            self.block(3, 64, 11, 3, 0), nn.MaxPool2d(2),
            self.block(64, 256, 5, 2, 2), nn.MaxPool2d(2),
            self.block(256, 384, 3, 1, 1), nn.MaxPool2d(2),
            nn.Dropout(0.5),
            self.block(384, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.net(x)
    
def salva_dados(train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc):
    file = 'data_jp_mod2.csv'
    with open(file, 'a') as f:
        f.write('\n' + str(train_loss) + ',' + str(train_acc) + ',' +
                str(test_loss) + ',' + str(test_acc) + ',' +
                str(valid_loss) + ',' + str(valid_acc))

BATCH_SIZE = 64

std=torch.Tensor([0.4687, 0.4667, 0.4540])
mean=torch.Tensor([0.2792, 0.2717, 0.2852])

t_trans=transforms.Compose([transforms.Resize((224,224)), transforms.ColorJitter(contrast=0.5), transforms.RandomRotation(30),
                            transforms.ToTensor(),
                 transforms.Normalize(mean,std)])
v_trans=transforms.Compose([transforms.Resize((224,224)),
                   transforms.ToTensor(),
                   transforms.Normalize(mean,std)])

train_dir = '/home/joao/Desktop/projetos/trabalho redes neurais profundas/Sports/train'
test_dir ='/home/joao/Desktop/projetos/trabalho redes neurais profundas/Sports/test'
valid_dir = '/home/joao/Desktop/projetos/trabalho redes neurais profundas/Sports/valid'

train_dataset = datasets.ImageFolder(train_dir, transform= t_trans)
valid_dataset = datasets.ImageFolder(valid_dir , transform=v_trans)
test_dataset = datasets.ImageFolder(test_dir, transform=v_trans)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

loss_train = []
loss_valid = []
loss_test = []
accuracy_train = []
accuracy_valid = []
accuracy_test = []
loss_val = []
loss_train = []
loss_test = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    loss_train = []
    loss_val = []
    loss_med = []
    accuracy_train = []
    accuracy_valid = []
    accuracy_test = []

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=80)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_train.append(loss.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Train Loss": loss.item()})

            del images, labels, outputs
            torch.cuda.empty_cache()

        train_loss = sum(loss_train) / len(loss_train)
        scheduler.step()
        train_accuracy, train_loss = evaluate(model, train_loader)
        valid_accuracy, valid_loss = evaluate(model, valid_loader)
        test_accuracy, test_loss = evaluate(model, test_loader)

        accuracy_train.append(train_accuracy)
        accuracy_valid.append(valid_accuracy)
        accuracy_test.append(test_accuracy)

        salva_dados(train_loss, train_accuracy, test_loss, test_accuracy, valid_loss, valid_accuracy)

        torch.save(model.state_dict(), 'model_jp/model' + str(epoch) + '.pt')
        print(f"Epoch {epoch+1}/{num_epochs} - Train Accuracy: {train_accuracy:.2f}% - Valid Accuracy: {valid_accuracy:.2f}%")

    return loss_train, loss_val, accuracy_train, accuracy_valid, accuracy_test



def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    loss_med = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss_med.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            del images, labels, outputs
            torch.cuda.empty_cache()

    accuracy = 100.0 * correct / total
    return accuracy, sum(loss_med) / len(loss_med)

if __name__ == '__main__':
    num_epochs = 300

    model = Classifier()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.07, momentum=0.9, weight_decay = 6e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max = 20, eta_min = 1e-4)

    torch.backends.cudnn.benchmark = True

    loss_train, loss_val, accuracy_train, accuracy_valid, accuracy_test = train(model, criterion, optimizer, train_loader, valid_loader, num_epochs)

    test_accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), 'model.pth')