import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


class Classifier(nn.Module):

    def block(self, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
             nn.BatchNorm2d(out_channels),nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1),
             nn.BatchNorm2d(out_channels),nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            )

    def __init__(self, num_classes=100):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            self.block(3, 128, 11, 3, 0), nn.MaxPool2d(3,2),
            self.block(128, 256, 5, 2, 2), nn.MaxPool2d(3,2),
            self.block(256, 384, 3, 1, 1), nn.MaxPool2d(3,2),
            nn.Dropout(0.5),
            self.block(384, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.net(x)
    
BATCH_SIZE = 64

std=torch.Tensor([0.4687, 0.4667, 0.4540])
mean=torch.Tensor([0.3, 0.3, 0.3])

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

def salva_dados(train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc):
    file = 'data_jp.csv'
    with open(file, 'a') as f:
        f.write('\n' + str(train_loss) + ',' + str(train_acc) + ',' +
                str(test_loss) + ',' + str(test_acc) + ',' +
                str(valid_loss) + ',' + str(valid_acc))
        

if __name__ == '__main__':
        
    model = Classifier()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.07, momentum=0.9, weight_decay = 6e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max = 20, eta_min = 1e-4)

    torch.backends.cudnn.benchmark = True

    num_epochs = 500
    loss_train, loss_val, accuracy_train, accuracy_valid, accuracy_test = train(model, criterion, optimizer, train_loader, valid_loader, num_epochs)

    test_accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")


    torch.save(model.state_dict(), 'model.pth')

