import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import modelo1
import modelo2

model1 = modelo1.Classifier().to(modelo1.device)
model2 = modelo2.Classifier().to(modelo2.device)

# Load the models
model1.load_state_dict(torch.load('/home/joao/Desktop/projetos/trabalho redes neurais profundas/modelos epoca 143/model423.pt',map_location=torch.device('cpu')))
model2.load_state_dict(torch.load('/home/joao/Desktop/projetos/trabalho redes neurais profundas/modelos epoca 143/model143_modelo2.pt',map_location=torch.device('cpu')))

model1.eval()
model2.eval()

test_data = datasets.ImageFolder('/home/joao/Desktop/projetos/trabalho redes neurais profundas/Sports/test', transform= transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=10, shuffle=True, pin_memory=True)

def test_and_show(model1,model2):
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(modelo1.device), labels.to(modelo1.device)
        output1 = model1(images)
        output2 = model2(images)
        pred1 = torch.argmax(output1,1)
        pred2 = torch.argmax(output2,1)
        for i, (image, classe) in enumerate(zip(images,labels)):
            text = 'Pred true: ' + str(test_data.classes[classe.item()]) + '\nPred model 1: ' + str(test_data.classes[pred1[i].item()]) + '\nPred model 2: ' + str(test_data.classes[pred2[i].item()])
            plt.clf()
            image = image.permute(1, 2, 0)
            plt.imshow(image)
            plt.text(0, 0, text, fontsize=12)
            plt.axis('off')
            plt.savefig(f'imagem_{i}.png')
    
        break

test_and_show(model1, model2)