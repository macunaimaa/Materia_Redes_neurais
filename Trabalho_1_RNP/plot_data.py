import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv 


def plot_acuracia(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data[:206]
    #data = data
    data = data.to_numpy()
    data = np.transpose(data)
    plt.plot(data[0], label='Train Loss')
    plt.plot(data[1], label='Train Acc')
    plt.plot(data[2], label='Test Loss')
    plt.plot(data[3], label='Test Acc')
    plt.plot(data[4], label='Valid Loss')
    plt.plot(data[5], label='Valid Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()

def plot_loss(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data[:206]
    #data = data
    data = data.to_numpy()
    data = np.transpose(data)
    plt.plot(data[0], label='Train Loss')
    plt.plot(data[2], label='Test Loss')
    plt.plot(data[4], label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def show_pic_side_to_side(figure_1, figure_2):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(figure_1)
    axs[1].imshow(figure_2)
    plt.show()

#dados: train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc

def find_max_value(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.to_numpy()
    data = np.transpose(data)
    print(f'train accuracy : {np.max(data[1])}')
    print(f'test accuracy : {np.max(data[3])}')
    print(f'valid accuracy : {np.max(data[5])}')

def extract_every_20_epochs(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.to_numpy()
    data = np.transpose(data)
    data = data[:, ::15]
    data = np.transpose(data)
    data = pd.DataFrame(data)
    data.to_csv('data_jp_mod2.csv', index=False, header=False)


#extract_every_20_epochs('data_jp_mod2.csv')

find_max_value('data_jp_mod2.csv')
plot_acuracia('data_jp_final_mod1_melhor.csv')