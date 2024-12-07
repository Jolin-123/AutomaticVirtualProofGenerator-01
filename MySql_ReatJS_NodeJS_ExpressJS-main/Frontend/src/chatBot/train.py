import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        
        # tokenize our pattern 
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

#lower + stem the words
# ignore_words = ['?', '!', ',', '.']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(all_words)
# print (tags)

# #creating train data
# x_train = []
# y_train = []

# for (pattern_sentence, tag ) in xy:
#     bag = bag_of_words(pattern_sentence, all_words)
#     x_train.append(bag)

#     label = tags.index(tag)
#     y_train.append(label) #1 CrossEntropyLoss

# x_train = np.array(x_train)
# y_train = np.array(y_train)



# class ChatDataset(Dataset):
#     def __init__(self):
#         self.n_samples = len(x_train)
#         self.x_data = x_train
#         self.y_data = y_train

#     #dataset[idx]
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
    
#     def __len__(self):
#         return self.n_samples
    
# # Hyperparameters 
# batch_size=8

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset, batch_size=batch_size .shuffle=True, num_workers=2)
    
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# # Hyperparameters 
# batch_size = 8
# hidden_size=8
# output_size= len(tags)
# input_size = len(x_train[0])
# # output size is same len of all_words

# learning_rate = 0.001
# num_epochs = 1000

# if __name__ == '__main__':
#     # Hyperparameters
#     batch_size = 8
#     hidden_size = 8
#     output_size = len(tags)
#     input_size = len(x_train[0])
#     learning_rate = 0.001
#     num_epochs = 1000

# #2
# print('========================================================')
# # print(input_size, len(all_words))
# # print(output_size, tags)

# # Create dataset and DataLoader
# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# # Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device).float()  # Convert to float if necessary
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Print progress every 100 epochs
#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# print(f'Final Loss: {loss.item():.4f}')






# # # Create dataset and DataLoader
# # dataset = ChatDataset()
# # train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# # #check if GPU work 
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # # loss and ooptimize 
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# # for epoch in range(num_epochs):
# #     #using trainning loader 
# #     for (words, labels) in train_loader:
# #         words= words.to(device)
# #         labels= labels.to(device)

# #         #forward
# #         outputs= model(words)
# #         loss = criterion(outputs, labels)

# #         # backward and optimizer step 
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #     if (epoch +1) % 100 == 0:
# #         print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')


# # print(f'final loss, loss = {loss.item():.4f}')


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    # Create dataset and DataLoader
    dataset = ChatDataset()
    
    # Set num_workers=0 for debugging
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device).float()  # Convert to float if necessary
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    print(f'Final Loss: {loss.item():.4f}')

    # New parts 
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f"trainning complete, file save to {FILE}")

    