import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./root',train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=True,num_workers=0)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1,64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

num_epochs = 5
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for images,labels in train_loader:
        output = model(images)
        loss = criterion(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    print(f"Epoch[{epoch+1}/{num_epochs}], Loss:{total_loss/len(train_dataset):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images,labels in test_loader:
        output = model(images)
        _,predicted = torch.max(output,dim=1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
