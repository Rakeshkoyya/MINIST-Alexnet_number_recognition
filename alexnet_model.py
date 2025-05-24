
import sys
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim



class AlexNet(nn.Module):   
    def __init__(self, num=10):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            # Define feature extractor here...
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),

        )
        self.classifier = nn.Sequential(
            # Define classifier here...
            nn.Dropout(),
            nn.Linear(32*12*12,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,10)
        )
    
    def forward(self, x):
        # define forward network 'x' that combines feature extractor and classifier
        x = self.feature(x)
        x = x.view(-1, 32*12*12)
        x = self.classifier(x)
        return x


def train(model,optimizer,train_loader,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()#size_average=False
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    acc=100. * float(correct.to(torch.device('cpu')).numpy())
    test_accuracy = (acc / len(test_loader.dataset))
    return test_accuracy

if __name__ == '__main__':
 
    epochs = sys.argv[3].strip()
    
    if epochs.isdigit():
        epochs = int(epochs)
        
        torch.manual_seed(42)
        # Load MNIST dataset
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        full_train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
        full_test_set = dset.MNIST(root='./data', train=False, transform=trans)
        batch_size = 16
        # Get final train and test sets
        train_set, test_set = load_subset(full_train_set,full_test_set,label_one,label_two)
            
        train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

        model = AlexNet()
        if torch.cuda.is_available():
            model.cuda()
                    
        optimizer = optim.SGD(model.parameters(), lr=0.01)
            
        for epoch in range(1, epochs+1):
            train(model,optimizer,train_loader,epoch)
            accuracy = test(model,test_loader)
        torch.save(model.state_dict(), "trained_model.pth")
        print(round(accuracy,2))
            
            
    else:
        print("Invalid input")
 