import torch

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import torch.nn as nn

import torch.backends.cudnn as cudnn

from torch import optim 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print to make sure we're using the correct device

print(f"device: {device}")


#download the datasets
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

#print your dataset information
print(train_data)
print(test_data)

#set up dataloaders
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}


#define the model here

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

cnn = CNN()
#put model on GPU, set it up for parallel computation
cnn = torch.nn.DataParallel(cnn).cuda()

cudnn.benchmark=True
#print info about your model
print(cnn)

#set the loss function
loss_func=nn.CrossEntropyLoss()

#create an optimizer for gradient descent
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
#print info about your optimizer
print(optimizer)

epochs = 5

def train(epochs, cnn, loaders):
    cnn.train() #set cnn to train mode

    total_step = len(loaders['train'])

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(loaders['train']):
            #to run this on cuda, you must move your data to the cuda device
            imgs, labels = imgs.to(device), labels.to(device) 
            #get model output
            out = cnn(imgs)
            #calculate loss
            loss = loss_func(out, labels)
            #clear gradients for training step
            optimizer.zero_grad()

            #backprop
            loss.backward()

            #update model weights
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

#call the train function
train(epochs, cnn, loaders)

#function to test the outputs
def test():
    # Test the model
    cnn.eval()    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            #unless we want to move the model back to the CPU, you must also move the TEST images to GPU, next line
            images, labels = images.to(device), labels.to(device)
            test_output = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    pass
#test the model on the test set
test()