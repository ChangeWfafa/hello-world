import Net.Logic as Net

import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.nn as nn




##############################################################################

#                             超参                                           #

##############################################################################



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

filename = 'Logic.pkl'

BATCH_SIZE = 50

lr = 0.0001

epochs = 10



Down = True

save = False

load = False







##############################################################################

#                数据读入与图像操作                                           #

##############################################################################



transform = transforms.Compose([

    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.RandomRotation(2),

    torchvision.transforms.ToTensor()

])





train_data = torchvision.datasets.FashionMNIST(

    root='./mnist',

    train=True,

    transform=transform,

    download=Down

)

test_data = torchvision.datasets.FashionMNIST(

    root='./mnist',

    train=False,

    transform=transforms.ToTensor(),

    download=Down

)





train_loader = torch.utils.data.DataLoader(train_data,BATCH_SIZE)

test_loader = torch.utils.data.DataLoader(test_data,BATCH_SIZE)



def pridict(model,test_loader):

    cnt = 0

    for(x,y) in test_loader:

        x=x.to(device)

        y=y.to(device)

        out = model(x).argmax(1)

        l = out - y

        cnt += torch.sum(l==0)

    print('[{}:{}] accuracy:{}'.format(cnt,10000,cnt/10000.))



##############################################################################

#                开始训练                                                     #

##############################################################################

if not load:

    model = Net.Net().to(device)

else:

    model = torch.load(filename).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr)

print(model)



best_loss = 1e5



for epoch in range(epochs):

    all_loss = 0

    for i,(x,y) in enumerate(train_loader):

        x = x.to(device)

        y = y.to(device)



        out = model(x)

        loss = criterion(out,y)



        all_loss += loss



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        if(((i+1)*BATCH_SIZE) % 10000 == 0):

            print('epoch : {} [{}:{}]  training  loss:{}'.format((epoch+1),(i+1)*BATCH_SIZE,60000,loss))

    if all_loss  < best_loss:

        best_loss = all_loss

        if save:

            torch.save(model,filename)



    pridict(model,test_loader)
