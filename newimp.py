import torch
import time

from cnn_setup import Pipesound
from cnn import FlowClassify
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

#abspath = "/itf-fi-ml/shared/users/daniejaa/test"   #/itf-fi-ml/shared/users/daniejaa/test
abspath = "/run/media/daniel/ATLEJ-STT/data/test"   #/itf-fi-ml/shared/users/daniejaa/test
ds = Pipesound("meta.csv", abspath, transform = "pad", width_cut = 700)

l = len(ds)
trainsize = int(0.75*l)
testsize = int(l - trainsize)

train, test = random_split(ds, [trainsize, testsize])
print(f"Length train set: {len(train)}")
print(f"Length test set: {len(test)}")

trainloader = DataLoader(train, batch_size=1, shuffle=True, num_workers=2)
testloader = DataLoader(test, batch_size=5, shuffle=True, num_workers=2)

sample = torch.rand([1, ds.max_h, ds.max_w], device = device)
model = FlowClassify(sample_input = sample,
                     device=device).double()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters())

print("Starting training. Good luck")
epochs = 10
for e in range(epochs):
    print(f"\nEpoch {e+1}/{epochs}")

    epoch_loss = 0
    tmp_loss = 0
    epoch_test_acc = 0
    epoch_train_acc = 0

    t0 = time.time()
    for i, data in enumerate(trainloader):
        tens, label = data
        tens /= torch.max(tens)
        tens = tens.to(device)
        label = label.to(device)

        #print(f"Input tensor shape: {tens.shape}")
        #print(f"Input label shape: {label.shape}")

        # Prediction
        pred = model(tens)

        #print(f"Output tensor shape: {pred.shape}")
        #print(f"Label:\t{label}\nPrediction:\t{pred}")
        print(pred)
        print(pred.shape)
        print(label)
        print(label.shape)
        exit(1)

        # Calculate loss
        loss = criterion(pred, label)

        epoch_loss += loss.item()
        tmp_loss += loss.item()

        # Backward and step in param space
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exit(0)
        if (i+1)%100 == 0:
            print(f"[{i+1}/{len(train)}]\tAvg. loss: {tmp_loss/100}")
            tmp_loss = 0

    t1 = time.time()
    print(f"Time for epoch: {t1-t0} seconds")
    for i, data in enumerate(testloader):
        with torch.no_grad():
            tens, label = data
            tens /= torch.max(tens)
            tens = tens.to(device)
            label = label.to(device)
            
            pred = model(tens)
            _, pred_class = torch.max(pred, 1)
            epoch_test_acc += (pred_class == label).sum().item()

    for i, data in enumerate(trainloader):
        with torch.no_grad():
            tens, label = data
            tens /= torch.max(tens)
            tens = tens.to(device)
            label = label.to(device)
            
            pred = model(tens)
            _, pred_class = torch.max(pred, 1)
            epoch_train_acc += (pred_class == label).sum().item()

    epoch_test_acc /= len(test)
    epoch_train_acc /= len(train)

    print(f"Accuracy on train set: {epoch_train_acc}")
    print(f"Accuracy on test set: {epoch_test_acc}")
