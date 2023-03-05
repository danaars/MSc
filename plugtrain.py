import time
import torch

from plugCNN import PlugSet, PlugRegClass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_dir = "/run/media/daniel/ATLEJ-STT/plugslug"
ds = PlugSet("plugmeta.csv", data_dir, transform="pad")
sample_input = ds.getsample(3, device)

l = len(ds)
trainsize = int(0.8*l)
testsize = int(l - trainsize)

train, test = torch.utils.data.random_split(ds, [trainsize, testsize])
print(f"Length train set: {len(train)}")
print(f"Length test set: {len(test)}")

trainloader = torch.utils.data.DataLoader(train,
                                          batch_size=3,
                                          shuffle=True,
                                          num_workers=1)
testloader = torch.utils.data.DataLoader(test,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=1)
model = PlugRegClass(sample_input=sample_input, 
                     device=device).double()

class_loss = torch.nn.CrossEntropyLoss()
vel_reg_loss = torch.nn.MSELoss()
len_reg_loss = torch.nn.MSELoss()

def total_loss(output, target):
    class_pred, vel_pred, len_pred = output
    class_target, vel_target, len_target = target

    velmask = torch.isnan(vel_target)
    lenmask = torch.isnan(len_target)

    #print("Inside loss")
    #print(class_pred)
    #print(class_pred.shape)
    #print(class_target)
    #print(class_target.shape)
    classification_loss = class_loss(class_pred, class_target)
    print(classification_loss.item())
    #print("Classification loss calculated")

    loss_vals = {"ClassificationCEL": classification_loss.item()}

    if (not torch.isnan(vel_target).all()) and (not torch.isnan(len_target).all()):
        #print(f"Vel pred: {vel_pred}")
        #print(f"Vel target: {vel_target}")
        velocity_loss = vel_reg_loss(vel_pred[~velmask], vel_target[~velmask])
        loss_vals["VelocityMSE"] = velocity_loss.item()
        #print("Velocity loss calculated")
        length_loss = len_reg_loss(len_pred[~lenmask], len_target[~lenmask])
        loss_vals["LengthMSE"] = length_loss.item()
        #print("Length loss calculated")

        total_loss = classification_loss + velocity_loss + length_loss
    else:
        total_loss = classification_loss

    return total_loss, loss_vals

optimizer = torch.optim.Adam(params = model.parameters())

#print("Starting training")
epochs = 100

running_loss = {"ClassificationCEL":[], "VelocityMSE":[], "LengthMSE":[]}

for e in range(epochs):
    print(f"\nEpoch {e+1}/{epochs}")

    t0 = time.time()
    for i, data in enumerate(trainloader):
        tens, label_idx, velocity, length = data
        tens /= torch.max(tens)
        tens = tens.to(device)
        label_idx = label_idx.to(device)
        velocity = velocity.to(device)
        length = length.to(device)

        #print(velocity)
        #print(length)

        #print(tens.shape)
        prediction = model(tens)
        #print(prediction[0])
        #print(prediction[0].shape)
        #print(label_idx)
        #print(label_idx.shape)
        #print("Prediction made")
        #print(prediction[1].shape == velocity.unsqueeze(1).shape)
        target = (label_idx, velocity.unsqueeze(1), length.unsqueeze(1))

        loss, loss_dict = total_loss(prediction, target)
        for key in loss_dict:
            running_loss[key].append(loss_dict[key])
        #print(f"Total loss: ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%20==0:
            #print(loss_dict)
            for key in running_loss:
                if len(running_loss[key])==0:
                    continue
                else:
                    print(f"Avg. {key}:\t{sum(running_loss[key])}/{len(running_loss[key])} = {sum(running_loss[key])/len(running_loss[key])}")
                running_loss[key] = []
            print("")

    t1 = time.time()
    print(f"Time for epoch: {t1-t0} seconds")
