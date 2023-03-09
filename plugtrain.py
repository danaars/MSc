import time
import json
import torch
import torchmetrics

from plugCNN import PlugSet, PlugRegClass

print(f"PyTorch: {torch.__version__}")
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
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(test,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=2)

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

    classification_loss = class_loss(class_pred, class_target)

    loss_vals = {"ClassificationCEL": classification_loss.item()}

    if (not torch.isnan(vel_target).all()) and (not torch.isnan(len_target).all()):
        velocity_loss = vel_reg_loss(vel_pred[~velmask], vel_target[~velmask])
        loss_vals["VelocityMSE"] = velocity_loss.item()
        length_loss = len_reg_loss(len_pred[~lenmask], len_target[~lenmask])
        loss_vals["LengthMSE"] = length_loss.item()

        total_loss = classification_loss + velocity_loss + length_loss
    else:
        total_loss = classification_loss

    return total_loss, loss_vals

optimizer = torch.optim.Adam(params = model.parameters())

epochs = 100

running_loss = {"ClassificationCEL":[], "VelocityMSE":[], "LengthMSE":[]}

# Define metrics
train_acc = torchmetrics.MetricCollection([
    torchmetrics.Accuracy(task="binary")])

test_acc = torchmetrics.MetricCollection([
    torchmetrics.Accuracy(task="binary")])

train_vel_reg = torchmetrics.MetricCollection([
    torchmetrics.MeanSquaredError(),
    torchmetrics.R2Score()])

test_vel_reg = torchmetrics.MetricCollection([
    torchmetrics.MeanSquaredError(),
    torchmetrics.R2Score()])

train_len_reg = torchmetrics.MetricCollection([
    torchmetrics.MeanSquaredError(),
    torchmetrics.R2Score()])

test_len_reg = torchmetrics.MetricCollection([
    torchmetrics.MeanSquaredError(),
    torchmetrics.R2Score()])

epoch_metrics = {}     # Needs to be implementer during wednesday 8th
for e in range(epochs):
    print(f"\nEpoch {e+1}/{epochs}")
    
    # Training section
    model.train()

    t0 = time.time()
    for i, data in enumerate(trainloader):
        tens, label_idx, velocity, length = data
        tens /= torch.max(tens)
        tens = tens.to(device)
        label_idx = label_idx.to(device)
        velocity = velocity.to(device)
        length = length.to(device)

        prediction = model(tens)
        pred_label, pred_vel, pred_len = prediction
        pred_idx = torch.argmax(torch.softmax(pred_label,dim=1),dim=1)
        #print(f"\nPredicted label dist: {pred_label}, {pred_label.shape}")
        #print(f"Predicted label: {pred_idx.flatten().tolist()}")
        #print(f"Predicted velocity: {pred_vel.flatten().tolist()}")
        #print(f"Predicted length: {pred_len.flatten().tolist()}")
        #print(f"Target label: {label_idx.flatten().tolist()}")
        #print(f"Target velocity: {velocity.flatten().tolist()}")
        #print(f"Target lenght: {length.flatten().tolist()}")
        target = (label_idx, velocity.unsqueeze(1), length.unsqueeze(1))

        loss, loss_dict = total_loss(prediction, target)
        for key in loss_dict:
            running_loss[key].append(loss_dict[key])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%20==0:
            print(f"{i+1}/{len(trainloader)}")
            for key in running_loss:
                if len(running_loss[key])==0:
                    continue
                else:
                    print(f"Avg. {key}:\t{sum(running_loss[key])}/{len(running_loss[key])} = {sum(running_loss[key])/len(running_loss[key])}")
                running_loss[key] = []
            print("")

    t1 = time.time()
    print(f"Time for epoch: {t1-t0} seconds")

    # Testing section
    model.eval()
    
    metrics = {}
    # Get metrics for epoch and save model
    predicted_labels = []
    predicted_velocities = []
    predicted_lengths = []
    target_labels = []
    target_velocities = []
    target_lengths = []
    counter = 0
    for i, data in enumerate(trainloader):
        with torch.no_grad():
            tens, label_idx, velocity, length = data
            tens /= torch.max(tens)
            tens = tens.to(device)
            label_idx = label_idx.to(device)
            velocity = velocity.to(device)
            length = length.to(device)

            prediction = model(tens)
            pred_label, pred_vel, pred_len = prediction
            pred_idx = torch.argmax(torch.softmax(pred_label,dim=1),dim=1)

            predicted_labels += pred_idx.flatten().tolist()
            predicted_velocities += pred_vel.flatten().tolist()
            predicted_lengths += pred_len.flatten().tolist()
            target_labels += label_idx.flatten().tolist()
            target_velocities += velocity.flatten().tolist()
            target_lengths += length.flatten().tolist()
            #print(f"Predicted label: {pred_label}")
            #print(f"Predicted velocity: {pred_velocity}")
            #print(f"Predicted length: {pred_length}")
            #print(f"Target label: {label_idx}")
            #print(f"Target velocity: {velocity}")
            #print(f"Target lenght: {length}")
            
            train_acc.update(pred_idx, label_idx)
            if (not torch.isnan(velocity).all()) and (not torch.isnan(length).all()):
                train_vel_reg.update(pred_vel, velocity.unsqueeze(1))
                train_len_reg.update(pred_len, length.unsqueeze(1))

    computed_acc = train_acc.compute()
    computed_vel = train_vel_reg.compute()
    computed_len = train_len_reg.compute()
    acc_dict = {}
    vel_dict = {}
    len_dict = {}
    for key in computed_acc.keys():
        print(key)
        print(computed_acc[key])
        acc_dict[key] = computed_acc[key].item()
    for key in computed_vel.keys():
        vel_dict[key] = computed_vel[key].item()
        len_dict[key] = computed_len[key].item()

    metrics["train"] = {"pred_label":predicted_labels,
                        "target_label":target_labels,
                        "pred_vel":predicted_velocities,
                        "target_vel":target_velocities,
                        "pred_len":predicted_lengths,
                        "target_len":target_lengths,
                        "acc":acc_dict,
                        "vel_reg":vel_dict,
                        "len_reg":len_dict
                        }

    print(f"Train Acc: {computed_acc}")
    print(f"Train Velocity: {computed_vel}")
    print(f"Train Length: {computed_len}")
    print(metrics["train"])

    # Reset metric
    train_acc.reset()
    train_vel_reg.reset()
    train_len_reg.reset()

    predicted_labels = []
    predicted_velocities = []
    predicted_lengths = []
    target_labels = []
    target_velocities = []
    target_lengths = []
    counter = 0
    for i, data in enumerate(testloader):
        with torch.no_grad():
            tens, label_idx, velocity, length = data
            tens /= torch.max(tens)
            tens = tens.to(device)
            label_idx = label_idx.to(device)
            velocity = velocity.to(device)
            length = length.to(device)

            prediction = model(tens)
            pred_label, pred_vel, pred_len = prediction
            pred_idx = torch.argmax(torch.softmax(pred_label,dim=1),dim=1)

            predicted_labels += pred_idx.flatten().tolist()
            predicted_velocities += pred_vel.flatten().tolist()
            predicted_lengths += pred_len.flatten().tolist()
            target_labels += label_idx.flatten().tolist()
            target_velocities += velocity.flatten().tolist()
            target_lengths += length.flatten().tolist()
            
            test_acc.update(pred_idx, label_idx)
            if (not torch.isnan(velocity).all()) and (not torch.isnan(length).all()):
                test_vel_reg.update(pred_vel, velocity.unsqueeze(1))
                test_len_reg.update(pred_len, length.unsqueeze(1))

    computed_acc = test_acc.compute()
    computed_vel = test_vel_reg.compute()
    computed_len = test_len_reg.compute()
    acc_dict = {}
    vel_dict = {}
    len_dict = {}
    for key in computed_acc.keys():
        #print(key)
        #print(computed_acc[key])
        acc_dict[key] = computed_acc[key].item()
    for key in computed_vel.keys():
        vel_dict[key] = computed_vel[key].item()
        len_dict[key] = computed_len[key].item()

    metrics["train"] = {"pred_label":predicted_labels,
                        "target_label":target_labels,
                        "pred_vel":predicted_velocities,
                        "target_vel":target_velocities,
                        "pred_len":predicted_lengths,
                        "target_len":target_lengths,
                        "acc":acc_dict,
                        "vel_reg":vel_dict,
                        "len_reg":len_dict
                        }

    print(f"Test Acc: {computed_acc}")
    print(f"Test Velocity: {computed_vel}")
    print(f"Test Length: {computed_len}")

    # Reset metric
    test_acc.reset()
    test_vel_reg.reset()
    test_len_reg.reset()

    epoch_metrics[str(e+1)] = metrics

    with open("epoch_metrics.json", "w", encoding='utf-8') as outfile:
        json.dump(epoch_metrics, outfile, indent=2)
