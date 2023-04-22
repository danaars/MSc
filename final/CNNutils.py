import os
import time
import json
import torch
import torchmetrics

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

def init_metrics(device):
    train_acc = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task="multiclass", num_classes=4),
        torchmetrics.ConfusionMatrix(task="multiclass", num_classes=4)]).to(device)

    test_acc = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task="multiclass", num_classes=4),
        torchmetrics.ConfusionMatrix(task="multiclass", num_classes=4)]).to(device)

    train_vel_reg = torchmetrics.MetricCollection([
        torchmetrics.MeanSquaredError(),
        torchmetrics.R2Score()]).to(device)

    test_vel_reg = torchmetrics.MetricCollection([
        torchmetrics.MeanSquaredError(),
        torchmetrics.R2Score()]).to(device)

    train_len_reg = torchmetrics.MetricCollection([
        torchmetrics.MeanSquaredError(),
        torchmetrics.R2Score()]).to(device)

    test_len_reg = torchmetrics.MetricCollection([
        torchmetrics.MeanSquaredError(),
        torchmetrics.R2Score()]).to(device)

    return train_acc, test_acc, train_vel_reg, test_vel_reg, train_len_reg, test_len_reg

def train_model(model,
                device,
                epochs,
                epochtrainloader,
                metricstrainloader,
                metricstestloader,
                loss,
                optimizer,
                metrics=True,
                save_metrics=True,
                printinterval = 20
                ):

    model = model.double()
    optimizer

    if metrics:
        train_acc, test_acc, train_vel_reg, test_vel_reg, train_len_reg, test_len_reg =\
                init_metrics(device)

    running_loss = {"ClassificationCEL":[], "VelocityMSE":[], "LengthMSE":[]}
    epoch_metrics = {}

    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}")
        
        # Training section
        model.train()

        t0 = time.time()
        for i, data in enumerate(epochtrainloader):
            tens, label_idx, velocity, length = data

            tens = tens.to(device)
            label_idx = label_idx.to(device)
            velocity = velocity.to(device)
            length = length.to(device)

            prediction = model(tens)
            pred_label, pred_vel, pred_len = prediction
            pred_idx = torch.argmax(torch.softmax(pred_label,dim=1),dim=1)

            target = (label_idx, velocity.unsqueeze(1), length.unsqueeze(1))

            loss, loss_dict = total_loss(prediction, target)

            for key in loss_dict:
                running_loss[key].append(loss_dict[key])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%printinterval==0:
                print(f"{i+1}/{len(epochtrainloader)}")
                for key in running_loss:
                    if len(running_loss[key])==0:
                        continue
                    else:
                        print(f"Avg. {key}:\t{sum(running_loss[key])}/{len(running_loss[key])} = {sum(running_loss[key])/len(running_loss[key])}")
                    running_loss[key] = []
                print("")

        t1 = time.time()
        print(f"Time for epoch: {t1-t0} seconds")

        if metrics:
            model.eval()

            metrics = {}
            # Compute metrics for epoch
            for i, data in enumerate(metricstrainloader):
                with torch.no_grad():
                    tens, label_idx, velocity, length = data
                    tens = tens.to(device)
                    label_idx = label_idx.to(device)
                    velocity = velocity.to(device)
                    length = length.to(device)

                    prediction = model(tens)
                    pred_label, pred_vel, pred_len = prediction
                    pred_idx = torch.argmax(torch.softmax(pred_label,dim=1),dim=1)

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
                try:
                    acc_dict[key] = computed_acc[key].item()
                except ValueError as err:
                    acc_dict[key] = computed_acc[key].tolist()
            for key in computed_vel.keys():
                vel_dict[key] = computed_vel[key].item()
                len_dict[key] = computed_len[key].item()

            metrics["train"] = {"acc":acc_dict,
                                "vel_reg":vel_dict,
                                "len_reg":len_dict
                                }

            print(f"Train Acc: {computed_acc}")
            print(f"Train Velocity: {computed_vel}")
            print(f"Train Length: {computed_len}")

            # Reset metric and compute test
            train_acc.reset()
            train_vel_reg.reset()
            train_len_reg.reset()

            for i, data in enumerate(metricstestloader):
                with torch.no_grad():
                    tens, label_idx, velocity, length = data
                    tens = tens.to(device)
                    label_idx = label_idx.to(device)
                    velocity = velocity.to(device)
                    length = length.to(device)

                    prediction = model(tens)
                    pred_label, pred_vel, pred_len = prediction
                    pred_idx = torch.argmax(torch.softmax(pred_label,dim=1),dim=1)

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
                try:
                    acc_dict[key] = computed_acc[key].item()
                except ValueError as err:
                    acc_dict[key] = computed_acc[key].tolist()
            for key in computed_vel.keys():
                vel_dict[key] = computed_vel[key].item()
                len_dict[key] = computed_len[key].item()

            metrics["test"] = {"acc":acc_dict,
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

            if save_metrics:
                # Save state dict
                if not os.path.exists("models"):
                    os.makedirs("models")

                with open("epoch_metrics.json", "w", encoding='utf-8') as outfile:
                    json.dump(epoch_metrics, outfile, indent=2)

                statename = os.path.join("models", f"epoch{e+1}.pth")
                torch.save(model.state_dict(), statename)
