import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import time
import copy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x * (torch.tanh(self.softplus(x)))
        return x


class BaseNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 128)
        self.mish = Mish()
        self.dropout_50 = nn.Dropout(0.7)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 128)
        self.output = nn.Linear(64, output_size)

    def forward(self, input_data):
        x = self.input_layer(input_data)
        x = self.dropout_50(self.mish(x))
        x = self.hidden1(x)
        x = self.dropout_50(self.mish(x))
        # x = self.hidden2(x)
        # x = self.dropout_50(self.mish(x))
        return self.output(x)


# GPU
device = torch.device("cuda:0")
cudnn.benchmark = True
cudnn.deterministic = True

scaler = StandardScaler(copy=False)


def load_data():
    real_df = pd.read_csv(
        "feat/real.txt",
        delimiter=" ",
        header=None,
        names=["a", "b", "c", "d", "e", "f"],
        index_col=False,
        float_precision="high",
    )

    real_df["target"] = 1

    fake_df = pd.read_csv(
        "feat/fake.txt",
        delimiter=" ",
        header=None,
        names=["a", "b", "c", "d", "e", "f"],
        index_col=False,
        float_precision="high",
    )
    fake_df["target"] = 0

    real_df.head()

    df = pd.concat([real_df, fake_df], ignore_index=True, sort=False)

    del real_df, fake_df

    y = df.target.values
    df = df.drop("target", axis="columns").values
    return df, y


data, target = load_data()
scaler = scaler.fit(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.15, random_state=12
)

train_dataset = TensorDataset(
    torch.from_numpy(scaler.transform(X_train)), torch.from_numpy(y_train)
)
test_dataset = TensorDataset(
    torch.from_numpy(scaler.transform(X_test)), torch.from_numpy(y_test)
)
datasets = {"train": train_dataset, "test": test_dataset}
dataloaders = {
    x: DataLoader(datasets[x], shuffle=False, num_workers=75, batch_size=8192, pin_memory=True)
    for x in ["train", "test"]
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                inputs = inputs.float()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # if phase == 'test':
                    #     print(preds[0])
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "test":
                scheduler.step(epoch_loss)

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"hp_epoch_{epoch}.pt")

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = BaseNet(6, 2)
model = model.float()
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.000001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
model.load_state_dict(torch.load('hpo.pt'))
model = train_model(model, criterion, optimizer, scheduler, 50)

torch.save(model.state_dict(), "hpo.pt")