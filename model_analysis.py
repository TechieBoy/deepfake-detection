import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_recall_fscore_support,
    log_loss,
    classification_report
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mesonet import MesoInception4
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from IPython.core.debugger import set_trace

num_classes = 2

device = torch.device("cuda:0")
data_dir = "/home/teh_devs/deepfake/dataset/test"
data_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

test_dataset = datasets.ImageFolder(data_dir, data_transform)
testloader = DataLoader(
    test_dataset, batch_size=1120, shuffle=True, num_workers=78, pin_memory=True
)
model = MesoInception4(2)
model = model.to(device)
model.load_state_dict(torch.load("saved_models/meso.pt"))

probabilites = torch.zeros(0, dtype=torch.float32, device="cpu")
predictions = torch.zeros(0, dtype=torch.long, device="cpu")
true_val = torch.zeros(0, dtype=torch.long, device="cpu")

with torch.no_grad():
    for i, (inputs, classes) in enumerate(tqdm(testloader)):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        softmaxed = F.softmax(outputs, dim=1)
        probs = softmaxed[:, 1]
        _, preds = torch.max(softmaxed, 1)
        probabilites = torch.cat([probabilites, probs.view(-1).cpu()])
        predictions = torch.cat([predictions, preds.view(-1).cpu()])
        true_val = torch.cat([true_val, classes.view(-1).cpu()])

true_val = true_val.numpy()
predictions = predictions.numpy()
probabilites = probabilites.numpy()
conf_mat = confusion_matrix(true_val, predictions)
print(conf_mat)


sns.set(rc={"figure.figsize": (11.7, 8.27)})
ax = plt.subplot()
sns.heatmap(conf_mat, fmt="g", cmap="Greens", annot=True, ax=ax)
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
classes = ["fake", "real"]
ax.set(yticklabels=classes, xticklabels=classes)
# Bug in latest matplotlib :/
ax.set_ylim(2.0, 0)
plt.show()


sns.distplot(probabilites)
# print("Log loss")
# print(log_loss(true_val, probabilites))

print("ROC AUC Curve")
fpr, tpr, _ = roc_curve(true_val, probabilites)
auc = roc_auc_score(true_val, probabilites)
plt.plot(fpr, tpr, label="auc=" + str(auc))
plt.legend(loc=4)
plt.show()

print(classification_report(true_val, predictions, target_names=classes))


# Reference

"""
The accuracy of the model is basically the total number of correct predictions divided by total number
of predictions. The precision of a class define how trustable is the result when the model answer that a point
belongs to that class. The recall of a class expresses how well the model is able to detect that class.
The F1 score of a class is given by the harmonic mean of precision and recall.

1 - recall is how many of that class you will miss.

"""

"""
high recall + high precision : the class is perfectly handled by the model
low recall + high precision : the model can’t detect the class well but is highly trustable when it does
high recall + low precision : the class is well detected but the model also include points of other classes in it
low recall + low precision : the class is poorly handled by the model
"""

