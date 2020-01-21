from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from annoy import AnnoyIndex
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from shutil import move
import os
import pandas as pd
with open('../scripts/fakes.txt', 'r') as txt:
    fakes = txt.readlines()

with open('../scripts/real.txt', 'r') as txt:
    reals = txt.readlines()

fakes = fakes[1:]
reals = reals[1:]

# fakes.extend(reals)
print(len(reals))

index_size = 512
t = AnnoyIndex(index_size, "euclidean")
resnet = InceptionResnetV1(pretrained="vggface2").eval()

img_map = {}

for i, real in enumerate(tqdm(reals)):
    img_map[i] = real
    # img = Image.open(real.strip())
    # img = img.resize((160, 160))
    # img = transforms.ToTensor()(img)

    # img_embedding = resnet(img.unsqueeze(0)).detach().flatten().tolist()
    # t.add_item(i, img_embedding)

# t.build(5000)
# t.save('real_5000_trees.ann')

t.load('real_5000_trees.ann')
# min_so_far = 0.8
# min_i = 0
# for i in tqdm(range(len(reals))): 
#     items = t.get_nns_by_item(i, 475, 5000000, True)
#     if items[1][-1] < min_so_far:
#         min_so_far = items[1][-1]
#         min_i = i
#         print()
#         print(i, min_so_far)
# print("Mininum is ", min_i, "distance ", min_so_far)
items = t.get_nns_by_item(1900, 475, 5000000)
w=15
h=15
fig=plt.figure(figsize=(8, 8))
columns = 10
rows = 8
for i in range(1, 77):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(plt.imread(img_map[items[i]].strip()))
plt.show()

real_vids = []
for i in range(1, 77):
    real_vids.append(img_map[items[i]].split('/')[-1].split('_')[0] + '.mp4')

corresponding_fakes = []

df = pd.read_csv('~/deepfake/raw/combined_metadata.csv')

for vid in real_vids:
    corresponding_fakes.extend(df[df.original == vid]['index'].tolist())


dest_real = '/data/deepfake/seperated/real/'
dest_fake = '/data/deepfake/seperated/fake/'

for folder in [dest_fake, dest_real]:
    if not os.path.isdir(folder):
        os.path.mkdir(folder)

base_folder = '/data/deepfake/'

for vid in real_vids:
    vid_name = vid.split('.')[0]
    for i in range(100):
        vid_path = os.path.join(base_folder, 'real', vid_name + f'_face_{i}.jpg')
        if os.path.isfile(vid_path):
            move(vid_path, dest_real)

for vid in corresponding_fakes:
    vid_name = vid.split('.')[0]
    for i in range(100):
        vid_path = os.path.join(base_folder, 'fake', vid_name + f'_face_{i}.jpg')
        if os.path.isfile(vid_path):
            move(vid_path, dest_fake)