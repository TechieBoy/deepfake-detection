from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from annoy import AnnoyIndex
from glob import glob
from tqdm import tqdm

with open('../scripts/fakes.txt', 'r') as txt:
    fakes = txt.readlines()

with open('../scripts/real.txt', 'r') as txt:
    reals = txt.readlines()

fakes = fakes[1:]
reals = reals[1:]

# fakes.extend(reals)

index_size = 512
t = AnnoyIndex(index_size, "euclidean")
resnet = InceptionResnetV1(pretrained="vggface2").eval()

img_map = {}

for i, real in enumerate(tqdm(reals)):
    img_map[i] = real
    img = Image.open(real.strip())
    img = img.resize((160, 160))
    img = transforms.ToTensor()(img)

    img_embedding = resnet(img.unsqueeze(0)).detach().flatten().tolist()
    t.add_item(i, img_embedding)

t.build(5000)
t.save('real_5000_trees.ann')

# fig = plt.figure(figsize=(15, 7))
# gs = fig.add_gridspec(2, 6)
# ax1 = fig.add_subplot(gs[0:2, 0:2])
# ax2 = fig.add_subplot(gs[0, 2])
# ax3 = fig.add_subplot(gs[0, 3])
# ax4 = fig.add_subplot(gs[0, 4])
# ax5 = fig.add_subplot(gs[0, 5])
# ax6 = fig.add_subplot(gs[1, 2])
# ax7 = fig.add_subplot(gs[1, 3])
# ax8 = fig.add_subplot(gs[1, 4])
# ax9 = fig.add_subplot(gs[1, 5])
# axx = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
# list_plot = [img_map[0].strip()] + [imp_map[i].strip() for i in nearest]
# for i, ax in enumerate(axx):
#     ax.imshow(plt.imread(list_plot[i]))
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)