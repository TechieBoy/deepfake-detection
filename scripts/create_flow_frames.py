from pwc_net import convert_to_optical_flow
import os
from itertools import combinations
from tqdm import tqdm
base_dir = '/home/teh_devs/deepfake/of/'

for folder in tqdm(sorted(os.listdir(base_dir))[12263:], ncols=0):
    try:
        print(folder)
        image_folder = os.path.join(base_dir, folder)
        images = os.listdir(image_folder)
        nums = sorted([int(i.split('.')[0].split('_')[-1]) for i in images])
        if nums:
            base_num = nums[0]
            gset = []
            all_sets = []
            for num in nums:
                if num - base_num < 5:
                    gset.append(num)
                else:
                    base_num = num
                    all_sets.append(gset)
                    gset = []
                    gset.append(num)
            else:
                all_sets.append(gset)

            for l in all_sets:
                for comb in combinations(l, 2):
                    if comb[0] >= comb[1]:
                        continue
                    first_file = os.path.join(image_folder, f'{folder}_face_{comb[0]}.jpg')
                    second_file = os.path.join(image_folder, f'{folder}_face_{comb[1]}.jpg')
                    flo_file = os.path.join(image_folder, f'{folder}_{comb[0]}_{comb[1]}.flo')
                    png_file = os.path.join(image_folder, f'{folder}_{comb[0]}_{comb[1]}.png')
                    convert_to_optical_flow(first_file, second_file, flo_file, png_file)
    except Exception as e:
        print(e)
        continue

