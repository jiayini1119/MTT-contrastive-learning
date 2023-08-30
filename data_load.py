import torch
import random

random.seed(42)
torch.manual_seed(42)

file_path = '/home/jennyni/data/2023-03-0818:37:28.795937-distill-cifar10.pt'
loaded_data = torch.load(file_path)


syn_data_dict = loaded_data['syn_dataset']

# print(syn_data_dict)

selected_images_list = []

for epoch, class_id in syn_data_dict.keys():
    images = syn_data_dict[(epoch, class_id)]
    indices = list(range(images.shape[0]))
    random_indices = random.sample(indices, 10)
    selected_images = images[random_indices]
    selected_images_list.append(selected_images)

selected_images_tensor = torch.cat(selected_images_list, dim=0)

print(selected_images_tensor.shape)

torch.save(selected_images_tensor, './benchmark_10.pt')
