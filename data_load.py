# import torch
# import random

# random.seed(42)
# torch.manual_seed(42)

# file_path = '/home/jennyni/data/2023-03-0818:37:28.795937-distill-cifar10.pt'
# loaded_data = torch.load(file_path)


# syn_data_dict = loaded_data['syn_dataset']

# # print(syn_data_dict)

# selected_images_list = []

# for epoch, class_id in syn_data_dict.keys():
#     images = syn_data_dict[(epoch, class_id)]
#     indices = list(range(images.shape[0]))
#     random_indices = random.sample(indices, 10)
#     selected_images = images[random_indices]
#     selected_images_list.append(selected_images)

# selected_images_tensor = torch.cat(selected_images_list, dim=0)

# print(selected_images_tensor.shape)

# torch.save(selected_images_tensor, './benchmark_10.pt')



# import pickle
# import torch
# from utils.data_util import get_datasets

# with open("/home/jennyni/data/clcore-subset-idx.pkl", "rb") as f:
#     data = pickle.load(f)

# class_wise_subset = data["class_idx"]

# dataset = "cifar10"

# ori_datasets = get_datasets(dataset, need_train_ori=True)
# trainset = ori_datasets.trainset_ori

# num_top_pictures = 10

# selected_tensors = []

# for class_idx, indices in class_wise_subset.items():
#     top_indices = indices[-num_top_pictures:]

#     for idx in top_indices:
#         image, _ = trainset[idx]
#         selected_tensors.append(image)

# concatenated_tensor = torch.stack(selected_tensors)

# torch.save(concatenated_tensor, 'bad_ipc_10.pt')



import torch
import random

random.seed(42)
torch.manual_seed(42)

file_path = '/home/jennyni/MTT-contrastive-learning/images_best.pt'
loaded_data = torch.load(file_path)

print(loaded_data)




# syn_data_dict = loaded_data['syn_dataset']

# # print(syn_data_dict)

# selected_images_list = []

# for epoch, class_id in syn_data_dict.keys():
#     images = syn_data_dict[(epoch, class_id)]
#     indices = list(range(images.shape[0]))
#     random_indices = random.sample(indices, 10)
#     selected_images = images[random_indices]
#     selected_images_list.append(selected_images)

# selected_images_tensor = torch.cat(selected_images_list, dim=0)

# print(selected_images_tensor.shape)

# torch.save(selected_images_tensor, './benchmark_10.pt')


