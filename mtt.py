"""
Data Distillation by Matching Training Trajectories for Contrastive Learning
"""
import argparse
from utils.data_util import *
import numpy as np
from convNet import *
from torch.utils.data import random_split
import torch.optim as optim

from sas_cl.projection_heads.critic import LinearCritic
from sas_cl.trainer import Trainer

import glob

def main(args):
    """
    I have a ckpt folder with structure:
    trajectory 0:
      - trajectory_0_epoch_0
      - trajectory_0_epoch_1
      - trajectory_0_epoch_2

    
    trajectory 1:
      - trajectory_1_epoch_0
      - trajectory_1_epoch_1
      - trajectory_1_epoch_2
    
    trajectory 2:
      - trajectory_2_epoch_0
      - trajectory_2_epoch_1
      - trajectory__epoch_2

    
    Those are saved duirng training the expert trajectory

    mll.py:
    Initialize Syn dataset D_syn (TODO)
    Initialize learning rate(not trainable, fixed now)

    optimizer for D_syn

    for i in range(distillation_step + 1):
    
        Sample expert trajectory 
        Choose random start epoch < max_start_epoch => (initial_trajectory)
        Initialize student network with expert params from initial_trajectory
        Get trainset, clfset, and testset from D_syn (TODO, make a class)

        for j in range(N):
            sample a minibatch from the  D_syn 
            simclr training loop
        
        get target params
        loss between student params and expert params
        update D_sync with respect to the loss
    
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize Syn dataset D_syn
    ori_datasets = get_datasets(args.dataset)

    dataset_images = nn.Parameter(torch.randn(ori_datasets.num_classes * args.ipc, ori_datasets.channel, ori_datasets.img_size, ori_datasets.img_size))

    labels = torch.cat([torch.tensor([i] * args.ipc) for i in range(ori_datasets.num_classes)])

    kornia_augmentations = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((32, 32), scale=(0.08, 1.0), same_on_batch=True, keepdim=True),
        kornia.augmentation.RandomHorizontalFlip(same_on_batch=True, keepdim=True),
        kornia.augmentation.ColorJiggle(0.4, 0.4, 0.4, 0.1, same_on_batch=True, p=0.8, keepdim=True),
        kornia.augmentation.RandomGrayscale(same_on_batch=True, p=0.2, keepdim=True),
    )
    
    distilled_dataset = CustomDataset(dataset_images, labels, transform=kornia_augmentations)

    optimizer_img = torch.optim.SGD([dataset_images], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()

    for i in range(args.distillation_step + 1):

        # sample an expert trajectory
        subdirectories = [d for d in os.listdir('ckpt') if os.path.isdir(os.path.join('ckpt', d))]

        selected_directory = np.random.choice(subdirectories)

        trajectories = os.path.join('ckpt', selected_directory)
        matching_files = glob.glob(os.path.join(trajectories, f"{selected_directory}_epoch_*.pt"))

        initial_trajectory = np.random.choice([f for f in matching_files if int(f.split('_epoch_')[-1].split('.pt')[0]) < args.max_start_epoch])

        # Initialize student network with expert params from the expert trajectory
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

        net.load_state_dict(torch.load(initial_trajectory))

        num_params = sum([np.prod(p.size()) for p in (net.parameters())])
        starting_params = [net.state_dict()[key] for key in net.state_dict()]

        print("Assigned the initial weight to the student network")

        # get train, test, clf data
        total_length = len(distilled_dataset)
        train_length = int(total_length * 0.8)
        test_length = int(total_length * 0.1)
        clf_length = total_length - train_length - test_length

        trainset, testset, clfset = random_split(distilled_dataset, [train_length, test_length, clf_length])

        critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

        optimizer_simclr = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6)
        if args.dataset == SupportedDatasets.TINY_IMAGENET.value:
            optimizer_simclr = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=2 * args.lr, weight_decay=1e-6)

        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        clftrainloader = torch.utils.data.DataLoader(
            dataset=clfset,
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

        testloader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True,
        )

        param_loss_list = []
        param_dist_list = []

        print("***Start simclr training***")
        net = net.to(device)
        critic = critic.to(device)

        # synclr training
        trainer = Trainer(
            device=device,
            distributed=False,
            net=net,
            critic=critic,
            trainloader=trainloader,
            clftrainloader=clftrainloader,
            testloader=testloader,
            num_classes=ori_datasets.num_classes,
            optimizer=optimizer_simclr,
        )

        for epoch in range(0, args.syn_steps):
            print(f"step: {epoch}")
            train_loss = trainer.train()
            print(f"train_loss: {train_loss}")
            test_acc = trainer.test()
            print(f"test_acc: {test_acc}")
        

        # After training - udpate Dsync
        print("Training of the student network finished")

        # Get target params
        initial_epoch_num = int(initial_trajectory.split('_epoch_')[-1].split('.pt')[0])
        final_epoch_num = initial_epoch_num + args.expert_epochs

        final_trajectory_name = f"{selected_directory}_epoch_{final_epoch_num}.pt"
        final_trajectory_path = os.path.join(trajectories, final_trajectory_name)

        if os.path.exists(final_trajectory_path):
            print(f"Found final trajectory at {final_trajectory_path}")
        else:
            print(f"Final trajectory does not exist for epoch {final_epoch_num}")

        final_state_dict = torch.load(final_trajectory_path)
        current_state_dict = net.state_dict()

        target_params = [final_state_dict[key] for key in final_state_dict]
        student_params = [current_state_dict[key] for key in current_state_dict]

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()

        grand_loss.backward()

        optimizer_img.step()

        for _ in student_params:
            del _


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mtt contrastive learning data distillation process')

    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--distillation_step', type=int, default=10, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--lr', type=float, default=1e-05, help='learning rate for updating simclr learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument("--batch-size", type=int, default=1024, help='Training batch size for inner loop')
        
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')

    args = parser.parse_args()

    main(args)