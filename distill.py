"""
Data Distillation by Matching Training Trajectories for Self-Supervised Contrastive Learning
"""
import argparse
import copy
import glob
import wandb
from datetime import datetime
from utils.reparam_module import *
from utils.supported_dataset import *
from utils.custom_dataset import get_init_syn_data
import numpy as np
from models.networks.convNet import *
from utils.augmentation import KorniaAugmentation
import torch.optim as optim
from utils.random import Random
from models.projection_heads.critic import LinearCritic
from trainer import SynTrainer

def main(args):

    wandb.init(
        project="data-distillation-by-mtt-contrastive-learning",
        config=args
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    DT_STRING = "".join(str(datetime.now()).split())

    ori_datasets = get_datasets(args.dataset)

    testset = ori_datasets.testset

    clfset = ori_datasets.clftrainset

    ##############################################################
    # Initialize Synthetic Dataset and Synthetic lr
    ##############################################################

    trainset_images = get_init_syn_data(method=args.syn_init_method, dataset=args.dataset, ipc=args.ipc, path=args.path)
    # torch.save(trainset_images.cpu(), f"random_100_real_initial_images.pt")

    labels = torch.cat([torch.tensor([i] * args.ipc) for i in range(ori_datasets.num_classes)])
    torch.save(labels.cpu(), "label.pt")

    trainset_images = trainset_images.detach().to(device).requires_grad_(True)
    syn_lr = torch.tensor(args.lr_teacher).to(device)

    if args.optimizer_img == "Adam":
        optimizer_img = optim.Adam([trainset_images], lr=args.lr_img, weight_decay=1e-4)
    else:
        optimizer_img = optim.SGD([trainset_images], lr=args.lr_img, momentum=0.5)

    syn_lr = syn_lr.detach().to(device).requires_grad_(True)

    optimizer_lr = optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    optimizer_img.zero_grad()
    
    for i in range(args.distillation_step + 1):

        ##############################################################
        # Load initial parameters
        ##############################################################

        trajectories_dir = os.path.join(f'checkpoint_{args.dataset}')
        subdirectories = [d for d in os.listdir(trajectories_dir) if os.path.isdir(os.path.join(trajectories_dir, d))]
        selected_directory = np.random.choice(subdirectories) # randomly choose an expert trajectory
        trajectories = os.path.join(trajectories_dir, selected_directory)
        matching_files = glob.glob(os.path.join(trajectories, 'net', f"{selected_directory}_epoch_*.pt"))
        if len(matching_files) < 0:
            raise ValueError("no matching files")

        initial_trajectory_epoch = np.random.choice([int(f.split('_epoch_')[-1].split('.pt')[0]) for f in matching_files if int(f.split('_epoch_')[-1].split('.pt')[0]) < args.max_start_epoch])

        # network initial parameter
        net_checkpoint = os.path.join(trajectories, 'net', f'{selected_directory}_epoch_{initial_trajectory_epoch}.pt')
        if os.path.exists(net_checkpoint):
            initial_trajectory_dict = torch.load(net_checkpoint)
        else:
            raise ValueError("net checkpoint does not exist")
        
        # critic initial parameter
        projection_head_checkpoint = os.path.join(trajectories, 'projection_head', f'{selected_directory}_epoch_{initial_trajectory_epoch}.pt')
        if os.path.exists(projection_head_checkpoint):
            initial_critic_trajectory_dict = torch.load(projection_head_checkpoint)
        else:
            raise ValueError("projection head checkpoint does not exist")

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=i)

        ##############################################################
        # Encoder
        ##############################################################
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        ori_net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, channel=ori_datasets.channel, add_bn=False).to(device)
        ori_net_cp = copy.deepcopy(ori_net)

        net = ReparamModule(ori_net)
        # num_params = sum([np.prod(p.size()) for p in (net.parameters())])

        ori_net_cp.load_state_dict(initial_trajectory_dict)

        student_params_list = list(ori_net_cp.parameters())
        for param in student_params_list:
            param.detach_()

        starting_params = torch.cat([p.data.to(device).reshape(-1) for p in student_params_list], 0)
        student_params = [torch.cat([p.data.to(device).reshape(-1) for p in student_params_list], 0).requires_grad_(True)]

        ##############################################################
        # Critic
        ##############################################################

        ori_critic = LinearCritic(ori_net_cp.representation_dim, temperature=args.temperature).to(device)
        ori_critic_cp = copy.deepcopy(ori_critic)

        critic = ReparamModule(ori_critic)

        ori_critic_cp.load_state_dict(initial_critic_trajectory_dict)

        critic_params_list = list(ori_critic_cp.parameters())
        for param in critic_params_list:
            param.detach_()
        
        student_params_critic = [torch.cat([p.data.to(device).reshape(-1) for p in critic_params_list], 0).requires_grad_(True)]

        ##############################################################
        # Data Loaders
        ##############################################################

        clftrainloader = torch.utils.data.DataLoader(
            dataset=clfset,
            batch_size=args.test_batch_size, 
            shuffle=False, 
            num_workers=6, 
            pin_memory=True
        )

        testloader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=args.test_batch_size,
            shuffle=False, 
            num_workers=6,
            pin_memory=True,
        )

        ##############################################################
        # SimCLR training
        ##############################################################

        print("***Start simclr training***")

        kornia_augmentations = KorniaAugmentation(dataset=args.dataset).to(device)

        bn_layer = torch.nn.BatchNorm2d(ori_datasets.channel).to(device)

        if args.batch_size is None:
            args.batch_size = ori_datasets.num_classes * args.ipc

        trainer = SynTrainer(
            trainset_images=trainset_images,
            batch_size=args.batch_size,
            n_augmentations=2,
            transform=kornia_augmentations,
            student_params=student_params,
            student_params_critic=student_params_critic,
            syn_lr = syn_lr,
            reparam_net = net,
            reparam_critic = critic,
            device=device,
            distributed=False,
            net=ori_net,
            critic=ori_critic,
            clftrainloader=clftrainloader,
            testloader=testloader,
            num_classes=ori_datasets.num_classes,
            reg_weight=args.reg_weight,
            bn_layer=bn_layer
        )

        param_loss_list = []
        param_dist_list = []

        ##############################################################
        # Main Training Loop
        ##############################################################

        indices_chunks = []

        for epoch in range(0, args.syn_steps):
            print(f"step: {epoch}")
            
            if not indices_chunks:
                indices = torch.randperm(len(trainset_images))
                indices_chunks = list(torch.split(indices, args.batch_size))
            
            these_indices = indices_chunks.pop()

            train_loss = trainer.train(these_indices)
            
            print(f"train_loss: {train_loss}")
            # wandb.log(
            #     data={"train": {
            #     "loss": train_loss,
            #     }},
            #     step=epoch
            # )


            # if (args.test_freq > 0) and (epoch + 1) % args.test_freq == 0:
            #     test_acc = trainer.test()
            #     print(f"test_acc: {test_acc}")
            #     wandb.log(
            #         data={"test": {
            #         "acc": test_acc,
            #         }},
            #         step=epoch
            #     )
        
        ##############################################################
        # Update Synthetic Dataset
        ##############################################################

        print("Training of the student network finished")

        ##############################################################
        # Get target Parameters
        ##############################################################
        final_epoch_num = initial_trajectory_epoch + args.expert_epochs

        final_trajectory_name = f"{selected_directory}_epoch_{final_epoch_num}.pt"
        final_trajectory_path = os.path.join(trajectories, 'net', final_trajectory_name)

        if os.path.exists(final_trajectory_path):
            print(f"Found final trajectory at {final_trajectory_path}")
        else:
            raise ValueError(f"Final trajectory does not exist for epoch {final_epoch_num}")

        final_state_dict = torch.load(final_trajectory_path)
        target_params = [final_state_dict[key] for key in final_state_dict]
        target_params = torch.cat([p.data.to(device).reshape(-1) for p in target_params], 0)

        ##############################################################
        # Calculate Synthetic Data Loss
        ##############################################################

        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        print("param_loss", param_loss)
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        print("param_dist: ", param_dist)

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        wandb.log(
            {
                "param_loss": param_loss.detach().cpu(),
                "param_dist": param_dist.detach().cpu()
            },
            step=i
        )

        # param_loss /= num_params
        # param_dist /= num_params

        param_loss /= param_dist
        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        print(trainset_images.grad)
        grad_avg = torch.mean(abs(trainset_images.grad)).item()
        wandb.log(
            {
                "syn_data_grad": {
                    "grad_average": grad_avg
                }
            },
            step=i
        )

        if args.verbose:
            grad_avg = torch.mean(trainset_images.grad).item()
            with open(f"real_grad_averages_vs_ipc.txt", "a") as f:
                f.write(f"IPC: {args.ipc}, Gradient Average: {grad_avg}\n")

            with open(f"real_grad_averages_vs_N.txt", "a") as f:
                f.write(f"N: {args.syn_steps}, Gradient Average: {grad_avg}\n")

        optimizer_img.step()
        optimizer_lr.step()

        print(f"Distillation Step {i}: loss {grand_loss}")
        wandb.log(
            data={"syn_data": {
            "loss": grand_loss.detach().cpu(),
            }},
            step=i
        )

        for _ in student_params:
            del _

        for _ in student_params_critic:
            del _
    
    ##############################################################
    # Save Final Distilled Data and Synthetic LR
    ##############################################################

    save_dir = f"distilled_data_{args.dataset}_{DT_STRING}"
    os.makedirs(save_dir, exist_ok=True) 
    torch.save(trainset_images.cpu(), os.path.join(save_dir, "distilled_images.pt"))
    torch.save(syn_lr.cpu(), os.path.join(save_dir, "synthetic_lr.pt"))
    print("Distilled dataset and synthetic lr saved.")

    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mtt contrastive learning data distillation process')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--distillation_step', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic data')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic data learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for training the network')
    parser.add_argument("--test_batch_size", type=int, default=1024, help='Testing and classification set batch size')
    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=30, help='max epoch we can start at')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--reg_weight', type=float, default=0.001, help="regularization weight")
    parser.add_argument('--syn_init_method', type=str, default="real", help="how to initialize the synthetic data")
    parser.add_argument("--path", type=str, default=None, help='Path of the initial image. Should be specified if syn_init_method is set to path.')
    parser.add_argument("--test-freq", type=int, default=100000, help='Frequency to fit a linear clf with L-BFGS for testing')
    parser.add_argument("--lr_lr", type=float, default=1e-05, help='lr for lr')
    parser.add_argument("--optimizer_img", type=str, default="SGD", choices=['Adam', 'SGD'], help='synthetic image optimizer')
    parser.add_argument("--verbose", action='store_true', help='Whether to plot or not')

    args = parser.parse_args()

    main(args)