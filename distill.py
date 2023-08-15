"""
Data Distillation by Matching Training Trajectories for Contrastive Learning
"""
import argparse
import glob
import wandb
from datetime import datetime


from utils.data_util import *
import numpy as np
from models.networks.convNet import *
from utils.augmentation import KorniaAugmentation
import torch.optim as optim
from utils.random import Random


from models.projection_heads.critic import LinearCritic
from trainer import SynTrainer

def main(args):
    """
    I have ckpt folder with structure:
    trajectory_0:
      - trajectory_0_epoch_0
      - trajectory_0_epoch_1
      - trajectory_0_epoch_2

    
    trajectory_1:
      - trajectory_1_epoch_0
      - trajectory_1_epoch_1
      - trajectory_1_epoch_2
    
    trajectory_2:
      - trajectory_2_epoch_0
      - trajectory_2_epoch_1
      - trajectory_2_epoch_2

    
    Those are saved duirng training the expert trajectory

    mll.py:
    Initialize Syn dataset D_syn
       Only train the trainset
    Initialize learning rate(not trainable, fixed now)

    optimizer for D_syn

    for i in range(distillation_step + 1):
    
        Sample expert trajectory 
        Choose random start epoch < max_start_epoch => (initial_trajectory)
        Initialize student network with expert params from initial_trajectory
        Get trainset, clfset, and testset from D_syn

        for j in range(N):
            sample a minibatch from the  D_syn 
            simclr training loop
        
        get target params
        loss between student params and expert params
        update D_sync with respect to the loss
    
    """

    wandb.init(
        project="data-distillation-by-mtt-contrastive-learning",
        config=args
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    DT_STRING = "".join(str(datetime.now()).split())

    # Initialize Syn dataset D_syn
    ori_datasets = get_datasets(args.dataset)

    testset = ori_datasets.testset

    clfset = ori_datasets.clftrainset

    trainset_images = torch.randn(ori_datasets.num_classes * args.ipc, ori_datasets.channel, ori_datasets.img_size, ori_datasets.img_size)
    labels = torch.cat([torch.tensor([i] * args.ipc) for i in range(ori_datasets.num_classes)])

    trainset_images = trainset_images.to(device).detach().requires_grad_(True)

    print(trainset_images)

    optimizer_img = torch.optim.SGD([nn.Parameter(trainset_images)], lr=args.lr_img, momentum=0.5)

    optimizer_img.zero_grad()

    for _ in range(args.distillation_step + 1):

        # sample an expert trajectory
        subdirectories = [d for d in os.listdir(f'ckpt_{args.dataset}') if os.path.isdir(os.path.join(f'ckpt_{args.dataset}', d))]

        selected_directory = np.random.choice(subdirectories)

        trajectories = os.path.join(f'ckpt_{args.dataset}', selected_directory)
        matching_files = glob.glob(os.path.join(trajectories, f"{selected_directory}_epoch_*.pt"))

        initial_trajectory = np.random.choice([f for f in matching_files if int(f.split('_epoch_')[-1].split('.pt')[0]) < args.max_start_epoch])

        # Initialize student network with expert params from the expert trajectory
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, channel=ori_datasets.channel)

        expert_state_dict = torch.load(initial_trajectory)
        net.load_state_dict(expert_state_dict)

        num_params = sum([np.prod(p.size()) for p in (net.parameters())])
        starting_params = [net.state_dict()[key] for key in net.state_dict()]
        starting_params = torch.cat([p.data.to(device).reshape(-1) for p in starting_params], 0)

        print("Assigned the initial weight to the student network")

        critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

        optimizer_simclr = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6)
        if args.dataset == SupportedDatasets.TINY_IMAGENET.value:
            optimizer_simclr = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=2 * args.lr, weight_decay=1e-6)

        clftrainloader = torch.utils.data.DataLoader(
            dataset=clfset,
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=6, 
            pin_memory=True
        )

        testloader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=6,
            pin_memory=True,
        )

        # synclr training
        print("***Start simclr training***")
        net = net.to(device)
        critic = critic.to(device)

        kornia_augmentations = KorniaAugmentation(dataset=args.dataset).to(device)

        trainer = SynTrainer(
            trainset_images=trainset_images,
            labels=labels,
            batch_size = args.batch_size,
            n_augmentations=2,
            transform=kornia_augmentations,
            device=device,
            distributed=False,
            net=net,
            critic=critic,
            clftrainloader=clftrainloader,
            testloader=testloader,
            num_classes=ori_datasets.num_classes,
            optimizer=optimizer_simclr,
            reg_weight=args.reg_weight,
        )

        for epoch in range(0, args.syn_steps):
            print(f"step: {epoch}")
            train_loss = trainer.train()
            print(f"train_loss: {train_loss}")
            wandb.log(
                data={"train": {
                "loss": train_loss,
                }},
                step=epoch
            )
            test_acc = trainer.test()
            print(f"test_acc: {test_acc}")
            wandb.log(
                data={"test": {
                "acc": test_acc,
                }},
                step=epoch
            )
        

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

        student_params = [p.reshape(-1) for p in net.parameters()]
        student_params = torch.cat(student_params).to(device)

        print(student_params.requires_grad)

        target_params = [final_state_dict[key] for key in final_state_dict]
        target_params = torch.cat([p.data.to(device).reshape(-1) for p in target_params], 0)

        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        param_loss += torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist
        grand_loss = param_loss

        optimizer_img.zero_grad()
        grand_loss.backward()

        optimizer_img.step()

        print("dataset grad final")
        print(trainset_images)
        # print(trainset_images.requires_grad)
        # print(trainset_images.grad)

        for _ in student_params:
            del _
    
    # save the final dataset
    save_dir = f"distilled_data_{args.dataset}_{DT_STRING}"
    os.makedirs(save_dir, exist_ok=True) 
    torch.save(trainset_images.cpu(), os.path.join(save_dir, "distilled_images.pt"))
    torch.save(labels.cpu(), os.path.join(save_dir, "labels.pt"))
    print("Distilled dataset saved.")

    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mtt contrastive learning data distillation process')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--epoch_eval_train', type=int, default=5, help='epochs to train a model with synthetic data')
    parser.add_argument('--distillation_step', type=int, default=50, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic data')
    parser.add_argument('--lr', type=float, default=1e-05, help='simclr learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic data learning rate')
    parser.add_argument("--batch-size", type=int, default=10, help='Training batch size for inner loop')

    parser.add_argument('--expert_epochs', type=int, default=10, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=10, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--reg_weight', type=float, default=0.001, help="regularization weight")

    args = parser.parse_args()

    main(args)