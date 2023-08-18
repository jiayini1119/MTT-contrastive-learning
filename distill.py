"""
Data Distillation by Matching Training Trajectories for Contrastive Learning
"""
import argparse
import glob
import wandb
from datetime import datetime
from utils.reparam_module import *


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

    # Initialize Synthetic Dataset
    ori_datasets = get_datasets(args.dataset)

    testset = ori_datasets.testset

    clfset = ori_datasets.clftrainset

    trainset_images = get_init_syn_data(method=args.syn_init_method, dataset=args.dataset, ipc=args.ipc, path=args.path)

    torch.save(trainset_images.cpu(), f"real_initial_images.pt")

    labels = torch.cat([torch.tensor([i] * args.ipc) for i in range(ori_datasets.num_classes)])

    torch.save(labels.cpu(), "label.pt")

    trainset_images = nn.Parameter(trainset_images.to(device).requires_grad_(True)).to(device)

    optimizer_img = torch.optim.SGD([trainset_images], lr=args.lr_img, momentum=0.5)

    optimizer_img.zero_grad()

    for i in range(args.distillation_step + 1):

        # sample an expert trajectory
        subdirectories = [d for d in os.listdir(f'ckpt_{args.dataset}') if os.path.isdir(os.path.join(f'ckpt_{args.dataset}', d))]

        selected_directory = np.random.choice(subdirectories)

        trajectories = os.path.join(f'ckpt_{args.dataset}', selected_directory)
        matching_files = glob.glob(os.path.join(trajectories, f"{selected_directory}_epoch_*.pt"))

        initial_trajectory = np.random.choice([f for f in matching_files if int(f.split('_epoch_')[-1].split('.pt')[0]) < args.max_start_epoch])
        initial_trajectory_dict = torch.load(initial_trajectory)

        student_params_start = [initial_trajectory_dict[key] for key in initial_trajectory_dict]

        student_params = [torch.cat([p.data.to(device).reshape(-1) for p in student_params_start], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(device).reshape(-1) for p in student_params_start], 0)

        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        ori_net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, channel=ori_datasets.channel).to(device)

        ori_critic = LinearCritic(ori_net.representation_dim, temperature=args.temperature).to(device)

        num_params = sum([np.prod(p.size()) for p in (ori_net.parameters())])

        net = ReparamModule(ori_net)
        critic_params_list = list(ori_critic.parameters())
        critic = ReparamModule(ori_critic)

        student_params_critic = [torch.cat([p.data.to(device).reshape(-1) for p in critic_params_list], 0).requires_grad_(True)]

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

        # synclr training
        print("***Start simclr training***")

        kornia_augmentations = KorniaAugmentation(dataset=args.dataset).to(device)

        trainer = SynTrainer(
            trainset_images=trainset_images,
            # batch_size = args.batch_size,
            n_augmentations=2,
            transform=kornia_augmentations,
            student_params=student_params,
            student_params_critic=student_params_critic,
            syn_lr = args.lr,
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
        )

        test_acc = trainer.test()
        print("linear probe over randomly initialized network test accuracy: ", test_acc)

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


            if (epoch + 1) % args.test_freq == 0:
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

        optimizer_img.zero_grad()

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

        target_params = [final_state_dict[key] for key in final_state_dict]
        target_params = torch.cat([p.data.to(device).reshape(-1) for p in target_params], 0)

        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        print("param_loss", param_loss)
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        print("param_dist: ", param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        print(f"Distillation Step {i}: loss {grand_loss}")

        optimizer_img.zero_grad()

        grand_loss.backward()

        optimizer_img.step()

        for _ in student_params:
            del _
    
    # save the final dataset
    save_dir = f"distilled_data_{args.dataset}_{DT_STRING}"
    os.makedirs(save_dir, exist_ok=True) 
    torch.save(trainset_images.cpu(), os.path.join(save_dir, "distilled_images.pt"))
    print("Distilled dataset saved.")

    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mtt contrastive learning data distillation process')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=5, help='image(s) per class')
    parser.add_argument('--distillation_step', type=int, default=10, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic data')
    parser.add_argument('--lr', type=float, default=1e-03, help='simclr learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic data learning rate')
    # parser.add_argument("--batch_size", type=int, default=50, help='Training batch size for inner loop')
    parser.add_argument("--test_batch_size", type=int, default=1024, help='Testing and classification set batch size')


    parser.add_argument('--expert_epochs', type=int, default=50, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=500, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=30, help='max epoch we can start at')
    parser.add_argument('--seed', type=int, default=3407, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--reg_weight', type=float, default=0.001, help="regularization weight")
    parser.add_argument('--syn_init_method', type=str, default="real", help="how to initialize the synthetic data")
    parser.add_argument("--path", type=str, default=None, help='Path of the initial image. Should be specified if syn_init_method is set to path.')
    parser.add_argument("--test-freq", type=int, default=100, help='Frequency to fit a linear clf with L-BFGS for testing')

    args = parser.parse_args()

    main(args)