"""
Evaluate distilled dataset (Linear Probe / Supervised Contrastive Loss)
"""
import argparse
import wandb
import torch

from utils.data_util import *
from models.networks.convNet import *
import torch.optim as optim
from utils.random import Random
from models.projection_heads.critic import LinearCritic
from trainer import Trainer

def main(args):

    try:
        distilled_images = torch.load(args.dip)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project="distilled-image-evaluation",
        config=args
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    ori_datasets = get_datasets(args.dataset)

    testset = ori_datasets.testset

    clftrainset = ori_datasets.clftrainset

    distilled_images = distilled_images.detach().cpu()

    trainset = get_custom_dataset(dataset_images=distilled_images, device=device, dataset=args.dataset)

    ##############################################################
    # Encoder and Critics
    ##############################################################

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

    # TODO: Use trained synthetic learning rate for MTT distilled set

    optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6)
    if args.dataset == SupportedDatasets.TINY_IMAGENET.value:
        optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=2 * args.lr, weight_decay=1e-6)
    
    ##############################################################
    # Data Loaders
    ##############################################################

    trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=True,
        )

    clftrainloader = torch.utils.data.DataLoader(
            dataset=clftrainset,
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

    net.to(device)
    critic.to(device)

    ##############################################################
    # Main Loop
    ##############################################################

    trainer = Trainer(
        device=device,
        distributed=False,
        net=net,
        critic=critic,
        trainloader=trainloader,
        clftrainloader=clftrainloader,
        testloader=testloader,
        num_classes=ori_datasets.num_classes,
        optimizer=optimizer,
        reg_weight=args.reg_weight,
    )

    ##############################################################
    # Linear Probe
    ##############################################################

    if not args.supervised_cl:
        test_acc = trainer.test()
        print("linear probe over randomly initialized network test accuracy: ", test_acc)
        wandb.log({"linear probe over randomly initialized network test accuracy: ": test_acc})

        test_accuracies = []

        for epoch in range(0, args.num_epochs):
            print(f"step: {epoch}")
            train_loss = trainer.train()
            print(f"train_loss: {train_loss}")
            wandb.log(
                data={"train": {
                "loss": train_loss,
                }},
                step=epoch
            )
            
            if ((epoch + 1) % args.test_freq == 0):
                test_acc = trainer.test()
                test_accuracies.append(test_acc)
                print(f"test_acc: {test_acc}")
                wandb.log(
                    data={"test": {
                    "acc": test_acc,
                    }},
                    step=epoch
                )
        
        print("best test accuracy: ", max(test_accuracies))
        wandb.log({"best_test_accuracy": max(test_accuracies)})
    
    ##############################################################
    # Supervised Contrastive Loss
    ##############################################################

    else:
        # Get original trainset with labels (Normalize? / Shuffle?)
        ori_datasets_with_trainset = get_datasets(args.dataset, need_train_ori=True)
        ori_trainset = ori_datasets_with_trainset.trainset_ori
        ori_trainloader = torch.utils.data.DataLoader(
            dataset=ori_trainset,
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=True,
        )

        train_loss, test_loss = trainer.test_supcon(ori_trainloader)

        print(f"random encoder train accuracy: {train_loss}, test accuracy: {test_loss}")
        wandb.log({"random encoder train accuracy": train_loss, "random encoder test accuracy": test_loss})

        train_loss_evals = []
        test_loss_evals = []

        for epoch in range(0, args.num_epochs):
            print(f"step: {epoch}")
            train_loss = trainer.train()
            print(f"train_loss: {train_loss}")
            wandb.log(
                data={"train": {
                "loss": train_loss,
                }},
                step=epoch
            )
            
            if ((epoch + 1) % args.test_freq == 0):
                train_loss_eval, test_loss_eval = trainer.test_supcon(ori_trainloader)
                train_loss_evals.append(train_loss_eval)
                test_loss_evals.append(test_loss_eval)
                wandb.log(
                    data={
                        "train_eval": {
                            "loss": train_loss_eval,
                        },
                        "test_eval": {
                            "loss": test_loss_eval,
                        }
                    },
                    step=epoch
                )

        wandb.log({"best_train_loss": min(train_loss_evals)})
        wandb.log({"best_test_loss": min(test_loss_evals)})

        print("best train loss: ", min(train_loss_evals))
        print("best test loss: ", min(test_loss_evals))

    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate distilled dataset')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])    
    parser.add_argument('--dip', type=str, default='distill', help='distilled image path')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate') 
    parser.add_argument("--batch-size", type=int, default=100, help='Training batch size')
    parser.add_argument("--test-batch-size", type=int, default=1024, help='Testing and classification set batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--reg_weight', type=float, default=0.0001, help="regularization weight")
    parser.add_argument('--num_epochs', type=int, default=600, help="number of epochs to train")
    parser.add_argument("--test-freq", type=int, default=20, help='Frequency to fit a linear clf with L-BFGS for testing')
    parser.add_argument("--supervised_cl", action='store_true', help='Whether to evaluate using supervised contrastive learning loss')

    args = parser.parse_args()

    main(args)