"""
Evaluate distilled dataset
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
        labels = torch.load(args.label_path)
    
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

    clfset = ori_datasets.clftrainset

    trainset = get_custom_dataset(dataset_images=distilled_images, labels=labels, device=device, dataset=dataset)

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

    optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=1e-3, weight_decay=1e-6)

    trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=True,
        )

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

    trainer = Trainer(
        device=device,
        net=net,
        critic=critic,
        trainloader=trainloader,
        clftrainloader=clftrainloader,
        testloader=testloader,
        num_classes=ori_datasets.num_classes,
        optimizer=optimizer,
    )

    test_acc = trainer.test()
    print("linear probe over randomly initialized network test accuracy: ", test_acc)

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

    wandb.finish(quiet=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate distilled dataset')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])    
    parser.add_argument('--dip', type=str, default='distill', help='distilled image path')
    parser.add_argument('--label_path', type=str, default='distill', help='path for the label')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr', type=float, default=1e-05, help='learning rate')
    parser.add_argument("--batch-size", type=int, default=10, help='Training batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--reg_weight', type=float, default=0.001, help="regularization weight")


    args = parser.parse_args()

    main(args)