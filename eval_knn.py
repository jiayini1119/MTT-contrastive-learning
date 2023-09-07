import argparse
import wandb
import torch
import numpy as np
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from utils.supported_dataset import *
from utils.custom_dataset import *
from models.networks.convNet import *
from trainer import Trainer
from utils.random import Random
from models.projection_heads.critic import LinearCritic

def extract_embeddings(loader, model, device):
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            embeddings.append(output.cpu())
            labels.append(target.cpu())

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    return embeddings, labels

# Evaluation KNN on the normalized embeddings
def evaluate_knn(clftrain_embeddings, clftrain_labels, test_embeddings, test_labels):
    clftrain_embeddings = clftrain_embeddings / torch.norm(clftrain_embeddings, dim=1, keepdim=True)
    test_embeddings = test_embeddings / torch.norm(test_embeddings, dim=1, keepdim=True)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(clftrain_embeddings, clftrain_labels)
    accuracy = knn.score(test_embeddings, test_labels)
    return accuracy

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    wandb.init(project="distilled-image-knn-evaluation", config=args)

    ori_datasets = get_datasets(args.dataset)

    testset = ori_datasets.testset

    clfset = ori_datasets.clftrainset

    distilled_images = torch.load(args.dip).detach().cpu()

    trainset = get_custom_dataset(dataset_images=distilled_images, device=device, dataset=args.dataset)

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    net = ConvNet(net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

    optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6)

    net.to(device)
    critic.to(device)

    trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=True,
        )

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

    clftrain_embeddings, clftrain_labels = extract_embeddings(clftrainloader, net, device)
    test_embeddings, test_labels = extract_embeddings(testloader, net, device)
    initial_knn_accuracy = evaluate_knn(clftrain_embeddings, clftrain_labels, test_embeddings, test_labels)
    print("Initial KNN Accuracy: ", initial_knn_accuracy)

    wandb.log({"Initial KNN Accuracy": initial_knn_accuracy})

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

    clftrain_embeddings, train_labels = extract_embeddings(clftrainloader, net, device)
    test_embeddings, test_labels = extract_embeddings(testloader, net, device)
    updated_knn_accuracy = evaluate_knn(clftrain_embeddings, train_labels, test_embeddings, test_labels)
    print("KNN Accuracy after SimCLR training: ", updated_knn_accuracy)
    wandb.log({"KNN Accuracy after SimCLR training": updated_knn_accuracy})


    wandb.finish(quiet=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate distilled dataset using KNN on embeddings')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])    
    parser.add_argument('--dip', type=str, default='distill', help='distilled image path')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument("--test-batch-size", type=int, default=1024, help='Testing and classification set batch size')
    parser.add_argument('--reg_weight', type=float, default=0.0001, help="regularization weight")
    parser.add_argument('--num_epochs', type=int, default=600, help="number of epochs to train")

    args = parser.parse_args()
    main(args)
