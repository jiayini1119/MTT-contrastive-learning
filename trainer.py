import torch
from torch import Tensor, nn
import torch.distributed as dist 
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


from evaluate.lbfgs import encode_train_set, train_clf, test_clf
from models.projection_heads.critic import LinearCritic
from utils.augmentation import KorniaAugmentation


class Trainer():
    def __init__(
        self,
        net: nn.Module,
        critic: LinearCritic,
        trainloader: DataLoader,
        clftrainloader: DataLoader,
        testloader: DataLoader,
        num_classes: int,
        optimizer: Optimizer,
        device: torch.device,
        distributed: bool,
        rank: int = 0,
        world_size: int = 1,
        lr_scheduler = None,
        reg_weight = 0.00001,
    ):
        """
        :param device: Device to run on (GPU)
        :param net: encoder network
        :param critic: projection head
        :param trainset: training data
        :param clftrainloader: dataloader for train data (for linear probe)
        :param optimizer: Optimizer for the encoder network (net)
        :param lr_scheduler: learning rate scheduler
        """
        self.device = device
        self.rank = rank
        self.net = net 
        self.critic = critic
        self.trainloader = trainloader
        self.clftrainloader = clftrainloader
        self.testloader = testloader
        self.num_classes = num_classes
        self.encoder_optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.distributed = distributed
        self.world_size = world_size
        self.reg_weight = reg_weight

        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.best_rare_acc = 0

    #########################################
    #           Loss Functions              #
    #########################################
    
    def supcon_loss(self, z: Tensor, num_positive: int, label: int):
        batch_size = int(len(z) / num_positive)

        if self.distributed:
            all_z = [torch.zeros_like(z) for _ in range(self.world_size)]
            dist.all_gather(all_z, z)
            # Move all tensors to the same device
            aug_z = []
            for i in range(num_positive):
                aug_z.append([])
                for rank in range(self.world_size):
                    if rank == self.rank:
                        aug_z[-1].append(z[i * batch_size: (i+1) * batch_size])
                    else:
                        aug_z[-1].append(all_z[rank][i * batch_size: (i+1) * batch_size])
            z = [torch.cat(aug_z_i, dim=0) for aug_z_i in aug_z]
        else: 
            aug_z = []
            for i in range(num_positive):
                aug_z.append(z[i * batch_size : (i + 1) * batch_size])
            z = aug_z
        
        sim = self.critic(z)
        log_sum_exp_sim = torch.log(torch.sum(torch.exp(sim), dim=1))
        targets = torch.cat([label] * num_positive)
        # print(targets)
        pos_pairs = (targets.unsqueeze(1) == targets.unsqueeze(0)).to(self.device)
        # print(pos_pairs)
        inf_mask = (sim != float('-inf')).to(self.device)
        pos_pairs = torch.logical_and(pos_pairs, inf_mask)

        pos_count = torch.sum(pos_pairs, dim=1)
        pos_sims = torch.nansum(sim * pos_pairs, dim=-1)
        res = torch.mean(-pos_sims / pos_count + log_sum_exp_sim)
        if torch.isnan(res):
            raise ValueError("supcon_loss is nan!")
        return torch.mean(-pos_sims / pos_count + log_sum_exp_sim)

    def un_supcon_loss(self, z: Tensor, num_positive: int):
        batch_size = int(len(z) / num_positive)

        if self.distributed:
            all_z = [torch.zeros_like(z) for _ in range(self.world_size)]
            dist.all_gather(all_z, z)
            # Move all tensors to the same device
            aug_z = []
            for i in range(num_positive):
                aug_z.append([])
                for rank in range(self.world_size):
                    if rank == self.rank:
                        aug_z[-1].append(z[i * batch_size: (i+1) * batch_size])
                    else:
                        aug_z[-1].append(all_z[rank][i * batch_size: (i+1) * batch_size])
            z = [torch.cat(aug_z_i, dim=0) for aug_z_i in aug_z]
        else: 
            aug_z = []
            for i in range(num_positive):
                aug_z.append(z[i * batch_size : (i + 1) * batch_size])
            z = aug_z
        sim = self.critic(z)
        #print(sim)
        log_sum_exp_sim = torch.log(torch.sum(torch.exp(sim), dim=1))
        # Positive Pairs Mask 
        p_targets = torch.cat([torch.tensor(range(int(len(sim) / num_positive)))] * num_positive)
        #len(p_targets)
        pos_pairs = (p_targets.unsqueeze(1) == p_targets.unsqueeze(0)).to(self.device)
        #print(pos_pairs)
        inf_mask = (sim != float('-inf')).to(self.device)
        pos_pairs = torch.logical_and(pos_pairs, inf_mask)
        pos_count = torch.sum(pos_pairs, dim=1)
        pos_sims = torch.nansum(sim * pos_pairs, dim=-1)
        return torch.mean(-pos_sims / pos_count + log_sum_exp_sim)
    
    #########################################
    #           Train & Test Modules        #
    #########################################
    def train(self):
        self.net.train()
        self.critic.train()

        # Training Loop (over batches in epoch)
        train_loss = 0
        t = tqdm(enumerate(self.trainloader), desc='Loss: **** ', total=len(self.trainloader), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, inputs in t:
            num_positive = len(inputs)
            x = torch.cat(inputs, dim=0).to(self.device)
            self.encoder_optimizer.zero_grad()
            z = self.net(x)
            loss = self.un_supcon_loss(z, num_positive)
            loss.backward()
            self.encoder_optimizer.step()
            train_loss += loss.item()
            t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            print("lr:", self.scale_lr * self.lr_scheduler.get_last_lr()[0])
            
        return train_loss / len(self.trainloader)

    def test(self):
        X, y = encode_train_set(self.clftrainloader, self.device, self.net)
        representation_dim = self.net.module.representation_dim if self.distributed else self.net.representation_dim
        clf = train_clf(X, y, representation_dim, self.num_classes, self.device, reg_weight=self.reg_weight, iter=100)
        acc = test_clf(self.testloader, self.device, self.net, clf)

        if acc > self.best_acc:
            self.best_acc = acc
            
        return acc

    def test_supcon(self, ori_trainloader: DataLoader):
        self.net.eval()  
        self.critic.eval()

        train_loss = 0
        test_loss = 0
        with torch.no_grad():
            t = tqdm(enumerate(ori_trainloader), desc='Loss: **** ', total=len(ori_trainloader), bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, labels) in t:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                embeddings = self.net(inputs)
                loss = torch.tensor(0)
                try: 
                    loss = self.supcon_loss(embeddings, 1, labels)  
                except ValueError:
                    loss = torch.tensor(0)
                    print("Got nan for supcon loss")
                train_loss += loss.item()
                t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        train_loss /= len(ori_trainloader)
        
        with torch.no_grad():
            t = tqdm(enumerate(self.testloader), desc='Loss: **** ', total=len(self.testloader), bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, labels) in t:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                embeddings = self.net(inputs)
                loss = torch.tensor(0)
                try:
                    loss = self.supcon_loss(embeddings, 1, labels)  
                except ValueError:
                    loss = torch.tensor(0)
                    print("Got nan for supcon loss")
                loss = self.supcon_loss(embeddings, 1, labels)  
                test_loss += loss.item()
                t.set_description('Loss: %.3f ' % (test_loss / (batch_idx + 1)))

        test_loss /= len(self.testloader)

        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        return train_loss, test_loss

    def save_checkpoint(self, prefix):
        if self.world_size > 1:
            torch.save(self.net.module, f"{prefix}-net.pt")
        else:
            torch.save(self.net, f"{prefix}-net.pt")
        torch.save(self.critic, f"{prefix}-critic.pt")


class SynTrainer(Trainer):
    def __init__(
        self,
        trainset_images: Tensor,
        batch_size: int,
        n_augmentations: int,
        transform: KorniaAugmentation,
        student_params,
        student_params_critic,
        syn_lr: int,
        reparam_net,
        reparam_critic,
        bn_layer,
        *args,
        **kwargs
    ):
        super().__init__(*args, trainloader=None, optimizer=None, **kwargs)
        self.trainset_images = trainset_images.requires_grad_(True)
        self.batch_size = batch_size
        self.n_augmentations = n_augmentations
        self.transform = transform
        self.student_params = student_params
        self.student_params_critic = student_params_critic
        self.syn_lr = syn_lr
        self.reparam_net = reparam_net
        self.reparam_critic = reparam_critic
        self.bn_layer=bn_layer



    def un_supcon_loss(self, z: Tensor, num_positive: int):
        batch_size = int(len(z) / num_positive)

        if self.distributed:
            all_z = [torch.zeros_like(z) for _ in range(self.world_size)]
            dist.all_gather(all_z, z)
            # Move all tensors to the same device
            aug_z = []
            for i in range(num_positive):
                aug_z.append([])
                for rank in range(self.world_size):
                    if rank == self.rank:
                        aug_z[-1].append(z[i * batch_size: (i+1) * batch_size])
                    else:
                        aug_z[-1].append(all_z[rank][i * batch_size: (i+1) * batch_size])
            z = [torch.cat(aug_z_i, dim=0) for aug_z_i in aug_z]
        else: 
            aug_z = []
            for i in range(num_positive):
                aug_z.append(z[i * batch_size : (i + 1) * batch_size])
            z = aug_z

        sim = self.reparam_critic(z, flat_param=self.student_params_critic[-1])
        # print(sim)
        log_sum_exp_sim = torch.log(torch.sum(torch.exp(sim), dim=1))
        # Positive Pairs Mask 
        p_targets = torch.cat([torch.tensor(range(int(len(sim) / num_positive)))] * num_positive)
        #len(p_targets)
        pos_pairs = (p_targets.unsqueeze(1) == p_targets.unsqueeze(0)).to(self.device)
        #print(pos_pairs)
        inf_mask = (sim != float('-inf')).to(self.device)
        pos_pairs = torch.logical_and(pos_pairs, inf_mask)
        pos_count = torch.sum(pos_pairs, dim=1)
        pos_sims = torch.nansum(sim * pos_pairs, dim=-1)
        return torch.mean(-pos_sims / pos_count + log_sum_exp_sim)

    # def train(self):
    #     self.reparam_net.train()
    #     self.reparam_critic.train()

    #     train_loss = 0
    #     total_samples = len(self.trainset_images)
    #     t = tqdm(range(total_samples), desc='Loss: **** ', bar_format='{desc}{bar}{r_bar}')

    #     img_shape = self.trainset_images[0].shape

    #     inputs = [torch.empty((total_samples,) + img_shape, device=self.device) for _ in range(self.n_augmentations)]

    #     for img_idx, image in enumerate(self.trainset_images):
    #         for aug_idx in range(self.n_augmentations):
    #             augmented_image = self.transform(image)
    #             inputs[aug_idx][img_idx] = augmented_image

    #     num_positive = len(inputs)
    #     x = torch.cat(inputs, dim=0).to(self.device)
    #     forward_params = self.student_params[-1]
    #     x = self.bn_layer(x)
    #     z = self.reparam_net(x, flat_param = forward_params)
    #     loss = self.un_supcon_loss(z, num_positive)

    #     grad_net, grad_critic = torch.autograd.grad(loss, [self.student_params[-1], self.student_params_critic[-1]],create_graph=True)

    #     self.student_params.append(self.student_params[-1] - self.syn_lr * grad_net)
    #     self.student_params_critic.append(self.student_params_critic[-1] - self.syn_lr * grad_critic)

    #     train_loss += loss.item()
    #     t.set_description('Loss: %.3f ' % train_loss)


    #     if self.lr_scheduler is not None:
    #         raise NotImplementedError
        
    #     return train_loss

    def train(self, these_indices):
        self.reparam_net.train()
        self.reparam_critic.train()

        train_loss = 0
        batch_images = self.trainset_images[these_indices]
        batch_size_ind = len(batch_images)

        t = tqdm(these_indices, desc='Loss: **** ', bar_format='{desc}{bar}{r_bar}')

        img_shape = self.trainset_images[0].shape

        inputs = [torch.empty((batch_size_ind,) + img_shape, device=self.device) for _ in range(self.n_augmentations)]

        for img_idx, image in enumerate(batch_images):
            for aug_idx in range(self.n_augmentations):
                augmented_image = self.transform(image)
                inputs[aug_idx][img_idx] = augmented_image

        num_positive = len(inputs)
        x = torch.cat(inputs, dim=0).to(self.device)

        forward_params = self.student_params[-1]
        x = self.bn_layer(x)
        z = self.reparam_net(x, flat_param = forward_params)
        loss = self.un_supcon_loss(z, num_positive)

        grad_net, grad_critic = torch.autograd.grad(loss, [self.student_params[-1], self.student_params_critic[-1]],create_graph=True)

        self.student_params.append(self.student_params[-1] - self.syn_lr * grad_net)
        self.student_params_critic.append(self.student_params_critic[-1] - self.syn_lr * grad_critic)

        train_loss += loss.item()
        t.set_description('Loss: %.3f ' % train_loss)


        if self.lr_scheduler is not None:
            raise NotImplementedError
        
        return train_loss