import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, CohenKappa, F1Score
from torch.nn.functional import softmax
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler


class classifierModel(nn.Module):

    def __init__(self, model, experiment_id, save_path, train_loader, test_loader, wandb, batch_size=256, n_classes=5, weight_decay=3e-5, lr=0.001, num_epochs=60, criterion=None, optimizer=None, scheduler=None):
        super(classifierModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.exp_name = experiment_id
        self.save_path = save_path
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.lr = lr
        self.num_epochs = num_epochs

        self.best_accuracy = 0
        self.loggr = wandb
        self.train_loader = train_loader
        self.test_loader = test_loader
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.lr,
                betas=(0.9, 0.99),
                weight_decay=self.weight_decay,
            )
        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode="min",
                                           patience=5,
                                           factor=0.2
                                           )
        
    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float().to(self.device), y.long().to(self.device)
        outs = self.model(X)
        loss = self.criterion(outs, y)
        return loss, outs, y

    def testing_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float().to(self.device), y.long().to(self.device)
        outs = self.model(X)
        loss = self.criterion(outs, y)
        return loss, outs, y  

    def on_train_epoch_end(self, outputs, epoch):
        epoch_preds = torch.vstack([x for x in outputs["preds"]])
        epoch_targets = torch.hstack([x for x in outputs["targets"]])
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs['loss']]).mean()

        self.scheduler.step(epoch_loss)
        
        class_preds = epoch_preds.argmax(dim=1)
        acc = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)(epoch_preds, epoch_targets)
        f1_score = F1Score(task="multiclass", num_classes=self.n_classes, average="macro").to(self.device)(epoch_preds, epoch_targets)
        kappa = CohenKappa(task="multiclass", num_classes=self.n_classes).to(self.device)(epoch_preds, epoch_targets)
        bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(), class_preds.cpu().numpy())

        self.loggr.log({
                'F1 train': f1_score,
                'Kappa train': kappa,
                'Bal Acc train': bal_acc,
                'Acc train': acc,
                'Loss train': epoch_loss.item(),
                'Epoch': epoch,
                'LR': self.scheduler.optimizer.param_groups[0]["lr"],
            })

    def on_test_epoch_end(self, outputs, epoch):
        epoch_preds = torch.vstack([x for x in outputs["preds"]])
        epoch_targets = torch.hstack([x for x in outputs["targets"]])
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs['loss']]).mean()
        
        class_preds = epoch_preds.argmax(dim=1)
        acc = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)(epoch_preds, epoch_targets)
        f1_score = F1Score(task="multiclass", num_classes=self.n_classes, average="macro").to(self.device)(epoch_preds, epoch_targets)
        kappa = CohenKappa(task="multiclass", num_classes=self.n_classes).to(self.device)(epoch_preds, epoch_targets)
        bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(), class_preds.cpu().numpy())

        self.loggr.log({
                'F1 test': f1_score,
                'Kappa test': kappa,
                'Bal Acc test': bal_acc,
                'Acc test': acc,
                'Loss test': epoch_loss.item(),
                'Epoch': epoch
            })
        return acc

    def fit(self):
        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            
            print('='*50, end = '\n')
            print(f"Epoch: {epoch}")
            print('='*50, end = '\n')

            # Training Loop
            train_outputs = {"loss": [], "preds": [], "targets": []}
            self.model.train()

            for batch_idx, batch in tqdm(enumerate(self.train_loader), desc="Train", total=len(self.train_loader)):
                with torch.cuda.amp.autocast():
                    loss, outs, y = self.training_step(batch, batch_idx)
                    
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                train_outputs['loss'].append(loss.detach().item())
                train_outputs['preds'].append(softmax(outs.detach(), dim=1))
                train_outputs['targets'].append(y.detach())

            self.on_train_epoch_end(train_outputs, epoch)

            # Testing Loop
            test_outputs = {"loss": [], "preds": [], "targets": []}
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(self.test_loader), desc="Test", total=len(self.test_loader)):
                    loss, outs, y = self.testing_step(batch, batch_idx)
                    
                    test_outputs['loss'].append(loss.item())
                    test_outputs['preds'].append(softmax(outs, dim=1))
                    test_outputs['targets'].append(y)

            acc = self.on_test_epoch_end(test_outputs, epoch)
            if self.best_accuracy < acc:
                checkpoint = {
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'accuracy': acc,
                    }
                torch.save(
                    checkpoint,
                    os.path.join(self.save_path, self.exp_name + "_best.pt"),
                    )
                self.loggr.save(os.path.join(self.save_path, self.exp_name + "_best.pt"))
                self.best_accuracy = acc
                print(f"Best weights saved with accuracy: {self.best_accuracy*100:.2f} at epoch: {epoch}")