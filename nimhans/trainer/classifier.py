import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, CohenKappa, F1Score
from torch.nn.functional import softmax
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler


class ClassifierTrainer(nn.Module):

    def __init__(self, model, experiment_name, weights_path, train_loader, test_loader, loggr, n_classes=5, weight_decay=3e-5, lr=0.001, num_epochs=2, criterion=None, optimizer=None, scheduler=None, kfold=None):
        super(ClassifierTrainer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.experiment_name = experiment_name
        self.kfold = kfold
        self.weights_path = weights_path
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.lr = lr
        self.num_epochs = num_epochs

        self.best_accuracy = 0
        self.best_kappa = 0
        self.best_f1 = 0
        self.loggr = loggr
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

        if self.loggr is not None:
            self.loggr.log({
                    f'F1 train {self.kfold}': f1_score,
                    f'Kappa train {self.kfold}': kappa,
                    f'Bal Acc train {self.kfold}': bal_acc,
                    f'Acc train {self.kfold}': acc,
                    f'Loss train {self.kfold}': epoch_loss.item(),
                    'Epoch': epoch,
                    f'LR {self.kfold}': self.scheduler.optimizer.param_groups[0]["lr"],
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

        if self.loggr is not None:
            self.loggr.log({
                    f'F1 test {self.kfold}': f1_score,
                    f'Kappa test {self.kfold}': kappa,
                    f'Bal Acc test {self.kfold}': bal_acc,
                    f'Acc test {self.kfold}': acc,
                    f'Loss test {self.kfold}': epoch_loss.item(),
                    'Epoch': epoch
                })
        return acc, kappa, f1_score

    def fit(self):
        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            
            print('='*100, end = '\n')
            print(f"Epoch: {epoch}")
            print('='*100, end = '\n')

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

            current_accuracy, current_kappa, current_f1 = self.on_test_epoch_end(test_outputs, epoch)
            if self.best_accuracy < current_accuracy:
                checkpoint = {
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'accuracy': current_accuracy,
                        'kappa': current_kappa,
                        'f1': current_f1,
                    }
                torch.save(
                    checkpoint,
                    os.path.join(self.weights_path, self.experiment_name + "_best.pt"),
                    )
                if self.loggr is not None:
                    self.loggr.save(os.path.join(self.weights_path, self.experiment_name + "_best.pt"))
                self.best_accuracy = current_accuracy
                self.best_kappa = current_kappa
                self.best_f1 = current_f1
                print(f"\nBest weights saved with accuracy: {self.best_accuracy*100:.2f} at epoch: {epoch}")