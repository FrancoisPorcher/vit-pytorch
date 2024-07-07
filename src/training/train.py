import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 learning_rate=0.001, num_epochs=10, checkpoint_dir='./checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.device = get_device()

        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self):
        num_parameters = count_trainable_parameters(self.model)
        print(f"Number of trainable parameters: {num_parameters}")
        best_val_acc = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'loss': running_loss / (i + 1), 'acc': 100. * correct / total})

            val_loss, val_acc = self.evaluate(self.val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

            # Save checkpoint if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def test(self):
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    def save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_acc_{val_acc:.2f}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.2f}%")
        return checkpoint['epoch']