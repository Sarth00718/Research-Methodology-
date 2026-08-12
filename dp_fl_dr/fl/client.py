import torch
import torch.optim as optim
import copy

class FLClient:
    def __init__(self, client_id, train_loader, val_loader, criterion, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device

    def train_local(self, global_model, local_epochs, lr):
        """
        Trains the local model for a specified number of epochs.
        Returns the new state_dict and the number of samples trained on.
        """
        model = copy.deepcopy(global_model)
        model.to(self.device)
        model.train()
        
        # Setup optimizer. A local learning rate scheduler could also be used if needed.
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        num_samples = len(self.train_loader.dataset)
        
        for epoch in range(local_epochs):
            running_loss = 0.0
            
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            # Optional: could log local loss
            # epoch_loss = running_loss / num_samples
            # print(f"Client {self.client_id} - Epoch {epoch+1}/{local_epochs} Loss: {epoch_loss:.4f}")
            
        return model.state_dict(), num_samples

    def evaluate_local(self, global_model):
        """
        Evaluates the current global model on the client's local validation data.
        (For monitoring purposes only).
        """
        global_model.eval()
        global_model.to(self.device)
        
        running_corrects = 0
        total_samples = len(self.val_loader.dataset)
        
        if total_samples == 0:
            return 0.0
            
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = global_model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                
        return running_corrects.double().item() / total_samples
