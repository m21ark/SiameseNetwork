import os
import torch
from src.log_parse import my_plot

class Trainer:
    def __init__(self, multi_gpu_model, train_dataloader, test_dataloader, useTripletMode, use_2phase_training, device):
        # Model
        self.model = multi_gpu_model.module if hasattr(multi_gpu_model, 'module') else multi_gpu_model
        self.device = device
        
        # Training mode
        self.use_2phase_training = use_2phase_training
        self.useTripletMode = useTripletMode
        
        # Dataloaders
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.optimizer =  torch.optim.Adam([
        #         {'params': self.model.features.parameters(), 'lr': 1e-4},
        #         {'params': self.model.rep_gen.parameters(), 'lr': 1e-4},
        #         {'params': self.model.classifier.parameters(), 'lr': 1e-4}
        #     ], weight_decay=1e-5) # weight_decay as L2 regularization
        
        self.earlystopper = EarlyStopping(patience=30, min_delta=0.01)
    
    def _test(self):
        self.model.eval()  
        return self._train_test_triple(False, self.test_dataloader) if self.useTripletMode else self._train_test_pair(False, self.test_dataloader)
           
    def train(self, num_epochs=100):
            self.model.train()  
           
            log_file = open("saved_models/log.txt", "w")
            
            if self.use_2phase_training:
                self.model.set_train_mode_representation()
                print("Training representation generator...")
            else:
                print("Joint training mode...")
            
            for epoch in range(num_epochs):
                
                log_contents = [epoch]
                
                print(f"\nEpoch [{epoch}/{num_epochs}]")
                print("\tTraining...")
                
                # Run training epoch
                if self.useTripletMode:
                    (acc, prec, rec, f1), avg_loss, rep_loss, classif_loss = self._train_test_triple(True, self.train_dataloader)
                else:
                    (acc, prec, rec, f1), avg_loss, rep_loss, classif_loss = self._train_test_pair(True, self.train_dataloader)
                    
                print(f"Train Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}")
                log_contents.extend([acc, prec, rec, f1, avg_loss, rep_loss, classif_loss])

                # Run test epoch
                print("\tTesting...")
                (acc, prec, rec, f1), avg_loss, rep_loss, classif_loss = self._test()
                print(f"Test Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}")
                log_contents.extend([acc, prec, rec, f1, avg_loss, rep_loss, classif_loss])
                

                save_model(self.model, f"siamese_{epoch}_{acc:.2f}.pth")

                # Switch to classifier training
                if self.use_2phase_training and self.model.isClassifierFreezed and epoch >= 9:
                    print("Rep loss is low, switching to classifier training...")
                    self.model.set_train_mode_classifier()
                    self.earlystopper.reset()
                    
                # Log to file and plot
                log_file.write(f"{[str(round(x, 4)).zfill(1) for x in log_contents]}\n")
                log_file.flush()  # Force write to disk
                my_plot("saved_models/log.txt")

                # Check early stopping
                if self.earlystopper(avg_loss):
                    break
                
            print("Training complete. Saving final model...")
            log_file.close()      
                   
    def _train_test_triple(self, isTrain, dataloader):
        
        epoch_loss = 0
        correct = 0
        total_samples = 0
        
        rep_loss_func = self.model.triplet_rep_loss
        classif_loss_func = torch.nn.BCELoss() # torch.nn.BCEWithLogitsLoss()
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        rep_loss_val = 0
        classif_loss_val = 0
        
        for batch_idx, (anchor, pos, neg) in enumerate(dataloader):

            anchor, pos, neg = anchor.to(self.device), pos.to(self.device), neg.to(self.device)
            
            total_samples += anchor.size(0) # Batch size
            
            with torch.set_grad_enabled(isTrain):
                
                # Forward pass to convert images to embeddings
                anchor, pos, neg, pred_pos, pred_neg = self.model.forward_triplet(anchor, pos, neg)
                
                # Threshold predictions
                threshold = 0.5
                pred_labels_pos = (pred_pos > threshold).int()  # should be 1
                pred_labels_neg = (pred_neg > threshold).int()  # should be 0

                # True labels
                true_labels_pos = torch.ones_like(pred_labels_pos)  # [1, 1, ..., 1]
                true_labels_neg = torch.zeros_like(pred_labels_neg)  # [0, 0, ..., 0]

                # Concatenate all predictions and labels
                pred_labels = torch.cat([pred_labels_pos, pred_labels_neg], dim=0)
                true_labels = torch.cat([true_labels_pos, true_labels_neg], dim=0)

                # Compute metrics
                tp += ((pred_labels == 1) & (true_labels == 1)).sum().item()
                tn += ((pred_labels == 0) & (true_labels == 0)).sum().item()
                fp += ((pred_labels == 1) & (true_labels == 0)).sum().item()
                fn += ((pred_labels == 0) & (true_labels == 1)).sum().item()
                                
                # Compute loss
                rep_loss = rep_loss_func(anchor, pos, neg)
                rep_loss_val += rep_loss.item()
                
                aux = classif_loss_func(pred_pos, torch.ones_like(pred_pos))
                classif_loss_val += aux.item()
                classif_loss = aux
                
                aux = classif_loss_func(pred_neg, torch.zeros_like(pred_neg))
                classif_loss_val += aux.item()
                classif_loss += aux
                
                # Compute total loss
                if self.use_2phase_training:
                    batch_loss = rep_loss if self.model.isClassifierFreezed else classif_loss
                else: # Joint training mode
                    batch_loss = 0.3 * rep_loss + classif_loss
                    
                epoch_loss += batch_loss.item()
                
                # Backward pass   
                if isTrain: 
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                if batch_idx % 20 == 0:
                    print(f"\tBatch [{str(batch_idx).zfill(2)}/{len(dataloader)}], Loss: {batch_loss.item():.4f}")
        
        # Compute confusion matrix metrics
        acc = ((tp + tn) / (tp + tn + fp + fn)) * 100
        prec = (tp / (tp + fp) if (tp + fp) > 0 else 0) * 100
        rec = (tp / (tp + fn) if (tp + fn) > 0 else 0) * 100
        f1 = (2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0)

        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

        return (acc, prec, rec, f1), epoch_loss / len(dataloader), rep_loss_val / len(dataloader), classif_loss_val / len(dataloader)
    
    def _train_test_pair(self, isTrain, dataloader):
        
        epoch_loss = 0
        
        rep_loss_func = self.model.contrastive_rep_loss
        classif_loss_func = torch.nn.BCELoss() # torch.nn.BCEWithLogitsLoss()
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        rep_loss_val = 0
        classification_loss_val = 0
        
        for batch_idx, (img1, img2, label) in enumerate(dataloader):

            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)

            with torch.set_grad_enabled(isTrain):
                
                # Forward pass to convert images to embeddings
                rep1, rep2, pred = self.model.forward_pair(img1, img2)
                
                # Compute tp, tn, fp, fn
                threshold = 0.5
                pred_bin = (pred > threshold).int() # Convert to binary predictions (0 or 1)
                
                tp += (pred_bin[label == 1] == 1).sum().item()
                tn += (pred_bin[label == 0] == 0).sum().item()
                fp += (pred_bin[label == 0] == 1).sum().item()
                fn += (pred_bin[label == 1] == 0).sum().item()
                
                # Compute loss
                rep_loss = rep_loss_func(rep1, rep2, label)
                rep_loss_val += rep_loss.item()
                
                classification_loss = classif_loss_func(pred, label.float().unsqueeze(1))
                classification_loss_val += classification_loss.item()

                # Compute total loss
                if self.use_2phase_training:
                    batch_loss = rep_loss if self.model.isClassifierFreezed else classification_loss
                else: # Joint training mode
                    batch_loss = 0.3 * rep_loss + classification_loss
                    
                epoch_loss += batch_loss.item()
                
                # Backward pass   
                if isTrain: 
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    # self.scheduler.step()
                    
                if batch_idx % 20 == 0:
                    print(f"\tBatch [{str(batch_idx).zfill(2)}/{len(dataloader)}], Loss: {batch_loss.item():.4f}")
                    
        # Compute confusion matrix metrics
        acc = ((tp + tn) / (tp + tn + fp + fn)) * 100
        prec = (tp / (tp + fp) if (tp + fp) > 0 else 0) * 100
        rec = (tp / (tp + fn) if (tp + fn) > 0 else 0) * 100
        f1 = (2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0)
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

        return (acc, prec, rec, f1), epoch_loss / len(dataloader), rep_loss_val / len(dataloader), classification_loss_val / len(dataloader)

def save_model(model, path="siamese_model.pth", dir="saved_models"):
    # create directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, path)
    
    # save the model state
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def load_model(model, path="siamese_model.pth", device="cpu", dir="saved_models"):
    path = os.path.join(dir, path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()
        
    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset patience
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered! Stopping training.")
                return True
        return False