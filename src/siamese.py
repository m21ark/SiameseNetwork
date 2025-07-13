import torch
import torch.nn as nn

from src.vgg_features import vgg19_features

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.isClassifierFrozen = False
        
         # ==================================== 1. Feature Extractor ====================================
        
        self.features = vgg19_features(pretrained=True)
        temp = [i for i in self.features.modules() if isinstance(i, nn.Conv2d)]
        first_add_on_layer_in_channels = temp[-1].out_channels  # 512

        # ==================================== 2. Representation Generator ====================================
        
        desired_channel_count = 128
        self.rep_gen = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=desired_channel_count, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=desired_channel_count, out_channels=desired_channel_count, kernel_size=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1),  # Convert (7,7,128) -> (1,1,128)
            nn.Flatten() # Convert (1,1,128) -> (128)
        )
        
        # ==================================== 3. Classifier ====================================
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 2, 128),  # Concatenate embeddings (128+128)
            torch.nn.ReLU(),
            # nn.Dropout(0.3), # For regularization
            torch.nn.Linear(128, 1),  
            torch.nn.Sigmoid()  # Output probability (0 to 1)
        )
        
    def _forward_once(self, x):
        x = self.features(x)  # Extract feature maps
        x = self.rep_gen(x)  # Convert to representation
        return x
    
    def forward_pair(self, img1, img2):
        # Forward pass to get their representations
        rep1 = self._forward_once(img1)
        rep2 = self._forward_once(img2)

        pred = self._predict_similarity_score(rep1, rep2)
        
        return rep1, rep2, pred
    
    def forward_triplet(self, anchor, pos, neg):
        # Forward pass to get their representations
        anchor = self._forward_once(anchor)
        pos = self._forward_once(pos)
        neg = self._forward_once(neg)

        pred_pos = self._predict_similarity_score(anchor, pos)
        pred_neg = self._predict_similarity_score(anchor, neg)
       
        return anchor, pos, neg, pred_pos, pred_neg
    
    def _rep_distance(self, rep1, rep2):
        # rep: (batch_size, 128)
        return torch.sum((rep1 - rep2) ** 2, dim=1)
    
    def contrastive_rep_loss(self, rep1, rep2, label, margin=2.0):
        
        # Normalize representations
        # rep1 = nn.functional.normalize(rep1, p=2, dim=1)
        # rep2 = nn.functional.normalize(rep2, p=2, dim=1)
        
        squared_dist = self._rep_distance(rep1, rep2) # squared distance

        # Corrected loss function where 1 means similar and 0 means dissimilar
        eps = 1e-6  # for numerical stability
        lin_dist = torch.sqrt(squared_dist + eps)
        loss = label * squared_dist + (1 - label) * (torch.relu(margin - lin_dist) ** 2)
        return loss.mean()
    
    def _predict_similarity_score(self, rep1, rep2):
        # Concatenate embeddings
        combined = torch.cat((rep1, rep2), dim=1)
        
        # Pass through classifier
        similarity_score = self.classifier(combined)
        return similarity_score
    
    def triplet_rep_loss(self, anchor, pos, neg, margin=1.0):
        
        # Normalize representations
        # anchor = nn.functional.normalize(anchor, p=2, dim=1)
        # pos = nn.functional.normalize(pos, p=2, dim=1)
        # neg = nn.functional.normalize(neg, p=2, dim=1)
        
        # Compute distance between pairs
        pos_dist = self._rep_distance(anchor, pos)
        neg_dist = self._rep_distance(anchor, neg)
        
        # Compute loss
        loss = torch.relu(pos_dist - neg_dist + margin)
        return  loss.mean()
            
    def set_train_mode_representation(self):
            # Freeze the classifier
            for param in self.classifier.parameters():
                param.requires_grad = False
                
            self.isClassifierFreezed = True
                
            # Unfreeze the feature extractor and representation generator
            for param in self.features.parameters():
                param.requires_grad = True
            for param in self.rep_gen.parameters():
                param.requires_grad = True
                
    def set_train_mode_classifier(self):
            # Freeze the feature extractor and representation generator
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.rep_gen.parameters():
                param.requires_grad = False
                
            # Unfreeze the classifier
            for param in self.classifier.parameters():
                param.requires_grad = True
                
            self.isClassifierFreezed = False
            
    def set_joint_train_mode(self):
        # Unfreeze all layers for joint training
            for param in self.features.parameters():
                param.requires_grad = True
            for param in self.rep_gen.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True