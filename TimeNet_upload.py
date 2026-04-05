import os
import platform
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from PIL import Image
import pandas as pd 
import random
# random seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
g = torch.Generator()
g.manual_seed(42)
def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

########################################Initialization of Paths and Basic Settings################################################
# ===== Configure these paths for your environment =====
if platform.system() == "Windows":
    mat_files_folder = "X:\\project\\MEG_ChildrenBiomarker\\yourpath"  # .mat file
    save_folder = "X:\\project\\MEG_ChildrenBiomarker\\yourpath"  # images
    save_model_path = "X:\\project\\MEG_ChildrenBiomarker\\yourpath"
else:
    mat_files_folder = "/work/project/MEG_ChildrenBiomarker/yourpath"
    save_folder = "/work/project/MEG_ChildrenBiomarker/yourpath"
    save_model_path = "/work/project/MEG_ChildrenBiomarker/yourpath"
# =======================================================

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
best_auc = 0.0
save_dir = os.path.dirname(save_model_path)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
    print(f"Created directory for saving model: {save_model_path}")

import datetime
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_id = f"{script_name}_{timestamp}"

# This code is commented out becaused this step has already been completed
"""
# .mat --> image
for filename in os.listdir(mat_files_folder):
    if filename.endswith('.mat'):
        mat_file_path = os.path.join(mat_files_folder, filename)
        
        # set label according name
        if 'CL' in filename:
            label = 'CL'
        elif 'TD' in filename:
            label = 'TD'
        else:
            label = 'Unknown'

        # read .mat file
        with h5py.File(mat_file_path, 'r') as mat_file:
            vertex_signals = np.array(mat_file['vertex_signals'])
            sampling_rate = np.array(mat_file['sampling_rate'])[0]

            time = np.arange(0, vertex_signals.shape[1]) / sampling_rate
            for i in range(vertex_signals.shape[0]):
                plt.figure(figsize=(10, 5))  
                plt.plot(time, vertex_signals[i, :], color='blue')  # plot signal only
                plt.axis('off')
                save_path = os.path.join(save_folder, f"{label}_{filename[:-4]}_channel_{i+1}.png")
                plt.savefig(save_path, dpi=80, bbox_inches='tight', pad_inches=0)  
                plt.close()

        print(f"Processed and saved plots for {filename}")
"""
########################################Custom Dataset Handling Class################################################
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L")  # gray
        if self.transform:
            image = self.transform(image)
        return image, label

########################################Model Definition################################################
# image path
data_dir = save_folder
# load data
def load_data(data_dir):
    file_paths = []
    labels = []
    groups = []

    excluded_subjects = ['S1RETD0002', 'S1RETD0004', 'S1RETD0005', 'S1RETD0007', 'S2CL0332'] #skip if you don't have exluded subjects

    for filename in os.listdir(data_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            should_exclude = any(excluded_id in filename for excluded_id in excluded_subjects)
            if should_exclude:
                continue
            
            label = 1 if "CL" in filename else 0 if "TD" in filename else None
            if label is not None:
                file_paths.append(os.path.join(data_dir, filename))
                labels.append(label)
                groups.append('_'.join(filename.split('_')[:2]))
    return file_paths, labels, groups

# transform
transform = transforms.Compose([
    transforms.Resize((310, 154)),  # rezise
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
])

########################################Model Definition################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model defination
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 32 channel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)
        
        # ResNet 1：32 -> 64
        self.resblock1 = ResidualBlock(32, 64, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        # ResNet 2：64 -> 128
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        # ResNet 3：128 -> 128
        self.resblock3 = ResidualBlock(128, 128, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.fc1 = nn.Linear(128 * 12 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.resblock1(x)
        x = self.pool2(x)
        x = self.resblock2(x)
        x = self.pool3(x)
        x = self.resblock3(x)
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

########################################Majority Voting################################################
# majority voting
def majority_vote(file_paths, predictions):
    person_results = {}
    for path, pred in zip(file_paths, predictions):
        person_id = '_'.join(os.path.basename(path).split('_')[:2])
        if person_id not in person_results:
            person_results[person_id] = []
        person_results[person_id].append(pred)

    final_labels = {}
    for person_id, preds in person_results.items():
        asd_count = sum(preds)
        if asd_count / len(preds) > 0.5:
            final_labels[person_id] = 1  # ASD
        else:
            final_labels[person_id] = 0  # TD
    return final_labels

########################################Training and Validation Logic################################################
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_probs = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # AUC
        probs = F.softmax(output, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    train_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
    print(f'Epoch {epoch}, Train Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy:.2f}%, AUC-ROC: {train_auc:.4f}')
    return avg_loss, accuracy, train_auc

def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            probs = F.softmax(output, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    auc_score = roc_auc_score(all_targets, all_probs)
    print(f' Validation Loss: {running_loss/len(val_loader)}, Validation Accuracy: {accuracy:.2f}%, AUC-ROC: {auc_score:.4f}')
    return avg_loss, accuracy, auc_score, all_targets, all_probs

def find_best_threshold(model, device, val_loader, val_dataset):
    model.eval()
    person_probs = {}
    subject_true_labels = {}
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)[:, 1]
            
            for idx in range(data.size(0)):
                sample_index = batch_idx * val_loader.batch_size + idx
                if sample_index < len(val_dataset.file_paths):
                    file_name = os.path.basename(val_dataset.file_paths[sample_index])
                    person_id = '_'.join(file_name.split('_')[:2])
                    if person_id not in person_probs:
                        person_probs[person_id] = []
                        subject_true_labels[person_id] = target[idx].cpu().item()
                    person_probs[person_id].append(probs[idx].cpu().item())
    
    final_probs = [np.mean(probs) for probs in person_probs.values()]
    final_labels = list(subject_true_labels.values())
    
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    for threshold in thresholds:
        preds = [1 if p > threshold else 0 for p in final_probs]
        f1 = f1_score(final_labels, preds) if len(set(final_labels)) > 1 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, final_probs, final_labels

########################################Model Testing################################################
def evaluate_subject_level(model, device, val_loader, val_dataset):
    model.eval()
    person_probs = {}
    subject_true_labels = {}
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)[:, 1]
            
            for idx in range(data.size(0)):
                sample_index = batch_idx * val_loader.batch_size + idx
                if sample_index < len(val_dataset.file_paths):
                    file_name = os.path.basename(val_dataset.file_paths[sample_index])
                    person_id = '_'.join(file_name.split('_')[:2])
                    if person_id not in person_probs:
                        person_probs[person_id] = []
                        subject_true_labels[person_id] = target[idx].cpu().item()
                    person_probs[person_id].append(probs[idx].cpu().item())
    
    final_probs = []
    final_labels = []
    for person_id in person_probs:
        avg_prob = np.mean(person_probs[person_id])
        final_probs.append(avg_prob)
        final_labels.append(subject_true_labels[person_id])

    predictions = (np.array(final_probs) > 0.5).astype(int)
    accuracy = accuracy_score(final_labels, predictions)
    auc_score = roc_auc_score(final_labels, final_probs)
    f1 = f1_score(final_labels, predictions)

    return accuracy, auc_score, f1, final_probs, final_labels

########################################10折交叉验证主循环################################################
def run_10fold_cross_validation(file_paths, labels, groups):
    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    # storage
    fold_results = {
        'fold': [],
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],  
        'val_accuracies': [],
        'val_aucs': [],
        'sample_level_auc': [],
        'subject_level_accuracy': [],
        'subject_level_auc': [],
        'subject_level_f1': []
    }
    
    all_fold_histories = []
    all_subject_results = []
    all_confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(file_paths, labels, groups)):
        print(f"=================第 {fold + 1} 折 ==================")
        train_paths = [file_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_groups = [groups[i] for i in train_idx]
        
        val_paths = [file_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_groups = [groups[i] for i in val_idx]
        
        train_groups_unique = set(train_groups)
        val_groups_unique = set(val_groups)
        
        print(f"Training: {len(train_paths)}, {len(train_groups_unique)}")
        print(f"Validation: {len(val_paths)}, {len(val_groups_unique)}")
        print(f"Training: {train_groups_unique}")
        print(f"Validation: {val_groups_unique}")

        train_cl_count = train_labels.count(1)
        train_td_count = train_labels.count(0)
        val_cl_count = val_labels.count(1)
        val_td_count = val_labels.count(0)
        
        print(f"Training: CL(1)={train_cl_count}, TD(0)={train_td_count}")
        print(f"Validation: CL(1)={val_cl_count}, TD(0)={val_td_count}")
        
        overlap = train_groups_unique & val_groups_unique
        if overlap:
            print(f" Overlap！！: {overlap}")
        else:
            print(" no overlap")

        def check_data_leakage(train_groups, val_groups):
            train_set = set(train_groups)
            val_set = set(val_groups)
            overlap = train_set & val_set
            
            if overlap:
                raise ValueError(f"Overlap: {overlap}")
            
            return True
        
        train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
        val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        
        fold_history = {
            'train_losses': [],
            'train_accuracies': [],
            'train_aucs': [], 
            'val_losses': [],
            'val_accuracies': [],
            'val_aucs': []
        }
        
        fold_best_auc = 0.0
        num_epochs = 20
        
        print(f"\nStart {fold + 1} fold...")
        for epoch in range(1, num_epochs + 1):

            train_loss, train_acc, train_auc = train(
                model, device, train_loader, optimizer, criterion, epoch
            )
            
            val_loss, val_acc, val_auc, _, _ = validate(
                model, device, val_loader, criterion
            )
            
            fold_history['train_losses'].append(train_loss)
            fold_history['train_accuracies'].append(train_acc)
            fold_history['train_aucs'].append(train_auc)
            fold_history['val_losses'].append(val_loss)
            fold_history['val_accuracies'].append(val_acc)
            fold_history['val_aucs'].append(val_auc)
            
            if val_auc > fold_best_auc:
                fold_best_auc = val_auc
                torch.save(model.state_dict(), 
                    os.path.join(save_model_path, f'best_model_fold_{fold+1}_{experiment_id}.pth'))

            scheduler.step()
            
            if epoch % 5 == 0 or epoch == num_epochs:
                print(f'Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train AUC={train_auc:.4f}')
                print(f'           Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}')
        
        model.load_state_dict(torch.load(os.path.join(save_model_path, f'best_model_fold_{fold+1}_{experiment_id}.pth')))
        
        final_val_loss, final_val_acc, final_val_auc, _, _ = validate(
            model, device, val_loader, criterion
        )
        
        subj_accuracy, subj_auc, subj_f1, subject_probs, subject_labels = evaluate_subject_level(
            model, device, val_loader, val_dataset
        )

        fold_subject_data = {
            'fold': fold + 1,
            'subject_true_labels': subject_labels,    
            'subject_pred_probs': subject_probs,     
            'accuracy': subj_accuracy,
            'auc': subj_auc,
            'f1': subj_f1
        }
        all_subject_results.append(fold_subject_data)
        
        from sklearn.metrics import confusion_matrix
        subject_predictions = (np.array(subject_probs) > 0.5).astype(int)
        cm = confusion_matrix(subject_labels, subject_predictions)
        all_confusion_matrices.append(cm)
      
        best_threshold_fold, _, _ = find_best_threshold(model, device, val_loader, val_dataset)
        
        fold_results['fold'].append(fold + 1)
        fold_results['train_losses'].append(fold_history['train_losses'][-1])
        fold_results['train_accuracies'].append(fold_history['train_accuracies'][-1])
        fold_results['val_losses'].append(final_val_loss)
        fold_results['val_accuracies'].append(final_val_acc)
        fold_results['val_aucs'].append(final_val_auc)
        fold_results['sample_level_auc'].append(final_val_auc)
        fold_results['subject_level_accuracy'].append(subj_accuracy * 100)
        fold_results['subject_level_auc'].append(subj_auc)
        fold_results['subject_level_f1'].append(subj_f1)
      
        all_fold_histories.append(fold_history)
        
        print(f"\nFor {fold + 1} fold:")
        print(f"Sample level - Acc: {final_val_acc:.2f}%, AUC: {final_val_auc:.4f}")
        print(f"Subject level - Acc: {subj_accuracy*100:.2f}%, AUC: {subj_auc:.4f}, F1: {subj_f1:.4f}")
        print(f"Best threshold: {best_threshold_fold:.3f}")
        print("-" * 60)
    
    return fold_results, all_fold_histories, all_subject_results, all_confusion_matrices

########################################run证################################################
if __name__ == "__main__":

    print("Start 10 fold cross validation")
    file_paths, labels, groups = load_data(data_dir)
    fold_results, all_fold_histories, all_subject_results, all_confusion_matrices = run_10fold_cross_validation(file_paths, labels, groups)