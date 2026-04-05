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
    mat_files_folder = "X:\\project\\MEG_ChildrenBiomarker\\\\yourpath"  # .mat file
    save_folder = "X:\\project\\MEG_ChildrenBiomarker\\yourpath"  # images
    psd_folder = "X:\\project\\MEG_ChildrenBiomarker\\yourpath"  # PSD images
    save_model_path = "X:\\project\\MEG_ChildrenBiomarker\\yourpath"
else:
    mat_files_folder = "/work/project/MEG_ChildrenBiomarker//yourpath/"
    save_folder = "/work/project/MEG_ChildrenBiomarker/yourpath/"
    psd_folder = "/work/project/MEG_ChildrenBiomarker/yourpath/"
    save_model_path = "/work/project/MEG_ChildrenBiomarker/yourpath/"
# =======================================================

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
best_auc = 0.0
save_dir = os.path.dirname(save_model_path)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
    print(f"Created directory for saving model: {save_model_path}")

########################################Custom Dataset Handling Class################################################
class MultimodalDataset(Dataset):
    def __init__(self, signal_paths, psd_paths, labels, transform=None):
        self.signal_paths = signal_paths
        self.psd_paths = psd_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signal_paths)

    def __getitem__(self, idx):
        signal_path = self.signal_paths[idx]
        psd_path = self.psd_paths[idx]
        label = self.labels[idx]
        
        # Load original signal image and PSD image
        signal_image = Image.open(signal_path).convert("L")  # gray
        psd_image = Image.open(psd_path).convert("L")  # gray
        
        if self.transform:
            signal_image = self.transform(signal_image)
            psd_image = self.transform(psd_image)
            
        return signal_image, psd_image, label

########################################Data Loading Functions################################################
def load_multimodal_data(signal_dir, psd_dir):

    signal_paths = []
    psd_paths = []
    labels = []
    groups = []
    
    excluded_subjects = ['S1RETD0002', 'S1RETD0004', 'S1RETD0005', 'S1RETD0007', 'S2CL0332']#skip if you don't have exluded subjects
    
    signal_files = {}
    for filename in os.listdir(signal_dir):
        if filename.endswith(".png") and ("CL" in filename or "TD" in filename):
            parts = filename.split('_')
            if len(parts) >= 4 and parts[-2] == "channel": 
                label_str = parts[0]
                if parts[1].endswith('SSP'):
                    subject_id = parts[1][:-3] 
                else:
                    subject_id = parts[1]
                if subject_id in excluded_subjects:
                    continue
                channel_str = parts[-1].split('.')[0]
                channel_num = int(channel_str)
                channel_str = f"{channel_num:03d}"
                key = (label_str, subject_id, channel_str)
                signal_files[key] = os.path.join(signal_dir, filename)
    
    for filename in os.listdir(psd_dir):
        if filename.endswith("_psd.png"):

            parts = filename.split('_')
            if len(parts) >= 5 and parts[2] == "channel":
                label_str = parts[0]
                subject_id = parts[1]
                channel_str = parts[3]
                
                if subject_id in excluded_subjects:
                    continue
                
                channel_num = int(channel_str)
                channel_str = f"{channel_num:03d}"
                key = (label_str, subject_id, channel_str)
                
                if key in signal_files:
                    signal_paths.append(signal_files[key])
                    psd_paths.append(os.path.join(psd_dir, filename))
                    labels.append(1 if label_str == "CL" else 0)
                    groups.append(f"{label_str}_{subject_id}")
    
    print(f"Load: {len(signal_paths)} images, CL={labels.count(1)}, TD={labels.count(0)}, {len(set(groups))} subjects")
    return signal_paths, psd_paths, labels, groups

transform = transforms.Compose([
    transforms.Resize((310, 154)),  # resize
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
])

signal_paths, psd_paths, labels, groups = load_multimodal_data(save_folder, psd_folder)

########################################Model Definition################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class MultimodalConvNet(nn.Module):
    def __init__(self):
        super(MultimodalConvNet, self).__init__()

        self.signal_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.signal_bn1 = nn.BatchNorm2d(32)
        self.signal_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)
        
        self.signal_resblock1 = ResidualBlock(32, 64, stride=2)
        self.signal_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.signal_resblock2 = ResidualBlock(64, 128, stride=2)
        self.signal_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.signal_resblock3 = ResidualBlock(128, 128, stride=1)
        self.signal_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.psd_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.psd_bn1 = nn.BatchNorm2d(32)
        self.psd_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)
        
        self.psd_resblock1 = ResidualBlock(32, 64, stride=2)
        self.psd_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.psd_resblock2 = ResidualBlock(64, 128, stride=2)
        self.psd_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.psd_resblock3 = ResidualBlock(128, 128, stride=1)
        self.psd_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)

        self.fc1 = nn.Linear(128 * 12 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, signal_img, psd_img):

        x1 = F.relu(self.signal_bn1(self.signal_conv1(signal_img)))
        x1 = self.signal_pool1(x1)
        x1 = self.signal_resblock1(x1)
        x1 = self.signal_pool2(x1)
        x1 = self.signal_resblock2(x1)
        x1 = self.signal_pool3(x1)
        x1 = self.signal_resblock3(x1)
        x1 = self.signal_pool4(x1)
        x1 = x1.view(x1.size(0), -1)
        
        x2 = F.relu(self.psd_bn1(self.psd_conv1(psd_img)))
        x2 = self.psd_pool1(x2)
        x2 = self.psd_resblock1(x2)
        x2 = self.psd_pool2(x2)
        x2 = self.psd_resblock2(x2)
        x2 = self.psd_pool3(x2)
        x2 = self.psd_resblock3(x2)
        x2 = self.psd_pool4(x2)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

########################################Majority Voting################################################
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
            final_labels[person_id] = 1  # CL
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

    for signal_data, psd_data, target in train_loader:
        signal_data, psd_data, target = signal_data.to(device), psd_data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(signal_data, psd_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

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
        for signal_data, psd_data, target in val_loader:
            signal_data, psd_data, target = signal_data.to(device), psd_data.to(device), target.to(device)
            output = model(signal_data, psd_data)
            loss = criterion(output, target)
            running_loss += loss.item()

            probs = F.softmax(output, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
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
        for batch_idx, (signal_data, psd_data, target) in enumerate(val_loader): 
            signal_data, psd_data, target = signal_data.to(device), psd_data.to(device), target.to(device)
            output = model(signal_data, psd_data) 
            probs = F.softmax(output, dim=1)[:, 1]
            
            for idx in range(signal_data.size(0)):
                sample_index = batch_idx * val_loader.batch_size + idx
                if sample_index < len(val_dataset.signal_paths): 
                    file_name = os.path.basename(val_dataset.signal_paths[sample_index])
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

def evaluate_subject_level(model, device, val_loader, val_dataset):

    model.eval()
    person_probs = {}
    subject_true_labels = {}
    
    with torch.no_grad():
        for batch_idx, (signal_data, psd_data, target) in enumerate(val_loader): 
            signal_data, psd_data, target = signal_data.to(device), psd_data.to(device), target.to(device)
            output = model(signal_data, psd_data)  
            probs = F.softmax(output, dim=1)[:, 1]
            
            for idx in range(signal_data.size(0)):
                sample_index = batch_idx * val_loader.batch_size + idx
                if sample_index < len(val_dataset.signal_paths): 
                    file_name = os.path.basename(val_dataset.signal_paths[sample_index])
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

########################################10 fold cross validation################################################
def run_10fold_cross_validation(experiment_id=None):

    if experiment_id is None:
        import datetime
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{script_name}_{timestamp}"
    
    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
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

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(signal_paths, labels, groups)):
        print(f"=================For {fold + 1} fold ==================")
        
        train_signal_paths = [signal_paths[i] for i in train_idx]
        train_psd_paths = [psd_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_groups = [groups[i] for i in train_idx]
        
        val_signal_paths = [signal_paths[i] for i in val_idx]
        val_psd_paths = [psd_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_groups = [groups[i] for i in val_idx]
        
        print(f"Traning number: {len(train_signal_paths)}")
        print(f"Validation number: {len(val_signal_paths)}")
        
        train_dataset = MultimodalDataset(train_signal_paths, train_psd_paths, train_labels, transform=transform)
        val_dataset = MultimodalDataset(val_signal_paths, val_psd_paths, val_labels, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = MultimodalConvNet().to(device)
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
        
        print(f"\nStart {fold + 1} fold training...")
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
                model_filename = f'best_multimodal_model_{experiment_id}_fold_{fold+1}.pth'
                torch.save(model.state_dict(), os.path.join(save_model_path, model_filename))
                print(f"  Save best model: {model_filename}")
            
            scheduler.step()
            
            if epoch % 5 == 0 or epoch == num_epochs:
                print(f'Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train AUC={train_auc:.4f}')
                print(f'           Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}')
        
        model_filename = f'best_multimodal_model_{experiment_id}_fold_{fold+1}.pth'
        model.load_state_dict(torch.load(os.path.join(save_model_path, model_filename)))
        
        final_val_loss, final_val_acc, final_val_auc, _, _ = validate(
            model, device, val_loader, criterion
        )
        
        subj_accuracy, subj_auc, subj_f1, _, _ = evaluate_subject_level(
            model, device, val_loader, val_dataset
        )
        
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
        
        print(f"\n{fold + 1} fold final results:")
        print(f"Sample level - Accuracy: {final_val_acc:.2f}%, AUC: {final_val_auc:.4f}")
        print(f"Subject level - Accuracy: {subj_accuracy*100:.2f}%, AUC: {subj_auc:.4f}, F1: {subj_f1:.4f}")
        print(f"Best threshold: {best_threshold_fold:.3f}")
        print("-" * 60)
    
    return fold_results, all_fold_histories

########################################Analyze and plot results################################################
def analyze_and_plot_results(fold_results, all_fold_histories, experiment_id=None):
    if experiment_id is None:
        experiment_id = "multimodal_exp"
        
    print("\n" + "="*60)
    print(f"Multimodal 10 fold cross validation final results statistics [{experiment_id}]")
    print("="*60)
    
    print("\nSample level results:")
    sample_acc_mean = np.mean(fold_results['val_accuracies'])
    sample_acc_std = np.std(fold_results['val_accuracies'])
    sample_auc_mean = np.mean(fold_results['sample_level_auc'])
    sample_auc_std = np.std(fold_results['sample_level_auc'])
    
    print(f"Accuracy: {sample_acc_mean:.2f}% ± {sample_acc_std:.2f}%")
    print(f"AUC-ROC: {sample_auc_mean:.4f} ± {sample_auc_std:.4f}")
    
    print("\nSubject level results:")
    subj_acc_mean = np.mean(fold_results['subject_level_accuracy'])
    subj_acc_std = np.std(fold_results['subject_level_accuracy'])
    subj_auc_mean = np.mean(fold_results['subject_level_auc'])
    subj_auc_std = np.std(fold_results['subject_level_auc'])
    subj_f1_mean = np.mean(fold_results['subject_level_f1'])
    subj_f1_std = np.std(fold_results['subject_level_f1'])
    
    print(f"Accuracy: {subj_acc_mean:.2f}% ± {subj_acc_std:.2f}%")
    print(f"AUC-ROC: {subj_auc_mean:.4f} ± {subj_auc_std:.4f}")
    print(f"F1-Score: {subj_f1_mean:.4f} ± {subj_f1_std:.4f}")
    
    print("\nDetailed results for each fold:")
    results_df = pd.DataFrame({
        'Fold': fold_results['fold'],
        'Sample_Acc(%)': [f"{acc:.2f}" for acc in fold_results['val_accuracies']],
        'Sample_AUC': [f"{auc:.4f}" for auc in fold_results['sample_level_auc']],
        'Subject_Acc(%)': [f"{acc:.2f}" for acc in fold_results['subject_level_accuracy']],
        'Subject_AUC': [f"{auc:.4f}" for auc in fold_results['subject_level_auc']],
        'Subject_F1': [f"{f1:.4f}" for f1 in fold_results['subject_level_f1']]
    })
    print(results_df.to_string(index=False))
    
    results_filename = f'10fold_multimodal_detailed_results_{experiment_id}.csv'
    results_df.to_csv(os.path.join(save_model_path, results_filename), index=False)
    print(f"\nDetailed results saved: {results_filename}")

    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 4, 1)
    for fold, history in enumerate(all_fold_histories):
        epochs = range(1, len(history['train_losses']) + 1)
        plt.plot(epochs, history['train_losses'], alpha=0.3, color='blue')
        plt.plot(epochs, history['val_losses'], alpha=0.3, color='red')
    
    avg_train_loss = np.mean([h['train_losses'] for h in all_fold_histories], axis=0)
    avg_val_loss = np.mean([h['val_losses'] for h in all_fold_histories], axis=0)
    epochs = range(1, len(avg_train_loss) + 1)
    plt.plot(epochs, avg_train_loss, 'b-', linewidth=2, label='Average training loss')
    plt.plot(epochs, avg_val_loss, 'r-', linewidth=2, label='Average validation loss')
    plt.title(f'Multimodal training and validation loss\n[{experiment_id}]')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 2)
    for fold, history in enumerate(all_fold_histories):
        epochs = range(1, len(history['train_accuracies']) + 1)
        plt.plot(epochs, history['train_accuracies'], alpha=0.3, color='blue')
        plt.plot(epochs, history['val_accuracies'], alpha=0.3, color='red')
    
    avg_train_acc = np.mean([h['train_accuracies'] for h in all_fold_histories], axis=0)
    avg_val_acc = np.mean([h['val_accuracies'] for h in all_fold_histories], axis=0)
    plt.plot(epochs, avg_train_acc, 'b-', linewidth=2, label='Average training accuracy')
    plt.plot(epochs, avg_val_acc, 'r-', linewidth=2, label='Average validation accuracy')
    plt.title(f'Multimodal training and validation accuracy\n[{experiment_id}]')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 4, 3)
    for fold, history in enumerate(all_fold_histories):
        epochs = range(1, len(history['train_aucs']) + 1)
        plt.plot(epochs, history['train_aucs'], alpha=0.3, color='blue')
        plt.plot(epochs, history['val_aucs'], alpha=0.3, color='red')
    
    avg_train_auc = np.mean([h['train_aucs'] for h in all_fold_histories], axis=0)
    avg_val_auc = np.mean([h['val_aucs'] for h in all_fold_histories], axis=0)
    plt.plot(epochs, avg_train_auc, 'b-', linewidth=2, label='Average training AUC')
    plt.plot(epochs, avg_val_auc, 'r-', linewidth=2, label='Average validation AUC')
    plt.title(f'Multimodal training and validation AUC\n[{experiment_id}]')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 4, 4)
    metrics = ['Accuracy', 'AUC', 'F1']
    means = [subj_acc_mean, subj_auc_mean*100, subj_f1_mean*100]
    stds = [subj_acc_std, subj_auc_std*100, subj_f1_std*100]
    
    x_pos = np.arange(len(metrics))
    plt.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
    plt.xticks(x_pos, metrics)
    plt.ylabel('Performance (%)')
    plt.title(f'Multimodal subject level performance\n[{experiment_id}]')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    plot_filename = f'multimodal_training_curves_{experiment_id}.png'
    plt.savefig(os.path.join(save_model_path, plot_filename), dpi=300, bbox_inches='tight')
    print(f"Training curves saved: {plot_filename}")
    plt.show()

    summary_filename = f'10fold_multimodal_summary_{experiment_id}.txt'
    with open(os.path.join(save_model_path, summary_filename), 'w', encoding='utf-8') as f:
        f.write(f"Multimodal 10 fold cross validation results summary [{experiment_id}]\n")
        f.write("="*50 + "\n\n")
        f.write("Sample level results:\n")
        f.write(f"Accuracy: {sample_acc_mean:.2f}% ± {sample_acc_std:.2f}%\n")
        f.write(f"AUC-ROC: {sample_auc_mean:.4f} ± {sample_auc_std:.4f}\n\n")
        f.write("Subject level results:\n")
        f.write(f"Accuracy: {subj_acc_mean:.2f}% ± {subj_acc_std:.2f}%\n")
        f.write(f"AUC-ROC: {subj_auc_mean:.4f} ± {subj_auc_std:.4f}\n")
        f.write(f"F1-Score: {subj_f1_mean:.4f} ± {subj_f1_std:.4f}\n")
    
    print(f"Summary results saved: {summary_filename}")
    print(f"\nMultimodal results saved to: {save_model_path}")

########################################Execute multimodal 10 fold cross validation################################################
if __name__ == "__main__":
    print("Start multimodal fusion experiment...")
    
    if not os.path.exists(psd_folder):
        print(f"Error: PSD data directory not found: {psd_folder}")
        print("Please run PSD generation code to generate PSD images")
        exit(1)
    
    import datetime
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{script_name}_{timestamp}"
    
    print(f"Experiment ID: {experiment_id}")
    print("Start multimodal 10 fold cross validation (concat fusion)...")
    fold_results, all_fold_histories = run_10fold_cross_validation(experiment_id=experiment_id)
    print("\nAnalyze and plot results...")
    analyze_and_plot_results(fold_results, all_fold_histories, experiment_id=experiment_id)