import numpy as np
import torch
import optuna
import h5py
from utils.utils_transmil import *
import os
import sys
import pandas as pd
from matplotlib.pylab import plt
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam_twoclass import CLAM_MB, CLAM_SB
from models.model_transmil import TransMIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   #assigns rank to the data and deals with ties appropriately; calculates the percentage
    return scores
    
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

result_search = []
test_auc_search = []
val_auc_search = []
test_acc_search = []
val_acc_search = []

def objective(trial, datasets, cur, args):
    """   
        train for a single fold
    """
    l2_weight = trial.suggest_categorical('l2_weight', [5,10,20,50,100])
    lr = trial.suggest_categorical('lr', [2e-7,2e-6,2e-5,2e-4])
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur)+str("_")+str(lr)+str("_")+str(l2_weight))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}_{}_{}.csv'.format(cur, lr, l2_weight)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        print("svm bag loss")
        print("bag loss", args.bag_loss)
        from smooth_topk.topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        print("ce bag loss")
        print("bag loss", args.bag_loss)
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = { 'n_classes': args.n_classes}
    
    if args.model_type in ['clam_sb', 'clam_mb', 'transmil']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
  
        if args.inst_loss == 'svm':
            print("svm instance loss")
            print("instance loss", args.inst_loss)
            from smooth_topk.topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            print("ce instance loss")
            print("instance loss", args.inst_loss)
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            print("calling clam mb")
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'transmil':
            print("calling transmil")
            model = TransMIL(**model_dict).cuda()
        else:
            raise NotImplementedError
    
    else: 
        if args.n_classes > 2:
            print("testing mil fc mc")
            model = MIL_fc_mc(**model_dict)
        else:
            print("testing mil")
            model = MIL_fc(**model_dict)
    
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args,lr)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')
    
    
    plot_losses_total, plot_losses_val = [], []
    plot_losses_train, plot_val_class = [], []
    plot_losses_feed, plot_val_feed = [], []
    epoch_count = 0
    for epoch in range(args.max_epochs):
        running_loss_val = 0.0
        running_loss_total = 0.0
        epoch_count += 1
        if args.model_type in ['clam_sb', 'clam_mb', 'transmil'] and not args.no_inst_cluster:  
            print("Learning rate", lr, "l2_weight", l2_weight)
            epoch_loss_total, train_loss, l2_loss_value = train_loop_transmil(epoch, model, l2_weight, lr, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            plot_losses_total.append(epoch_loss_total)
            plot_losses_train.append(train_loss)
            plot_losses_feed.append(l2_loss_value)
            stop, epoch_val_loss, val_class_loss, l2_val_loss = validate_transmil(cur, epoch, model,l2_weight, lr, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
            plot_losses_val.append(epoch_val_loss)
            plot_val_class.append(val_class_loss)
            plot_val_feed.append(l2_val_loss)
            
        
        else:
            print("train_loop")
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint_{}_{}.pt".format(cur, lr, l2_weight))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_{}_{}.pt".format(cur, lr, l2_weight)))

    epoch_count_range = range(1, epoch_count+1)
    plt.plot(epoch_count_range, plot_losses_total, label='train')
    plt.plot(epoch_count_range, plot_losses_val, label='validation')
    plt.title('total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(args.results_dir,"total_loss_for_{}_checkpoint_{}_{}.png".format(cur, lr, l2_weight)))
    plt.clf()
    
    plt.plot(epoch_count_range, plot_losses_feed, label='train')
    plt.plot(epoch_count_range, plot_val_feed, label='validation')
    plt.title('feedback Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(args.results_dir,"feedback_loss_for_{}_checkpoint_{}_{}.png".format(cur, lr, l2_weight)))
    plt.clf()
    
    plt.plot(epoch_count_range, plot_losses_train, label='train')
    plt.plot(epoch_count_range, plot_val_class, label='validation')
    plt.title('classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(args.results_dir,"classification_loss_for_{}_checkpoint_{}_{}.png".format(cur, lr, l2_weight)))
    plt.clf()
    
    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    result_search.append(results_dict)
    test_auc_search.append(test_auc)
    test_acc_search.append(1-test_error)
    val_auc_search.append(val_auc)
    val_acc_search.append(1-val_error)
    best_val_sofar = max(test_acc_search)
    print("max acc",best_val_sofar)
    print("max acc at", test_acc_search.index(best_val_sofar))
    text_file_name = os.path.join(args.results_dir, 'prediction_metrics_split_{}_lr_{}_l2weight_{}.txt'.format(cur,lr,l2_weight))
    np.savetxt(text_file_name, (np.array(cur),np.array(test_auc), np.array(val_auc), np.array(1-test_error), np.array(1-val_error)))
    return 1-val_error #val_auc
	

def train(datasets, cur, args):
    study = optuna.create_study(direction='maximize')
    best_value_search = None
    for l2_weight in [5,10,20,50,100]:
        for lr in [2e-7,2e-6,2e-5,2e-4]:
            study.enqueue_trial({'l2_weight': l2_weight,'lr': lr})
    study.optimize(lambda trial: objective(trial, datasets, cur, args), n_trials=20)
    best_value_search = study.best_value
    max_index = val_acc_search.index(best_value_search)
    return result_search[max_index], test_auc_search[max_index], val_auc_search[max_index], test_acc_search[max_index], val_acc_search[max_index]


def train_loop_transmil(epoch, model,l2_weight,lr, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    torch.autograd.set_detect_anomaly(True) 
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    running_loss_total = 0.
    train_loss = 0.
    train_error = 0.
    l2_loss_value = 0.

    print('\n')
    slide_ids = loader.dataset.slide_data['slide_id']
    slide_labs = loader.dataset.slide_data['label']
    previous_batch_id = 0
    current_batch_id = 0
    for batch_idx, (data, label, _, slide_id, mn_M) in enumerate(loader): #feedback+no_feedback = train = 50
        if mn_M != None:
        	data, label, mn_M = data.to(device), label.to(device), mn_M.to(device)
        else:
        	data, label = data.to(device), label.to(device)
        data = data.unsqueeze(0)
        logits, Y_prob, Y_hat,  A_score, mn_new = model(data, label, mn_M)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        from torch.cuda import amp
        with amp.autocast():
            l2_loss = l2_weight * F.mse_loss(A_score.unsqueeze(0),mn_new.unsqueeze(0)) #5, 10(best auc), 20(best acc) 50, 100
            l2_val = l2_loss.item()
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            total_loss = loss + l2_loss
            train_loss += loss_value
            l2_loss_value += l2_val
            if (batch_idx + 1) % 20 == 0: print('batch {}, loss: {:.4f},  l2_loss: {:4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, l2_val, total_loss.item()) + 
                'label: {}, bag_size: {}, learning_rate: {}, l2_weight: {}'.format(label.item(), data.size(0), lr, l2_weight))
            error = calculate_error(Y_hat, label)
            train_error += error
            running_loss_total += total_loss.item()
            #a = list(model.parameters())[0].clone()
            total_loss.backward()
            #b = list(model.parameters())[0].clone()
            optimizer.step()
            #c = list(model.parameters())[0].clone()
            #print("check if model parameters are equal before update",torch.equal(a.data, b.data))
            #print("check if model parameters are equal after update",torch.equal(a.data, c.data))
            #print(torch.equal(b.data, c.data))
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    l2_loss_value /= len(loader)
    train_error /= len(loader)
    running_loss_total /= len(loader)
    
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,  train_error))
    allocated_memory = torch.cuda.memory_allocated()
    print("Allocated GPU memory:", allocated_memory, "bytes")
    cached_memory = torch.cuda.memory_reserved()
    print("Cached GPU memory:", cached_memory, "bytes")
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    return running_loss_total, train_loss, l2_loss_value


def validate_transmil(cur, epoch, model, l2_weight, lr, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    epoch_val_loss = 0.
    l2_loss_value = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, _, _,mn_M) in enumerate(loader):
            if mn_M != None:
            	data, label, mn_M = data.to(device), label.to(device), mn_M.to(device)
            else:
            	data, label = data.to(device), label.to(device)   
            data = data.unsqueeze(0)
            logits, Y_prob, Y_hat,  A_score, mn_new = model(data, label, mn_M)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            
            from torch.cuda import amp
            with amp.autocast():
            	l2_loss = l2_weight *  F.mse_loss(A_score.unsqueeze(0),mn_new.unsqueeze(0)) # 5, 10(best), 50, 100
            	l2_val = l2_loss.item()
            	l2_loss_value += l2_val
            	prob[batch_idx] = Y_prob.cpu().numpy()
            	labels[batch_idx] = label.item()
            	error = calculate_error(Y_hat, label)
            	val_error += error
            	total_loss =  loss  + l2_loss
            	epoch_val_loss += total_loss.item()

    val_error /= len(loader)
    val_loss /= len(loader)
    l2_loss_value /= len(loader)
    epoch_val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, l2_loss: {:.4f}, weighted_loss:{:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, l2_loss_value, epoch_val_loss, val_error, auc))
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, epoch_val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint_{}_{}.pt".format(cur,lr, l2_weight)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, epoch_val_loss, val_loss, l2_loss_value

    return False,epoch_val_loss, val_loss, l2_loss_value

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, _, _,mn_M) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        data = data.unsqueeze(0)
        with torch.no_grad():
            logits, Y_prob, Y_hat,  _, _ = model(data, label, mn_M)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
