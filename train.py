import torch.optim.optimizer as opt



epoch_count = 1
n_epochs = 100
n_epochs_decay = 100
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l


scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)