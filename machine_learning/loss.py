import torch

def L2Loss(model,device):
    l2_loss=torch.tensor(0.0,requires_grad=True).to(device)
    for name,param in model.named_parameters():
        if 'bias' not in name:
            l2_loss=l2_loss+torch.sum(torch.pow(param,2))
    return l2_loss

def L1Loss(model,device):
    l1_loss=torch.tensor(0.0,requires_grad=True).to(device)
    for name,param in model.named_parameters():
        if 'bias' not in name:
            l1_loss=l1_loss+torch.sum(torch.abs(param))
    return l1_loss

def loss_with_regularization_factor(alpha,lam,model):
    def loss_with_regularization_factor(y_pred,y_true):
        los_func=torch.nn.MSELoss()
        pred_loss=los_func(y_pred,y_true)
        reg_loss=alpha*lam*L1Loss(model)+alpha*(1-lam)*L2Loss(model)
        total_loss=pred_loss+reg_loss
        return(total_loss)
    return loss_with_regularization_factor