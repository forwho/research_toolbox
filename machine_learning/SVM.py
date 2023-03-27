from machine_learning.layer import ActLinear, ActLinear_2
import torch
import pytorch_lightning as pl

def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1 - y_pred.t() * y_true, min=0))

class SVM(pl.LightningModule):
    def __init__(
        self, n_inputs: int = 30135, l1_lambda=1, learning_rate=0.05, is_act=False):
        super().__init__()

        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.loss_fn = hinge_loss
        self.layer_1=ActLinear_2(n_inputs)
        self.layer_2 = torch.nn.Linear(n_inputs, 1)
        if is_act:
             self.model=torch.nn.Sequential(self.layer_1,torch.nn.ReLU(),self.layer_2)
            # self.model=torch.nn.Sequential(torch.nn.BatchNorm1d(n_inputs),torch.nn.ReLU(),self.layer_2)
        else:
            self.model=self.layer_2
        self.train_log = []

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def l2_reg(self):
        l2_norm = self.layer_2.weight.pow(2).sum()
        
        return  self.l1_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l2_reg()
        self.log("loss", loss)
        pred_label=torch.sign(y_hat)
        acc=torch.mul(pred_label,y)
        acc=torch.sum(torch.where(acc>0,1,0))/y.size()[0]
        self.log("acc",acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        pred_label=torch.sign(y_hat)
        acc=torch.mul(pred_label,y)
        acc=torch.sum(torch.where(acc>0,1,0))/y.size()[0]
        self.log("acc",acc)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def get_weights(self):
        return self.layer_1.weight, self.layer_1.bias, self.layer_2.weight

    def get_attmap(self,x):
        x=self.layer_1.forward(x)
        m=torch.nn.ReLU()
        x=m(x)
        x=torch.mul(x, self.layer_2.weight)
        return x

    def get_isselect(self,x):
        x=self.layer_1.forward(x)
        m=torch.nn.ReLU()
        x=m(x)
        x=x.detach().numpy()
        x[x>0]=1
        return x
