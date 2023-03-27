from research_toolbox.machine_learning.layer import ActLinear, ActLinear_2
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from research_toolbox.machine_learning.stopping import MyEarlyStopping
from research_toolbox.machine_learning.Datasets import numpy2loader
import numpy as np

class ElasticLinear(pl.LightningModule):
    def __init__(
        self, loss_fn, n_inputs: int = 30135, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05, is_act=False, is_norm=False
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
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

    def l1_reg(self):
        l1_norm = self.layer_2.weight.abs().sum()

        return self.l1_lambda * self.l2_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.layer_2.weight.pow(2).sum()
        
        return 0.5 * self.l1_lambda * (1-self.l2_lambda) * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()
        rval = torch.corrcoef(torch.cat((y.T,y_hat.T),0))[0,1]
        self.log("weight of 1st layer", self.layer_1.weight[0])
        self.log("bias of 1st layer", self.layer_1.bias[0])
        self.log("train rval", rval)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        rval = torch.corrcoef(torch.cat((y.T,y_hat.T),0))[0,1]
        bias = torch.mean(y_hat-y)
        self.log('rval',rval)
        self.log("mse", loss)
        self.log('bias',bias)
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

class AttMLP(pl.LightningModule):
    def __init__(
        self, loss_fn, n_inputs: int = 30135, learning_rate=0.05):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        self.Q=torch.nn.Linear(n_inputs,n_inputs)

        self.nn1=torch.nn.Linear(n_inputs,10)
        self.nn2=torch.nn.Linear(10,1)
        self.model=torch.nn.Sequential(self.nn1,torch.nn.ReLU(),self.nn2)
        self.train_log = []

    def forward(self, x):
        att_prob=self.Q(x)
        att_encod=torch.mul(att_prob,x)
        return self.model(att_encod)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # loss = self.loss_fn(y_hat, y)
        
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        bias = torch.mean(y_hat-y)
        self.log("mse", loss,on_step=True,on_epoch=True,sync_dist=True)
        self.log('bias',bias,on_step=True,on_epoch=True,sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def get_attmap(self,x):
        return self.Q(x)


def train(train_features,train_labels,val_features,val_labels,l1_lambda,l2_lambda,is_act=True, norm_method='minmax',max_epochs=10000,patience=100):
    trainer = pl.Trainer(gpus=[0],max_epochs=max_epochs,enable_checkpointing=False,num_sanity_val_steps=1,log_every_n_steps=1,callbacks=[MyEarlyStopping(monitor='loss',mode='min',patience=patience)])
    model=ElasticLinear(
        loss_fn=torch.nn.MSELoss(),
        n_inputs=train_features.shape[1],
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda,
        learning_rate=0.005,
        is_act=is_act
    )
    if norm_method=='minmax':
        tran=MinMaxScaler()
    elif norm_method=='norm':
        tran=StandardScaler()
    train_features=tran.fit_transform(train_features)
    val_features=tran.transform(val_features)
    dataloader_train=numpy2loader(train_features,train_labels,True)
    dataloader_val=numpy2loader(val_features,val_labels,False)
    trainer.fit(model, dataloader_train, dataloader_val)
    val_hat=trainer.predict(model, dataloader_val)
    rval=np.corrcoef(val_hat[0].numpy().T,val_labels.T)[0,1]
    return rval,model,trainer,tran,val_hat
