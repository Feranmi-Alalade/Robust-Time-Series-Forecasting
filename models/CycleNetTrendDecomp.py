import torch
import torch.nn as nn


class RecurrentCycle(nn.Module):
    def __init__(self,cycle_len,channel_size):
        super(RecurrentCycle,self).__init__()
        self.cycle_len=cycle_len
        self.channel_size=channel_size
        self.data=nn.Parameter(torch.zeros(cycle_len,channel_size),requires_grad=True)

    def forward(self,index,length):
        gather_index=(index.view(-1,1)+torch.arange(length,device=index.device).view(1,-1))%self.cycle_len
        return self.data[gather_index]


class MovingAvg(nn.Module):
    def __init__(self,kernel_size,stride=1):
        super(MovingAvg,self).__init__()
        self.kernel_size=kernel_size
        self.avg=nn.AvgPool1d(kernel_size=kernel_size,stride=stride,padding=0)

    def forward(self,x):
        front=x[:,0:1,:].repeat(1,(self.kernel_size-1)//2,1)
        end=x[:,-1:,:].repeat(1,(self.kernel_size-1)//2,1)
        x=torch.cat([front,x,end],dim=1)
        x=self.avg(x.permute(0,2,1))
        return x.permute(0,2,1)


class TrendBranch(nn.Module):
    def __init__(self,seq_len,pred_len,moving_avg_kernel,trend_layers,d_model):
        super(TrendBranch,self).__init__()
        self.decompose=MovingAvg(moving_avg_kernel,stride=1)

        if trend_layers==1:
            self.trend_head=nn.Linear(seq_len,pred_len)
        elif trend_layers==2:
            self.trend_head=nn.Sequential(
                nn.Linear(seq_len,d_model),
                nn.ReLU(),
                nn.Linear(d_model,pred_len)
            )
        else:
            raise ValueError("trend_layers must be 1 or 2")

    def forward(self,x):
        trend=self.decompose(x)
        trend_forecast=self.trend_head(trend.permute(0,2,1)).permute(0,2,1)
        return trend,trend_forecast


class Model(nn.Module):
    def __init__(self,configs):
        super(Model,self).__init__()
        self.seq_len=configs.seq_len
        self.pred_len=configs.pred_len
        self.enc_in=configs.enc_in
        self.cycle_len=configs.cycle
        self.model_type=configs.model_type
        self.d_model=configs.d_model
        self.use_revin=configs.use_revin

        self.trend_kernel=getattr(configs,"trend_kernel",25)
        self.trend_layers=getattr(configs,"trend_layers",1)

        if self.trend_kernel%2==0:
            self.trend_kernel+=1

        self.cycleQueue=RecurrentCycle(self.cycle_len,self.enc_in)

        self.trend_branch=TrendBranch(
            self.seq_len,
            self.pred_len,
            self.trend_kernel,
            self.trend_layers,
            self.d_model
        )

        assert self.model_type in ["linear","mlp"]

        if self.model_type=="linear":
            self.model=nn.Linear(self.seq_len,self.pred_len)
        elif self.model_type=="mlp":
            self.model=nn.Sequential(
                nn.Linear(self.seq_len,self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model,self.pred_len)
            )

    def forward(self,x,cycle_index):
        if self.use_revin:
            seq_mean=torch.mean(x,dim=1,keepdim=True)
            seq_var=torch.var(x,dim=1,keepdim=True)+1e-5
            x=(x-seq_mean)/torch.sqrt(seq_var)

        x=x-self.cycleQueue(cycle_index,self.seq_len)

        trend,trend_forecast=self.trend_branch(x)
        x_detrended=x-trend

        y=self.model(x_detrended.permute(0,2,1)).permute(0,2,1)
        y=y+trend_forecast
        y=y+self.cycleQueue((cycle_index+self.seq_len)%self.cycle_len,self.pred_len)

        if self.use_revin:
            y=y*torch.sqrt(seq_var)+seq_mean

        return y