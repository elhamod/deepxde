from .. import gradients as grad
from .. import geometry
from .. import icbc
from .. import data as d
import numpy as np
import torch
import pickle

#------
def Burgers_pde(x, y):
    dy_x = grad.jacobian(y, x, i=0, j=0)
    dy_t = grad.jacobian(y, x, i=0, j=1)
    dy_xx = grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

def Burgers_data():
    pde = Burgers_pde

    geom = geometry.Interval(-1, 1)
    timedomain = geometry.TimeDomain(0, 0.99)
    geomtime = geometry.GeometryXTime(geom, timedomain)

    bc = icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = icbc.IC(
        geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
    )
            
    data = d.TimePDE(
        geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160, train_distribution= "uniform"
    )
    return data

burgers_data = Burgers_data()



    
def Burgers_losses_criterion(args, loss_key='train_loss'):
    def losses_fn(x, y):
        if loss_key == 'train_loss':
            loss_fn = torch.nn.MSELoss()
            losses = burgers_data.losses(None, y, loss_fn, x, None, aux=None)
            if args.whichloss is None:
                losses = torch.stack(losses)
                return torch.sum(losses) #NOTE: It is not possible to say what the weights are for any point outside the trajectory. So, we just sum up
            return losses[args.task_name.index(args.whichloss)]
        else:
            f = Burgers_pde(x, y)
            return f.pow(2).mean()


    return losses_fn

def gen_testdata():
    data = np.load("/home/elhamod/projects/deepxde/examples/dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y