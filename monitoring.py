import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pickle


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads ) +1, lw=2, color="k" )
    plt.xticks(range(0 ,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.draw()
    plt.pause(0.01)


def plot_grad_flow2(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.ion()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)
    y=2


# def monitor_gradients_tensorboard():
#     with SummaryWriter(log_dir=log_dir, comment="GradTest", flush_secs=30) as writer:
#         # ... your learning loop
#         _limits = np.array([float(i) for i in range(len(gradmean))])
#         _num = len(gradmean)
#         writer.add_histogram_raw(tag=netname + "/abs_mean", min=0.0, max=0.3, num=_num,
#                                  sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(), bucket_limits=_limits,
#                                  bucket_counts=gradmean, global_step=global_step)
#         # where gradmean is np.abs(p.grad.clone().detach().cpu().numpy()).mean()
#         # _limits is the x axis, the layers
#         # and
#         _mean = {}
#         for i, name in enumerate(layers):
#             _mean[name] = gradmean[i]
#         writer.add_scalars(netname + "/abs_mean", _mean, global_step=global_step)

def register_gradient_velocities(model, velocities, step=0):
    pams = model.parameters()
    if step==0:
        for n,p in model.named_parameters():
            velocities[n] = []
            velocities[n].append(p.clone().detach().numpy())
    else:
        for n,p in model.named_parameters():
            val = p.clone().detach().numpy()
            previous = velocities[n][step-1]
            velocities[n].append(val - previous)
    with open('gradients.g', 'wb') as f:
        pickle.dump(velocities, f)
        f.close()