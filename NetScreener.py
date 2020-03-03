import numpy as np


class GradientsScreener(object):

    model = None
    monitor_every_steps = 5
    current_step = None
    writer = None


    gradients = {}
    weights = {}


    def __init__(self, model, writer, monitor_every_steps = 5):
        super().__init__()
        self.model = model
        self. monitor_every_steps = monitor_every_steps
        self.writer = writer


    def monitor(self, step):
        start = True
        for n, p in self.model.named_parameters():
            grad = p.grad.clone().detach().numpy()
            weight = p.clone().detach().numpy()
            if step == 0:
                self.gradients[n] = grad
                self.weights[n] = weight

            else:
                if (step > self.monitor_every_steps):
                    ponder = self.monitor_every_steps-1
                else:
                    ponder = step
                #gradients
                valgrad = self.gradients[n]
                newvalgrad = ((valgrad * ponder) + grad)/(ponder+1)
                self.gradients[n] = newvalgrad
                # weights
                valweight = self.weights[n]
                newvalweight = ((valweight * ponder) + weight) / (ponder + 1)
                self.weights[n] = newvalweight

        #writting to tensorboard
        for k in self.gradients.keys():
            self.writer.add_scalars('Gradients ' + k, {'Max':np.max(self.gradients[k]), 'Min':np.min(self.gradients[k])}, step)



        #printing some stats
        # print("Gradients")
        # for d in self.gradients.keys():
        #     print(d + ':max='+str(np.max(self.values[d]))+':min='+str(np.min(self.values[d])))
        # print("Weights")
        # for d in self.weights.keys():
        #     print(d + ':max=' + str(np.max(self.values[d])) + ':min=' + str(np.min(self.values[d])))






