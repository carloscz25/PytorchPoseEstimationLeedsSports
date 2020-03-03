import torch
from posimagedataset import PoseImageDataset
from  torchvision.transforms import *
from model import PoseModel
import os
from torch.utils.tensorboard import SummaryWriter




imagepath = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDatasetExtended/images'
labelsfilepath = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDatasetExtended/joints.mat'

batchsize = 64
numworkers = 2
epochs = 5

transforms = Compose([
    Resize((224,224)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize((0.5,),(0.5,))
])

# transforms = Compose([
#     Resize((224,224)),
#     ToTensor(),
#     Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# ])

#dataset/dataloader
dataset = PoseImageDataset(transforms, (224,224), imagepath, labelsfilepath)
dataloader_ = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        num_workers=numworkers

    )
#model
model = PoseModel()

if os.path.exists('checkpoints/chpcurrent.chp'):
    d = torch.load('checkpoints/chpcurrent.chp')
    try:
        model.load_state_dict(d)
    except BaseException:
        y=2

for name, p in model.named_parameters():
    pass
    y = 2
    # if 'bias' in name:
    #     p.requires_grad = False
    # if name == 'conv1.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # if name == 'conv2.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # if name == 'conv3.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # if name == 'conv4.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # if name == 'linear1.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # if name == 'linear2.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    # if name == 'linear3.weight':
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

model.train()

# parameters = list(model.linear1.parameters()) + list(model.linear2.parameters())
parameters = model.parameters()
loss = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(parameters, lr=0.00001)
writer = SummaryWriter()




from NetScreener import *

gs = GradientsScreener(model, writer)

for i in range(epochs):

    for i, (images,labels, filenames) in enumerate(dataloader_):
        optimizer.zero_grad()

        res = model(images)
        labels = labels.view(-1, 42)
        lossvalue = loss(res, labels)
        lossvalue.backward()


        # register_gradient_velocities(model, velocities, i)



        optimizer.step()

        # plot_grad_flow2(model.named_parameters())

        y = 2

        print(lossvalue.item())

        writer.add_histogram("activations", res, i, bins="auto")
        writer.add_scalar('Loss Value', lossvalue.item(), i)

        # if lossvalue.item() < 2000:
        #     gs.monitor(i)

        if i > 0:
            if (i%100)==0:
                print('checkpoint guardado' + str(i))
                torch.save(model.state_dict(), 'checkpoints/chpcurrent.chp');



writer.close()