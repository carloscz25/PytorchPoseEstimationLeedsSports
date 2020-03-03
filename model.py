import torch


class PoseModel(torch.nn.Module):

    def __init__(self):
        super(PoseModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 11, 3, 1, 0, bias=True);
        self.mp1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(11,33,3, 1,0, bias=True)
        self.mp2 = torch.nn.MaxPool2d(2)

        self.conv3 = torch.nn.Conv2d(22, 66, 3, 1, 0, bias=True)
        self.mp3 = torch.nn.MaxPool2d(2)


        #11,22,33 => 25432
        self.linear1 = torch.nn.Linear(96228,2048, bias=True)
        self.linear2 = torch.nn.Linear(2048, 42, bias=True)



    from torch.nn.functional import leaky_relu
    def init_weights(self):
        self.conv1.weight.data.fill_(0)
        self.conv2.weight.data.fill_(0)
        # self.conv3.weight.data.fill_(0)
        self.linear1.weight.data.fill_(0)
        self.linear3.weight.data.fill_(0)

    def forward(self, image):
        im = self.conv1(image)
        im = self.mp1(torch.nn.functional.leaky_relu(im))

        im = self.conv2(im)
        im = self.mp2(torch.nn.functional.leaky_relu(im))

        # im = self.conv3(im)
        # im = self.mp3(torch.nn.functional.leaky_relu(im))




        fv = im.view(image.shape[0], -1)

        fv = torch.nn.functional.leaky_relu(self.linear1(fv))
        # fv = self.linear2(fv)
        fv = torch.nn.functional.leaky_relu(self.linear2(fv))
        # fv = self.linear4(fv)


        return fv




