from torchvision.transforms import *
from posimagedataset import *
from model import *
from PIL import Image
import os
import numpy as np
import cv2

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDatasetExtended/images'
labelsfilepath = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDatasetExtended/joints.mat'

# imagepath = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDataset/images'
# labelsfilepath = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDataset/joints.mat'

# imagepath = '/mnt/disks/sdb/datasets/leeds/images'
# labelsfilepath = '/mnt/disks/sdb/datasets/leeds/joints.mat'


batchsize = 1


transforms = Compose([
    Resize((224,224)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize((0.5,),(0.5,))
])

#dataset/dataloader
dataset = PoseImageDataset(transforms, (224,224), imagepath, labelsfilepath)
dataloader_ = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
)
#model
model = PoseModel()
if os.path.exists('checkpoints/chpcurrent.chp'):
    d = torch.load('checkpoints/chpcurrent.chp')
    model.load_state_dict(d)
model.eval()

def drawlinesoverimage(points, image):
    for i in range(14):
        #right ankle => right knee
        cv2.line(image, (points[0][0], points[1][0]),(points[0][1], points[1][1]), (255,255,255),1)
        # right knee => right hip
        cv2.line(image, (points[0][1], points[1][1]), (points[0][2], points[1][2]), (255, 255, 255), 1)
        # left ankle => left knee
        cv2.line(image, (points[0][5], points[1][5]), (points[0][4], points[1][4]), (255, 255, 255), 1)
        # left knee => left hip
        cv2.line(image, (points[0][4], points[1][4]), (points[0][3], points[1][3]), (255, 255, 255), 1)
        #right hip => left hip
        cv2.line(image, (points[0][2], points[1][2]), (points[0][3], points[1][3]), (255, 255, 255), 1)
        # right wrist => right elbow
        cv2.line(image, (points[0][6], points[1][6]), (points[0][7], points[1][7]), (255, 255, 255), 1)
        # right elbow => right shoulder
        cv2.line(image, (points[0][7], points[1][7]), (points[0][8], points[1][8]), (255, 255, 255), 1)
        # left shoulder => left elbow
        cv2.line(image, (points[0][9], points[1][9]), (points[0][10], points[1][10]), (255, 255, 255), 1)
        # left elbow => left wrist
        cv2.line(image, (points[0][10], points[1][10]), (points[0][11], points[1][11]), (255, 255, 255), 1)
        # neck => head top
        cv2.line(image, (points[0][12], points[1][12]), (points[0][13], points[1][13]), (255, 255, 255), 1)
        # left shoulder => neck
        cv2.line(image, (points[0][9], points[1][9]), (points[0][12], points[1][12]), (255, 255, 255), 1)
        # neck => right shoulder
        cv2.line(image, (points[0][12], points[1][12]), (points[0][8], points[1][8]), (255, 255, 255), 1)
        # left shoulder => left hip
        cv2.line(image, (points[0][9], points[1][9]), (points[0][3], points[1][3]), (255, 255, 255), 1)
        # right shoulder => right hip
        cv2.line(image, (points[0][8], points[1][8]), (points[0][2], points[1][2]), (255, 255, 255), 1)

iter_ = iter(dataloader_)
for i in range(10):
    img, label, filename = next(iter_)
    res = model(img)
    res = res.view(-1,3,14).squeeze()

    filename = filename[0]
    orim = Image.open(os.path.join(imagepath,filename))
    resds = destandardize_label(res, orim)
    #displaying the image with overlayed featured points
    orimnp = np.array(orim)
    drawlinesoverimage(resds, orimnp)
    for j in range(14):
        cv2.circle(orimnp, (int(resds[0][j]),int(resds[1][j])), 1, (255, 0,0), 1, 0)
        # cv2.putText(orimnp, str(j), )
    #drawing the lines between feature points for better comprehension of pose estimation


    orimnp = cv2.resize(orimnp, None, fx=4, fy=4)
    cv2.imwrite('frame'+str(i)+'.jpg', orimnp)
    y = 2




