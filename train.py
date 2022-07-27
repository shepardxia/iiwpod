import torch
import torchvision
import cv2
from create_model import iiwpod
from src.data_generation import ALPRDataGenerator
import numpy as np
from src.utils import image_files_from_folder
from src.label import readShapes, Shape
from os.path import isfile, isdir, splitext
from tqdm import tqdm
from src.loss import poly_loss, gen_loss

def load_network(version=0, weights=None):
    if weights is None:
        return iiwpod().to(dtype=torch.float32)
    else:
        return iiwpod()



model = load_network()

Files = image_files_from_folder('./dataset')
	
#
#  Defines size of "fake" tiny LP annotation, used when no LP is present
#				
fakepts = np.array([[0.5, 0.5001, 0.5001, 0.5], [0.5, 0.5, 0.5001, 0.5001]])
fakeshape = Shape(fakepts)
Data = []
ann_files = 0
for file in Files:
	labfile = splitext(file)[0] + '.txt'
	if isfile(labfile):
		ann_files += 1
		L = readShapes(labfile)
		I = cv2.imread(file)
		if len(L) > 0:
			Data.append([I, L])
	else:
		#
		#  Appends a "fake"  plate to images without any annotation
		#
		I = cv2.imread(file)
		Data.append(  [I, [fakeshape] ]  )

data_gen = ALPRDataGenerator(Data, batch_size=1)
training_loader = torch.utils.data.DataLoader(data_gen, batch_size = 2, shuffle = True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
maxEpoch = 100

for epoch in range(maxEpoch):

	for batch, (x, y) in enumerate(training_loader):
		
		x = x.type(torch.float32)
		pred = model(x)
		loss = gen_loss(y, pred)
		optimizer.zero_grad()
		loss.mean().backward()
		if torch.isnan(loss.mean()):
			print('caught')
		else:
			optimizer.step()
			print(loss.mean())

	print(loss)


torch.save(model, './model.pth')
