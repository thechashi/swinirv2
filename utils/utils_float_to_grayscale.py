import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.act(x)
        return x

class NeighborDistanceCalc:
    def __init__(self, kernel_size, img_matrix):
        self.kernel_size = kernel_size
        self.kernel_distance_dict = []
        self.img_matrix = img_matrix

    def calculate_positions(self,pos_h, pos_w, height, width):
        list_of_possible_poss = []
        '''
        if pos_h == 0 or pos_w == 0 or pos_h + 1 == height or pos_w + 1 == width:

            if not pos_w + 1 == width:
                # top
                list_of_possible_poss.append((pos_h, pos_w + 1))

            if not (pos_h == 0 or pos_w == 0):
                # top left
                list_of_possible_poss.append((pos_h - 1, pos_w - 1))

            if not (pos_h == 0 or pos_w + 1 == width):
                # top right
                list_of_possible_poss.append((pos_h - 1, pos_w + 1))

            if not pos_h == 0:
                # left
                list_of_possible_poss.append((pos_h - 1, pos_w))

            if not pos_w == 0:
                # bottom
                list_of_possible_poss.append((pos_h, pos_w - 1))

            if not (pos_h + 1 == height or pos_w == 0):
                # bottom left
                list_of_possible_poss.append((pos_h + 1, pos_w - 1))

            if not (pos_h + 1 == height or pos_w + 1 == width):
                # bottom right
                list_of_possible_poss.append((pos_h + 1, pos_w + 1))

            if not pos_h + 1 == height:
                # right
                list_of_possible_poss.append((pos_h + 1, pos_w))
        '''

        if not (pos_h == 0 or pos_w == 0 or pos_h + 1 == height or pos_w + 1 == width):
            list_of_possible_poss = [(pos_h, pos_w + 1),
                                         (pos_h - 1, pos_w - 1),
                                         (pos_h - 1, pos_w + 1),
                                         (pos_h - 1, pos_w),
                                         (pos_h, pos_w + 1),
                                         (pos_h + 1, pos_w - 1),
                                         (pos_h + 1, pos_w + 1),
                                         (pos_h + 1, pos_w)]
        return list_of_possible_poss

    def process_distance_vectors_for_kernel(self, img_matrix, kernel_pos):
        height, width = img_matrix.shape
        neighboring_dist_dict = []
        pos_index = 0
        for pos_h in range(height):
            for pos_w in range(width):
                neighboring_dist_list = []
                neighboring_pos_list = self.calculate_positions(pos_h, pos_w, height, width)
                if not neighboring_pos_list:
                    continue
                for neigh_pos_element in neighboring_pos_list:
                    neigh_pos_h, neigh_pos_w = neigh_pos_element
                    dist = torch.abs(img_matrix[pos_h, pos_w] - img_matrix[neigh_pos_h, neigh_pos_w])
                    neighboring_dist_list.append(dist)
                neighboring_dist_dict.insert(pos_index, neighboring_dist_list)
                pos_index += 0
                del neighboring_dist_list, neighboring_pos_list
        self.kernel_distance_dict.insert(kernel_pos, neighboring_dist_dict)

    def image_proc(self):
        kernel_tensor = F.unfold(self.img_matrix, kernel_size=self.kernel_size, stride=self.kernel_size).transpose(2, 1)
        kernel_tensor = F.fold(kernel_tensor, output_size=(self.kernel_size, self.kernel_size),
                               kernel_size=(1, 1))
        batch, kernel_mats, height, width = kernel_tensor.size()
        for kernel_pos in range(kernel_mats):
            self.process_distance_vectors_for_kernel(kernel_tensor[0, kernel_pos, :, :], kernel_pos)

        self.kernel_distance_dict = torch.tensor(self.kernel_distance_dict, 
                                                 dtype=torch.float,
                                                 requires_grad=True)

        return self.kernel_distance_dict


class DistanceLoss(nn.Module):
    """"""
    def __init__(self, kernel_size, loss_weight=1):
        """

        :param kernel_size: This is the size of the sliding window
        :param loss_weight: The weight for the loss
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.loss_weight = loss_weight
        #self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, sr_tensor, hr_tensor):
        """

        :param sr_tensor: sr image in tensor
        :param hr_tensor: hr image in tensor
        :return: mse loss
        """
        neigh_obj = NeighborDistanceCalc(self.kernel_size, sr_tensor)
        sr_distance = neigh_obj.image_proc()

        neigh_obj = NeighborDistanceCalc(self.kernel_size, hr_tensor)
        hr_distance = neigh_obj.image_proc()

        del neigh_obj

        #l2_loss = self.l2_loss(sr_distance, hr_distance) * self.loss_weight
        l1_loss = self.l1_loss(sr_distance, hr_distance) * self.loss_weight
        return l1_loss


'''
Test to check if we are getting loss output
if __name__ == '__main__':
    sr_image = torch.randn(1, 1, 256, 256)
    hr_image = torch.randn(1, 1, 256, 256)

    kernel_size = 3

    dist_obj = DistanceLoss(kernel_size)
    l2_loss = dist_obj(sr_image, hr_image)
    print(l2_loss.item())
'''






def train(trainset_path, epochs=10, loss_kernel_size=3 ):
    print("Starting training...")
    # read file names
    net = Net()

    loss_kernel_size = 3
    criterion = DistanceLoss(loss_kernel_size)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu") 
    

            
    for epoch in range(epochs):  
        print("Epoch: {}".format(epoch))
        running_loss = 0.0
        min_loss = math.inf
        net.train(True)
        net = net.to(device)
        i = 0
        for filename in os.listdir(trainset_path):
            if '.npz' in filename:
                file_path = os.path.join(trainset_path, filename)
                image = np.load(file_path)
                image = image.f.arr_0
                image = torch.tensor(image).reshape((1,1,256,256))
                image = (image/300890.0)*3
                print(image.shape)
                image = image.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                prediction = net(image)
                prediction = prediction * 6.0
                loss = criterion(prediction, image)
                print(loss.item())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                i += 1
                if i % 20 == 1:    # print every 20 mini-batches
                    print(f'[{epoch + 1}, {i:5d}] loss: {running_loss/i :.6f}')
                if min_loss > (running_loss/i):
                    min_loss = (running_loss/i)
                    print(f'Saving model. Loss decreased. [{epoch + 1}, {i:5d}] loss: {running_loss/i :.6f}')
                    torch.save(net.state_dict(), "f2g_trained.pt")
        running_loss = 0.0
                
            
    print('Finished Training')
    
if __name__ == "__main__":
    train('../trainsets/earth1_samples/earth1_100', epochs=10)

