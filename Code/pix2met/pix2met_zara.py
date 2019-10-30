import numpy as np
import torch 

zara_glob_info = torch.tensor(np.genfromtxt('Code/pix2met/obj_info.csv', delimiter=''))

def pix2met(pix_arr):
       """
       input: pix_array: shape (N, 2), each row is (x,y) pixel location
       output: met_array: shape (N, 2), each row is (x,y) meter location
       """
       hom_mat = torch.tensor([[0.02104651, 0, 0], [0, -0.0236598, 13.74680446], [0, 0, 1]])
       pix_arr_z = torch.cat((pix_arr, torch.zeros((pix_arr.shape[0],1))+1), dim =1)
       met_loc_z = pix_arr_z.mm(hom_mat.T)
       met_loc = met_loc_z/(met_loc_z[:,-1].reshape(-1,1))
       return met_loc[:,:2]

def met2pix(met_arr):
       """
       input: met_array: shape (N, 2), each row is (x,y) meter location
       output: pix_array: shape (N, 2), each row is (x,y) pixel location
       """
       hom_mat = torch.tensor([[0.02104651, 0, 0], [0, -0.0236598, 13.74680446], [0, 0, 1]])
       inv_mat = hom_mat.inverse()
       met_arr_z = torch.cat((met_arr, np.zeros((met_arr.shape[0],1))+1), dim=1)
       pix_loc_z = met_arr_z.mm(inv_mat.T)
       pix_loc = pix_loc_z/(pix_loc_z[:,-1].reshape(-1,1))
       return pix_loc[:,:2]


def all_local_info(global_info=zara_glob_info, neigh_size = 1):
       """
       input: global_info: shape (grid_H, grid_W): object info in scene
              neigh_size: int: max grid distance S to be considered a neighbor 
       output: local: shape (grid_H, grid_W, (2*neigh_size+1)^2) of local information
       """
       enlarge = torch.zeros((global_info.shape[0]+2*neigh_size, global_info.shape[1]+2*neigh_size))
       enlarge[neigh_size:(neigh_size+global_info.shape[0]), neigh_size:(neigh_size+global_info.shape[1])] = global_info
       local = torch.zeros((global_info.shape[0], global_info.shape[1], (2*neigh_size+1)**2))
       for i in range(global_info.shape[0]):
              for j in range(global_info.shape[1]):
                     temp = [] 
                     for k in range(i, i+2*neigh_size+1):
                            for t in range(j, j+2*neigh_size+1):
                                   temp.append(enlarge[k,t])
                     local[i,j,:] = torch.tensor(temp)
       return local

def local_info(curr_loc, all_local_info, g_size = 36):
       """
       input: curr_loc: shape (N,2), with each row a (x,y) meter location 
              global_info: shape (H, W): object info in scene
              all_local_info: array shape (grid_H, grid_W, (2*neigh_size+1)^2) of local information
       output: local_inf: shape (N, (2*neigh_size+1)^2), each row contains local info
       """ 
       pix_loc =  met2pix(curr_loc)
       pix_loc = pix_loc//g_size
       local_inf = all_local_info[pix_loc[:,0], pix_loc[:,1]]
       return local_inf
