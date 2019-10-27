import numpy as np

def pix2met(pix_arr):
       """
       input: pix_array: shape (N, 2), each row is (x,y) pixel location
       output: met_array: shape (N, 2), each row is (x,y) meter location
       """
       hom_mat = np.array([[0.02104651, 0, 0], [0, -0.0236598, 13.74680446], [0, 0, 1]])
       pix_arr_z = np.hstack(pix_arr, np.zeros(pix_arr.shape[0])+1)
       met_loc_z = pix_arr_z.dot(hom_mat.T)
       met_loc = met_loc_z/met_loc_z[:,-1]
       return met_loc[:,:1]

def met2pix(met_arr):
       """
       input: met_array: shape (N, 2), each row is (x,y) meter location
       output: pix_array: shape (N, 2), each row is (x,y) pixel location
       """
       hom_mat = np.array([[0.02104651, 0, 0], [0, -0.0236598, 13.74680446], [0, 0, 1]])
       inv_mat = np.linalg.pinv(hom_mat)
       met_arr_z = np.hstack(met_arr, np.zeros(met_arr.shape[0])+1)
       pix_loc_z = met_arr_z.dot(inv_mat.T)
       pix_loc = pix_loc_z/pix_loc_z[:,-1]
       return pix_loc[:,:1]

def all_local_info(global_info, neigh_size = 1):
       """
       input: global_info: shape (grid_H, grid_W): object info in scene
              neigh_size: int: max grid distance S to be considered a neighbor 
       output: local: shape (grid_H, grid_W, (2*neigh_size+1)^2) of local information
       """
       enlarge = np.zeros([global_info.shape[0]+2*neigh_size, global_info.shape[1]+2*neigh_size])
       enlarge[neigh_size:(neigh_size+global_info.shape[0]), neigh_size:(neigh_size+global_info.shape[1])] = global_info
       local = np.zeros((global_info.shape[0], global_info.shape[1], (2*neigh_size+1)**2))
       for i in range(global_info.shape[0]):
              for j in range(global_info.shape[1]):
                     temp = [] 
                     for k in range(i, i+2*neigh_size+1):
                            for t in range(j, j+2*neigh_size+1):
                                   temp.append(enlarge[k,t])
                     local[i,j,:] = temp
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
