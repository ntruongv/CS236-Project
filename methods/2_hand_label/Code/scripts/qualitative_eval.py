import argparse
import os
import torch
import sys
import numpy as numpy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models_w_local_context import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

codepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codepath)
from PIL import Image

from pix2met import pix2met_zara

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=200, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--local_neigh_size', default = 1, type =int) #NHI: local info neighbor size

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        local_neigh_size = args.local_neigh_size) # NHI: local neighbor size default is 1 
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def qualitative_eval(args, loader, generator, num_samples, processed_local_info, img, save_path):
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end) = batch

            
            fake_traj = []
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end, processed_local_info #NHI
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                
                fake_traj.append(pred_traj_fake)

            all_fake_traj = torch.stack(fake_traj) #shape sample, seq length, batch, 2
            mean_fake_traj = all_fake_traj.mean(axis=0)
            std_fake_traj = all_fake_traj.std(axis=0)
            
            for person_id in range(mean_fake_traj.shape[1]):
                input_traj_cpu = obs_traj[:,person_id,:].data.cpu()
                input_traj_met = pix2met_zara.met2pix_cpu(input_traj_cpu)
                mean_pred_traj_cpu = mean_fake_traj[:,person_id,:].data.cpu()
                mean_pred_traj_met = pix2met_zara.met2pix_cpu(mean_pred_traj_cpu)
                std_fake_traj_cpu = std_fake_traj[:,person_id,:].data.cpu()
                plt.imshow(img)
                plt.scatter(input_traj_met[:,0], input_traj_met[:,1], c='b')
                plt.scatter(mean_pred_traj_met[:,0], mean_pred_traj_met[:,1], c='y')
                plt.savefig(os.path.join(save_path, str(i) + "_" + str(person_id) + '.png'))
                plt.close()

        return None


def main(args):
    #processed_local_info = pix2met_zara.all_local_info(neigh_size = args.local_neigh_size)  #NHI: process local info now
    filepath = os.path.join(codepath, "vgg", "frame_1.png")
    img = Image.open(filepath) #NHI: graph local info
    processed_local_info = pix2met_zara.all_local_info(neigh_size = args.local_neigh_size)  #NHI: process local info now  
    save_path = os.path.join(codepath, "..", "qualitative_results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.isdir(args.model_path):
        checkpoint = torch.load(os.path.join(args.model_path, "checkpoint_with_model.pt"))
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        qualitative_eval(_args, loader, generator, args.num_samples, processed_local_info, img, save_path) #NHI


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
