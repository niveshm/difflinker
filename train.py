import torch

from src import utils
from src.datasets import ZincDataset, collate, collate_with_fragment_edges, get_dataloader
from src.edm import EDM
from src.egnn import Dynamics
from tqdm import tqdm

def get_model(hyperparams):
    
    # n_dims = 3
    # in_node_nf = 9 #const.GEOM_NUMBER_OF_ATOM_TYPES + const.NUMBER_OF_ATOM_TYPES

    dynamics = Dynamics(n_dims=hyperparams['n_dims'], in_node_nf=hyperparams['in_node_nf'], context_node_nf=hyperparams['context_node_nf'], hidden_nf=hyperparams['hidden_nf'], n_layers=hyperparams['n_layers'])

    edm = EDM(n_dims=3, in_node_nf=hyperparams['in_node_nf'], loss_type=hyperparams['diffusion_loss_type'], timesteps=hyperparams['diffusion_steps'], noise_schedule=hyperparams['diffusion_noise_schedule'], noise_precision=hyperparams['diffusion_noise_precision'], dynamics=dynamics, norm_values=hyperparams['normalize_factors'])

    return edm

def main(hyperparams):
    # dataset = torch.load('dataset/geom_multifrag_train.pt', map_location='cpu')

    train_dataset = ZincDataset('dataset', 'geom_multifrag_train', 'cpu')
    # val_dataset = ZincDataset('dataset/geom_multifrag_val.pt', 'geom_multifrag_val', 'cpu')

    train_dataloader = get_dataloader(train_dataset, batch_size=hyperparams['batch_size'], collate_fn=collate)
    model = get_model(hyperparams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr'], amsgrad=True, weight_decay=1e-12)


    #train loop
    for i, data in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        x = data['positions']
        h = data['one_hot']
        node_mask = data['atom_mask']
        edge_mask = data['edge_mask']
        anchors = data['anchors']
        fragment_mask = data['fragment_mask']
        linker_mask = data['linker_mask']
        context = fragment_mask
        if hyperparams['anchors_context']:
            context = torch.cat([anchors, fragment_mask], dim=-1)
        else:
            context = fragment_mask
        center_of_mass_mask = fragment_mask if hyperparams['center_of_mass'] == 'fragments' else anchors

        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

        if hyperparams['data_augmentation']:
            x = utils.random_rotation(x)



        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 =  model.forward(x, h, node_mask, fragment_mask, linker_mask, edge_mask, context)

        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if hyperparams['diffusion_loss_type'] == 'l2':
            loss = l2_loss
        elif hyperparams['diffusion_loss_type'] == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(hyperparams['diffusion_loss_type'])

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Iteration {i}, loss: {loss.item()}')
        
        break
        


if __name__ == '__main__':
    hyperparams = {'in_node_nf': 9, 'n_dims': 3, 'context_node_nf': 1, 'hidden_nf': 128, 'activation': 'silu', 'tanh': False, 'n_layers': 6, 'attention': False, 'norm_constant': 1e-06, 'inv_sublayers': 2, 'sin_embedding': False, 'normalization_factor': 100, 'aggregation_method': 'sum', 'diffusion_steps': 500, 'diffusion_noise_schedule': 'polynomial_2', 'diffusion_noise_precision': 1e-05, 'diffusion_loss_type': 'l2', 'normalize_factors': [1, 4, 10], 'include_charges': False, 'model': 'egnn_dynamics', 'data_path': '/home/igashov/work/diffusion_linker_data/e3_ddpm_linker_design/datasets_v2', 'train_data_prefix': 'geom_multifrag_train', 'val_data_prefix': 'geom_multifrag_val', 'batch_size': 64, 'lr': 0.0002, 'torch_device': 'cuda:0', 'test_epochs': 20, 'n_stability_samples': 10, 'normalization': 'batch_norm', 'log_iterations': None, 'samples_dir': '/home/igashov/work/diffusion_linker_data/e3_ddpm_linker_design/logs/samples/geom0_igashov_GEOM_6L_noanch_bs64_date18-08_time19-00-59.926896', 'data_augmentation': False, 'center_of_mass': 'fragments', 'inpainting': False, 'anchors_context': False}
    main(hyperparams)
        