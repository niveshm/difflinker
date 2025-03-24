import os
import subprocess
from tqdm import tqdm
from src import const
from src.datasets import ZincDataset, collate, create_templates_for_linker_generation, get_dataloader, parse_molecule
import numpy as np
import torch

from src.edm import EDM
from src.egnn import Dynamics
from src.visualizer import save_xyz_file
from train import preprocess_data
from rdkit import Chem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_molecule(path):
    if path.endswith('.sdf'):
        return Chem.SDMolSupplier(path, sanitize=False, removeHs=True)[0]
    elif path.endswith('.pdb'):
        return Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    else:
        raise ValueError('Unsupported file format')

def get_model(hyperparams):
    
    # n_dims = 3
    # in_node_nf = 9 #const.GEOM_NUMBER_OF_ATOM_TYPES + const.NUMBER_OF_ATOM_TYPES

    dynamics = Dynamics(n_dims=hyperparams['n_dims'], in_node_nf=hyperparams['in_node_nf'], context_node_nf=hyperparams['context_node_nf'], hidden_nf=hyperparams['hidden_nf'], n_layers=hyperparams['n_layers'], device=device)

    edm = EDM(n_dims=3, in_node_nf=hyperparams['in_node_nf'], loss_type=hyperparams['diffusion_loss_type'], timesteps=hyperparams['diffusion_steps'], noise_schedule=hyperparams['diffusion_noise_schedule'], noise_precision=hyperparams['diffusion_noise_precision'], dynamics=dynamics, norm_values=hyperparams['normalize_factors'])

    return edm


def main(hyperparams, output_dir, input_path, linker_size, anchors, n_samples):

    print(device)
    os.makedirs(output_dir, exist_ok=True)

    mol = read_molecule(input_path)
    mol = Chem.RemoveAllHs(mol)
    name = '.'.join(input_path.split('/')[-1].split('.')[:-1])
    print(mol.GetAtoms())

    def sample_fn(_data):
        return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size
    

    positions, one_hot, charges = parse_molecule(mol, True)
    fragment_mask = np.ones_like(charges)
    linker_mask = np.zeros_like(charges)
    anchor_flags = np.zeros_like(charges)
    if anchors is not None:
        for anchor in anchors.split(','):
            anchor_flags[int(anchor.strip()) - 1] = 1
    

    dataset = [{
        "uuid": 0,
        "name": 0,
        "positions": torch.tensor(positions, device=device, dtype=const.TORCH_FLOAT),
        "one_hot": torch.tensor(one_hot, device=device, dtype=const.TORCH_FLOAT),
        'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
        'anchors': torch.tensor(anchor_flags, dtype=const.TORCH_FLOAT, device=device),
        'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
        'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
        'num_atoms': len(positions),
    }] * n_samples

    dataloader = get_dataloader(dataset, batch_size=n_samples, collate_fn=collate)

    model = get_model(hyperparams)
    model.load_state_dict(torch.load('models/geom_diffliner_state_dict.pt', map_location='cpu', weights_only=True))
    model = model.to(device)
    breakpoint()


    # test_dataset = ZincDataset('dataset', 'geom_multifrag_test', device)

    # test_dataset = [test_dataset[0]]*3
    # test_dataloader = get_dataloader(test_dataset, batch_size=hyperparams['batch_size'], collate_fn=collate, shuffle=False)

    # model = get_model(hyperparams)
    # model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
    # model = model.to(device)
    # breakpoint()

    # val_losses = []
    model.eval()
    for i, data in tqdm(enumerate(dataloader), desc='Validation', total=len(dataloader)):
        linker_sizes = sample_fn(data)

        template_data = create_templates_for_linker_generation(data, linker_sizes)

        x, h, node_mask, fragment_mask, linker_mask, edge_mask, context = preprocess_data(template_data, hyperparams)
        # x = template_data['positions']
        # node_mask = template_data['atom_mask']
        # edge_mask = template_data['edge_mask']
        # h = template_data['one_hot']
        # anchors = template_data['anchors']
        # fragment_mask = template_data['fragment_mask']
        # linker_mask = template_data['linker_mask']
        # context = fragment_mask

        chain  = model.sample_chain(x, h, node_mask, fragment_mask, linker_mask, edge_mask, context)
        # breakpoint()

        x = chain[0][:, :, :3]
        h = chain[0][:, :, 3:]
        com_mask = data['fragment_mask']
        pos_masked = data['positions'] * com_mask
        N = com_mask.sum(1, keepdims=True)
        mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
        x = x + mean * node_mask
        batch_size = len(data['positions'])

        offset_idx = i * 3
        names = [f'output_{offset_idx+i}_{name}' for i in range(batch_size)]
        save_xyz_file(output_dir, h, x, node_mask, names=names, is_geom=True, suffix='')
        for i in range(batch_size):
            out_xyz = f'{output_dir}/output_{offset_idx+i}_{name}_.xyz'
            out_sdf = f'{output_dir}/output_{offset_idx+i}_{name}_.sdf'
            subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)

        
        # delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 =  model.forward(x, h, node_mask, fragment_mask, linker_mask, edge_mask, context)

        # vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        # if hyperparams['diffusion_loss_type'] == 'l2':
        #     loss = l2_loss
        # elif hyperparams['diffusion_loss_type'] == 'vlb':
        #     loss = vlb_loss
        # else:
        #     raise NotImplementedError(hyperparams['diffusion_loss_type'])

        # val_losses.append(loss.item())
    
    # print(f'Epoch, test loss: {sum(val_losses)/len(val_losses)}')


if __name__ == '__main__':
    hyperparams = {'in_node_nf': 9, 'n_dims': 3, 'context_node_nf': 1, 'hidden_nf': 128, 'activation': 'silu', 'tanh': False, 'n_layers': 6, 'attention': False, 'norm_constant': 1e-06, 'inv_sublayers': 2, 'sin_embedding': False, 'normalization_factor': 100, 'aggregation_method': 'sum', 'diffusion_steps': 500, 'diffusion_noise_schedule': 'polynomial_2', 'diffusion_noise_precision': 1e-05, 'diffusion_loss_type': 'l2', 'normalize_factors': [1, 4, 10], 'include_charges': False, 'model': 'egnn_dynamics', 'data_path': '/home/igashov/work/diffusion_linker_data/e3_ddpm_linker_design/datasets_v2', 'train_data_prefix': 'geom_multifrag_train', 'val_data_prefix': 'geom_multifrag_val', 'batch_size': 64, 'lr': 0.0002, 'torch_device': 'cuda:0', 'test_epochs': 20, 'n_stability_samples': 10, 'normalization': 'batch_norm', 'log_iterations': None, 'samples_dir': '/home/igashov/work/diffusion_linker_data/e3_ddpm_linker_design/logs/samples/geom0_igashov_GEOM_6L_noanch_bs64_date18-08_time19-00-59.926896', 'data_augmentation': False, 'center_of_mass': 'fragments', 'inpainting': False, 'anchors_context': False}
    input_path = './sample_data/5ou2_fragments_input.sdf'
    # model_path
    output_dir = 'output_dir'
    linker_size = 3
    anchors = None
    num_samples = 5

    main(hyperparams, output_dir, input_path, linker_size, anchors, num_samples)