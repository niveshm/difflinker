import subprocess
from rdkit import Chem
import numpy as np
from tqdm import tqdm
from src import const
from src.datasets import collate_with_fragment_edges, create_templates_for_linker_generation, get_dataloader, parse_molecule
from src.egnn import Dynamics
from src.edm import EDM
import os
import torch
import io

from src.lightning import DDPM
from src.visualizer import save_xyz_file

# from src.lightning import DDPM

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

def get_model():
    hyperparams = {'in_node_nf': 9, 'n_dims': 3, 'context_node_nf': 1, 'hidden_nf': 128, 'activation': 'silu', 'tanh': False, 'n_layers': 6, 'attention': False, 'norm_constant': 1e-06, 'inv_sublayers': 2, 'sin_embedding': False, 'normalization_factor': 100, 'aggregation_method': 'sum', 'diffusion_steps': 500, 'diffusion_noise_schedule': 'polynomial_2', 'diffusion_noise_precision': 1e-05, 'diffusion_loss_type': 'l2', 'normalize_factors': [1, 4, 10], 'include_charges': False, 'model': 'egnn_dynamics', 'data_path': '/home/igashov/work/diffusion_linker_data/e3_ddpm_linker_design/datasets_v2', 'train_data_prefix': 'geom_multifrag_train', 'val_data_prefix': 'geom_multifrag_val', 'batch_size': 64, 'lr': 0.0002, 'torch_device': 'cuda:0', 'test_epochs': 20, 'n_stability_samples': 10, 'normalization': 'batch_norm', 'log_iterations': None, 'samples_dir': '/home/igashov/work/diffusion_linker_data/e3_ddpm_linker_design/logs/samples/geom0_igashov_GEOM_6L_noanch_bs64_date18-08_time19-00-59.926896', 'data_augmentation': False, 'center_of_mass': 'fragments', 'inpainting': False, 'anchors_context': False}
    # n_dims = 3
    # in_node_nf = 9 #const.GEOM_NUMBER_OF_ATOM_TYPES + const.NUMBER_OF_ATOM_TYPES

    dynamics = Dynamics(n_dims=hyperparams['n_dims'], in_node_nf=hyperparams['in_node_nf'], context_node_nf=hyperparams['context_node_nf'], hidden_nf=hyperparams['hidden_nf'], n_layers=hyperparams['n_layers'])

    edm = EDM(n_dims=3, in_node_nf=hyperparams['in_node_nf'], loss_type=hyperparams['diffusion_loss_type'], timesteps=hyperparams['diffusion_steps'], noise_schedule=hyperparams['diffusion_noise_schedule'], noise_precision=hyperparams['diffusion_noise_precision'], dynamics=dynamics, norm_values=hyperparams['normalize_factors'])

    return edm
    

def main(input_path, model_path, output_dir, n_samples, n_steps, linker_size, anchors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    mol = read_molecule(input_path)
    mol = Chem.RemoveAllHs(mol)
    name = '.'.join(input_path.split('/')[-1].split('.')[:-1])
    print(mol.GetAtoms())

    def sample_fn(_data):
        return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size

    # for atom in mol.GetAtoms():
    #     print(atom.GetSymbol(), end=" ")
    # print(name)

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

    dataloader = get_dataloader(dataset, batch_size=n_samples, collate_fn=collate_with_fragment_edges)

    # ddpm = DDPM.load_from_checkpoint(model, map_location=device, strict=False).eval().to(device)
    # ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    # breakpoint()
    # breakpoint()

    model = get_model()

    # breakpoint()
    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    # with open(model, 'rb') as f:
    #     buffer = io.BytesIO(f.read())
    
    # model.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))


    
    # print()

    print('Sampling...')
    # breakpoint()
    for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_size = len(data['positions'])
        # breakpoint()
        linker_sizes = sample_fn(data)

        template_data = create_templates_for_linker_generation(data, linker_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        anchors = template_data['anchors']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']

        context = fragment_mask
        # breakpoint()

        res = model.forward(x, h, node_mask, fragment_mask, linker_mask, edge_mask, context)

        # breakpoint()

        chain = model.sample_chain(x, h, node_mask, fragment_mask, linker_mask, edge_mask, context)

        x = chain[0][:, :, :3]
        h = chain[0][:, :, 3:]
        com_mask = data['fragment_mask']
        pos_masked = data['positions'] * com_mask
        N = com_mask.sum(1, keepdims=True)
        mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
        x = x + mean * node_mask

        offset_idx = batch_i * n_samples
        names = [f'output_{offset_idx+i}_{name}' for i in range(batch_size)]
        save_xyz_file(output_dir, h, x, node_mask, names=names, is_geom=True, suffix='')
        for i in range(batch_size):
            out_xyz = f'{output_dir}/output_{offset_idx+i}_{name}_.xyz'
            out_sdf = f'{output_dir}/output_{offset_idx+i}_{name}_.sdf'
            subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)
        

    




    # breakpoint()


if __name__ == '__main__':
    main('./sample_data/5ou2_fragments_input.sdf', './models/geom_diffliner_state_dict.pt', 'output', 5, 1, 3, None)