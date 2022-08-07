import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from fastMRI.data import transforms
from training_utils.linear_mapping import R2star_B0_real_S0_complex_mapping, R2star_S0_mapping_from_ksp


class Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, acceleration, sequence, TEs, train_resolution=None, use_seed=True):
        """
        :param mask_func: common.subsample.MaskFunc, A mask sampling function
        :param sequence: [string] "ME_GRE" or "FUTURE_SEQUENCES" depending on which sequence the mapping is performed with.
        :TEs: [tuple] Echo times in "ME_GRE" sequence.
        :param train_resolution: Resolution of k-space measurement. 
        :param use_seed: If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if sequence not in ('MEGRE', 'FUTURE_SEQUENCES'):
            raise ValueError('sequence should be either "MEGRE" or "FUTURE_SEQUENCES"')
        self.mask_func = mask_func
        self.acceleration = acceleration
        self.train_resolution = train_resolution
        self.sequence = sequence
        self.use_seed = use_seed
        self.TEs = np.float32(TEs)

    def __call__(self, kspace, sense, mask_brain, fname, fname_full, slice, n_slices=1):
        fixed_ksp_patt = False  # Option for experiments with fixed (identical) ksp pattern for TEs

        masked_kspace = torch.zeros_like(kspace)

        # The subsampling masks were generaged with the RIM image reconstruction code, not in the qRIM. We just load them here
        # so it's ensured we use the same set of subsampling patterns.
        # Though it's also easy to directly generate the subsampling masks here as well, e.g. with the commented code snippet (try..except..) below.
        subsampling_mask_TE0 = None
        subsampling_mask_TE1 = None
        subsampling_mask_TE2 = None
        subsampling_mask_TE3 = None
        with h5py.File(str(fname_full), 'r') as data:
            if 'subsampling_mask_acce_' + str(self.acceleration) + '_TE_0' in data:
                subsampling_mask_TE0 = data['subsampling_mask_acce_' + str(self.acceleration) + '_TE_0']
                subsampling_mask_TE0 = torch.from_numpy(np.float32(subsampling_mask_TE0))
            if 'subsampling_mask_acce_' + str(self.acceleration) + '_TE_1' in data:
                subsampling_mask_TE1 = data['subsampling_mask_acce_' + str(self.acceleration) + '_TE_1']
                subsampling_mask_TE1 = torch.from_numpy(np.float32(subsampling_mask_TE1))
            if 'subsampling_mask_acce_' + str(self.acceleration) + '_TE_2' in data:
                subsampling_mask_TE2 = data['subsampling_mask_acce_' + str(self.acceleration) + '_TE_2']
                subsampling_mask_TE2 = torch.from_numpy(np.float32(subsampling_mask_TE2))
            if 'subsampling_mask_acce_' + str(self.acceleration) + '_TE_3' in data:
                subsampling_mask_TE3 = data['subsampling_mask_acce_' + str(self.acceleration) + '_TE_3']
                subsampling_mask_TE3 = torch.from_numpy(np.float32(subsampling_mask_TE3))

            if subsampling_mask_TE0 is not None and subsampling_mask_TE1 is not None and subsampling_mask_TE2 is not None and subsampling_mask_TE3 is not None:
                if fixed_ksp_patt:
                    subsampling_mask = torch.stack(
                        (subsampling_mask_TE0, subsampling_mask_TE0, subsampling_mask_TE0, subsampling_mask_TE0), 0)
                else:
                    subsampling_mask = torch.stack(
                        (subsampling_mask_TE0, subsampling_mask_TE1, subsampling_mask_TE2, subsampling_mask_TE3), 0)
            elif 'subsampling_mask_acce_' + str(self.acceleration) in data:
                subsampling_mask = data['subsampling_mask_acce_' + str(self.acceleration)]
                subsampling_mask = torch.from_numpy(np.float32(subsampling_mask))
            else:
                print('except in loading subsampling mask: ' + str(fname_full))
                # seeds = []
                # subsampling_mask = torch.zeros(kspace.shape[0], kspace.shape[-3],
                #                                kspace.shape[-2])  # nr_of_echoes, size of phase, size of slices
                # for echo_nr in range(kspace.shape[0]):
                #     fname_for_seed = fname + '_' + str(echo_nr)
                #     seed = None if not self.use_seed else tuple(map(ord, fname_for_seed))
                #     seeds.append(seed)
                #     np.random.seed(seed)
                #     if self.mask_func is not None:
                #         _, sampling_mask_echo = transforms.apply_mask_custom(kspace[:, ...], self.mask_func, seed)
                #     else:
                #         sampling_mask_echo = kspace != 0
                #         sampling_mask_echo = sampling_mask_echo.to(kspace)[..., :1, :, :1]
                #         sampling_mask_echo = sampling_mask_echo[:1, ...]
                #
                #     subsampling_mask[echo_nr, ...] = sampling_mask_echo.squeeze()

                subsampling_mask = torch.ones([kspace.shape[2], kspace.shape[3]]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).byte()

            # Apply mask
            # masked_kspace = kspace*subsampling_mask.unsqueeze(1).unsqueeze(-1)
            masked_kspace = kspace*subsampling_mask

            # load RIM recon images (for initialization of qRIM maps (at time step 0) )
            # reconimgs = torch.zeros(kspace.shape[0], kspace.shape[2], kspace.shape[3], kspace.shape[4])
            # for echo_nr in range(kspace.shape[0]):
            #     reconimgs_filename = str(fname_full.parents[2])+"/recon_diffpatt_"+str(self.acceleration)+"_ssim/rim_recon/"+str(echo_nr)+"_"+fname
            #     data = h5py.File(reconimgs_filename)
            #     data.keys()
            #     reconimgs_echo = data['reconstruction']
            #     reconimgs[echo_nr, ...] = torch.from_numpy(np.array(reconimgs_echo))/10000

            masked_kspace = masked_kspace[..., 0] + 1j * masked_kspace[..., 1]
            pred = torch.fft.ifft2(masked_kspace, dim=(2, 3))
            pred = torch.stack([pred.real, pred.imag], dim=-1)
            reconimgs = torch.sum(transforms.complex_mul(pred, transforms.complex_conj(sense)), 1)/10000

            # load mask for the brain region
            with h5py.File(str(fname_full), 'r') as data:
                mask_brain = data['mask_brain']
                mask_brain = np.array(mask_brain)

            mask_head = np.ones_like(mask_brain)

            # prepare init maps (\Phi_0). since computation is time consuming, we store it for future loading.
            if 'R2star_map_init_acce_' + str(self.acceleration) in data:
                R2star_map_init = data['R2star_map_init_acce_' + str(self.acceleration)]
                S0_map_init = data['S0_map_init_acce_' + str(self.acceleration)]
                B0_map_init = data['B0_map_init_acce_' + str(self.acceleration)]
                phi_map_init = data['phi_map_init_acce_' + str(self.acceleration)]

                R2star_map_init = torch.from_numpy(np.float32(R2star_map_init))
                S0_map_init = torch.from_numpy(np.float32(S0_map_init))
                B0_map_init = torch.from_numpy(np.float32(B0_map_init))
                phi_map_init = torch.from_numpy(np.float32(phi_map_init))
            else:
                print('except in loading init maps: ' + str(fname_full))
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = R2star_B0_real_S0_complex_mapping(
                    reconimgs, torch.from_numpy(self.TEs), torch.from_numpy(mask_brain), torch.from_numpy(mask_head),
                    fullysample=True)

            # prepare the reference maps
            if 'R2star_map_target_acce_' + str(self.acceleration) in data:
                R2star_map_target = data['R2star_map_target_acce_' + str(self.acceleration)]
                S0_map_target = data['S0_map_target_acce_' + str(self.acceleration)]
                B0_map_target = data['B0_map_target_acce_' + str(self.acceleration)]
                phi_map_target = data['phi_map_target_acce_' + str(self.acceleration)]

                R2star_map_target = torch.from_numpy(np.float32(R2star_map_target))
                S0_map_target = torch.from_numpy(np.float32(S0_map_target))
                B0_map_target = torch.from_numpy(np.float32(B0_map_target))
                phi_map_target = torch.from_numpy(np.float32(phi_map_target))
            else:
                print('except in loading target maps: ' + str(fname_full))

                kspace = kspace[..., 0] + 1j * kspace[..., 1]
                imspace = torch.fft.ifft2(kspace, dim=(2, 3))
                imspace = torch.stack([imspace.real, imspace.imag], dim=-1)
                kspace = torch.stack([kspace.real, kspace.imag], dim=-1)

                R2star_map_target, S0_map_target, B0_map_target, phi_map_target = R2star_S0_mapping_from_ksp(
                    imspace, torch.from_numpy(self.TEs), sense, torch.from_numpy(mask_brain),
                    torch.from_numpy(mask_head),
                    fullysample=True, option=0
                )

        return R2star_map_init.squeeze(), S0_map_init.squeeze(), B0_map_init.squeeze(), phi_map_init.squeeze(), \
               R2star_map_target.squeeze(), S0_map_target.squeeze(), B0_map_target.squeeze(), phi_map_target.squeeze(), \
               kspace, mask_brain, subsampling_mask.squeeze(1).squeeze(-1), self.TEs, sense, fname, slice
