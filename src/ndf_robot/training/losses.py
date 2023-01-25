# from sympy import Q
import math
from pydoc import describe
from turtle import distance
import torch
from torch.nn import functional as F

from ndf_robot.model.conv_occupancy_net.encoder.pointnet import LocalPoolPointnet


def occupancy_net(model_outputs, ground_truth, val=False, **kwargs):
    """
    NLL loss for predicting occupancy.  Primarily used with the vnn_ndf
    training scripts.

    Args:
        model_outputs (dict): Dictionary with the key 'occ' corresponding to
            tensor
        ground_truth (dict): Dictionary with the key 'occ' corresponding to
            tensor
        val (bool, optional): Unused. Defaults to False.

    Returns:
        dict: Dictionary containing 'occ' which corresponds to tensor of loss
            for each element in batch
    """
    # Good if using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # print('model outputs: ', model_outputs)
    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def conv_occupancy_net(model_outputs, ground_truth, val=False, **kwargs):
    """
    NLL loss for predicting occupancy with convolutional neural net
    Good if not using a sigmoid output of occupancy network decoder

    Args:
        model_outputs (dict): Dict containing the key 'standard' which maps
            to a dictionary which has the key 'occ' which maps to a
            tensor :(
        ground_truth (dict): Dict with the key 'occ' which maps to a tensor
            of ground truth occupancies
        val (bool, optional): If this is a validation set (Unused).
            Defaults to False.

    Returns:
        dict: Dictionary with key 'occ' corresponding to loss of occupancy
    """
    standard_output = model_outputs['standard']

    # Good if not using sigmoid on output of decoder
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # print('model outputs: ', model_outputs)
    occ_loss = -1 * (label * torch.log(standard_output['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - standard_output['occ'] + 1e-5)).mean()

    loss_dict['occ'] = occ_loss
    print('occ_loss: ', occ_loss)
    return loss_dict


def contrastive_cross_entropy(latent_loss_scale: int = 1, dis_offset: float = 0.002):
    """
    Loss function used in preprint.  Uses distance between reference and target
    points to weight cosine similarity.

    Args:
        latent_loss_scale (int, optional): weight on latent loss
            (in relation to occupancy loss). Defaults to 1.
        dis_offset (float, optional): Epsilon value in the 1/(dis + offset)
            calculation of target cosine similarity. Defaults to 0.002.
    """
    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        Use distance to reweight similarity
        """

        similar_occ_only = True

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']

        coords = model_outputs['coords']

        # rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        if similar_occ_only:
            # Create bool tensor to mask unoccupied stuff
            non_zero_label = torch.tensor(label.unsqueeze(-1)).bool()

            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label

        # -- Calculate loss of occupancy -- #
        standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
        rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

        occ_loss = (standard_loss_occ + rot_loss_occ) / 2

        # -- Get all points with non-zero ground truth occupancy -- #
        non_zero_label = torch.tensor(label).int()

        dev = non_zero_label.device

        non_zero_idx = torch.arange(0, label.shape[1])[None, :].repeat(6, 1).to(dev)
        non_zero_idx = non_zero_label * non_zero_idx
        non_zero_idx = non_zero_idx.sort(dim=1, descending=True)[0]

        n_sim_samples = 1
        n_diff_samples = 256

        # print('n_sum: ', non_zero_label.sum(dim=1))

        # -- Select indicies to use as similar and different samples -- $
        # Shuffle good indices
        valid_idxs = non_zero_idx[:, :n_sim_samples + n_diff_samples]
        r_perm = torch.randperm(valid_idxs.shape[-1])
        valid_idxs = non_zero_idx[:, r_perm]

        # Select similar examples
        sim_idxs = valid_idxs[:, :n_sim_samples][:, :, None]
        sim_idxs = sim_idxs.repeat(1, 1, standard_act_hat.shape[-1])

        # Select different examples
        diff_idxs = valid_idxs[:, n_sim_samples:n_diff_samples + n_sim_samples][:, :, None]
        diff_idxs = diff_idxs.repeat(1, 1, standard_act_hat.shape[-1])

        # Correlation
        latent_positive_sim = F.cosine_similarity(standard_act_hat.gather(1, sim_idxs),
            rot_act_hat.gather(1, sim_idxs), dim=2)

        # Difference
        # sim_tensor = rot_act_hat.gather(1, sim_idxs).repeat(1, n_diff_repeats, 1)
        sim_tensor = standard_act_hat.gather(1, sim_idxs).repeat(1, n_diff_samples, 1)
        latent_negative_sim = F.cosine_similarity(sim_tensor,
            rot_act_hat.gather(1, diff_idxs), dim=2)

        relative_sim = torch.cat((latent_positive_sim, latent_negative_sim), dim=1)  # (6, 257)

        # -- With cross entropy -- #

        # For a given sample point x, we calculate the euclidean distance between
        # it and the reference point.  We repeat this for all sample points {x}.
        # Then, we create a probability distribution P where the probability
        # of point x is 1 / (dis + epsilon).  This looks like a cone with the
        # reference point at the center.
        # Finally, we use torch F.cross_entropy to try to match the cosine similarity
        # of the sample point with its calculated target probability.

        # Get weight based on coordinates
        target_sim_idxs = sim_idxs[:, :, :3]  # Only take the first three repeats for gather
        target_diff_idxs = diff_idxs[:, :, :3]
        sim_coords = coords.gather(1, target_sim_idxs)
        diff_coords = coords.gather(1, target_diff_idxs)
        all_coords = torch.cat((sim_coords, diff_coords), dim=1)
        distances = ((all_coords - sim_coords)**2).sum(dim=-1).sqrt() # (6, 257)

        # Comment out if using max
        target = 1 / (distances + dis_offset)
        target = target / target.sum(dim=-1, keepdim=True)

        # print(weight)
        print(target.min())
        print(target.sort(descending=True)[0][:, :10])

        # print(weight.sum())
        # print(relative_sim)

        latent_loss = F.cross_entropy(relative_sim, target)

        overall_loss = occ_loss \
            + latent_loss_scale * latent_loss

        loss_dict['occ'] = overall_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_loss)
        print('overall loss: ', overall_loss)

        return loss_dict

    return loss_fn


def contrastive_l2(latent_loss_scale: int = 1, radius: float=0.1):
    """
    Replace cross entropy with L2.  Uses distance between reference and target
    points to weight cosine similarity.

    Args:
        latent_loss_scale (int, optional): weight on latent loss
            (in relation to occupancy loss). Defaults to 1.
        dis_offset (float, optional): Epsilon value in the 1/(dis + offset)
            calculation of target cosine similarity. Defaults to 0.002.
    """
    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        Use distance to reweight similarity
        """

        similar_occ_only = True

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']

        coords = model_outputs['coords']

        # rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        if similar_occ_only:
            # Create bool tensor to mask unoccupied stuff
            non_zero_label = torch.tensor(label.unsqueeze(-1)).bool()

            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label

        # -- Calculate loss of occupancy -- #
        standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
        rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

        occ_loss = (standard_loss_occ + rot_loss_occ) / 2

        # -- Get all points with non-zero ground truth occupancy -- #
        non_zero_label = torch.tensor(label).int()

        dev = non_zero_label.device

        non_zero_idx = torch.arange(0, label.shape[1])[None, :].repeat(6, 1).to(dev)
        non_zero_idx = non_zero_label * non_zero_idx
        non_zero_idx = non_zero_idx.sort(dim=1, descending=True)[0]

        n_sim_samples = 1
        n_diff_samples = 256

        # print('n_sum: ', non_zero_label.sum(dim=1))

        # -- Select indicies to use as similar and different samples -- $
        # Shuffle good indices
        valid_idxs = non_zero_idx[:, :n_sim_samples + n_diff_samples]
        r_perm = torch.randperm(valid_idxs.shape[-1])
        valid_idxs = non_zero_idx[:, r_perm]

        # Select similar examples
        sim_idxs = valid_idxs[:, :n_sim_samples][:, :, None]
        sim_idxs = sim_idxs.repeat(1, 1, standard_act_hat.shape[-1])

        # Select different examples
        diff_idxs = valid_idxs[:, n_sim_samples:n_diff_samples + n_sim_samples][:, :, None]
        diff_idxs = diff_idxs.repeat(1, 1, standard_act_hat.shape[-1])

        # Correlation
        latent_positive_sim = F.cosine_similarity(standard_act_hat.gather(1, sim_idxs),
            rot_act_hat.gather(1, sim_idxs), dim=2)

        # Difference
        # sim_tensor = rot_act_hat.gather(1, sim_idxs).repeat(1, n_diff_repeats, 1)
        sim_tensor = standard_act_hat.gather(1, sim_idxs).repeat(1, n_diff_samples, 1)
        latent_negative_sim = F.cosine_similarity(sim_tensor,
            rot_act_hat.gather(1, diff_idxs), dim=2)

        relative_sim = torch.cat((latent_positive_sim, latent_negative_sim), dim=1)  # (6, 257)

        # -- With L2 -- #

        # For a given sample point x, we calculate the euclidean distance between
        # it and the reference point.  We repeat this for all sample points {x}.
        # Then, we create a probability distribution P where the probability
        # of point x is 1 / (dis + epsilon).  This looks like a cone with the
        # reference point at the center.
        # Finally, we use torch F.cross_entropy to try to match the cosine similarity
        # of the sample point with its calculated target probability.

        # target = torch.zeros(relative_sim.shape[1]).to(dev)
        # target[:n_sim_samples] = 1

        # Get weight based on coordinates
        target_sim_idxs = sim_idxs[:, :, :3]  # Only take the first three repeats for gather
        target_diff_idxs = diff_idxs[:, :, :3]
        sim_coords = coords.gather(1, target_sim_idxs)
        diff_coords = coords.gather(1, target_diff_idxs)
        all_coords = torch.cat((sim_coords, diff_coords), dim=1)
        distances = ((all_coords - sim_coords)**2).sum(dim=-1).sqrt()  # (6, 257)

        def target_func(dis, rad):
            # return torch.exp(-dis**2 * (math.log(2) / rad**2))
            return 2 * torch.exp(-dis**2 * (math.log(2) / rad**2)) - 1

        target = target_func(distances, radius)

        print(target.min())
        print(target.sort(descending=True)[0][:, :10])

        latent_loss = F.mse_loss(relative_sim, target)

        overall_loss = occ_loss \
            + latent_loss_scale * latent_loss

        loss_dict['occ'] = overall_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_loss)
        print('overall loss: ', overall_loss)

        return loss_dict

    return loss_fn