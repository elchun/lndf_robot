# from sympy import Q
import math
from pydoc import describe
from turtle import distance
import torch
from torch.nn import functional as F

from ndf_robot.model.conv_occupancy_net.encoder.pointnet import LocalPoolPointnet


def occupancy(model_outputs, ground_truth, val=False):
    """
    LEGACY DO NOT USE???
    """
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False, **kwargs):
    """
    NLL loss for predicting occupacny

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


def distance_net(model_outputs, ground_truth, val=False):
    """
    UNUSED
    """
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    dist = torch.abs(model_outputs['occ'] - label * 100).mean()
    loss_dict['dist'] = dist

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    """
    UNUSED
    """
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5)
            + (1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict


def triplet(occ_margin=0, positive_loss_scale=0.3, negative_loss_scale=0.3,
    similar_occ_only=False, positive_margin=0.001, negative_margin=0.1):
    """
    Create triplet loss function enforcing similarity between rotated
    activations and difference between random coordinates defined as:

    loss = max(occ_loss - occ_margin, 0) \
        + positive_loss_scale * latent_positive_loss \
        + negative_loss_scale * latent_negative_loss

    where latent_positive_loss is cosine similarity between activations and
    activations of rotated shape while latent_negative_loss is cosine
    difference between rotated latent and a activations of a randomly sampled
    point

    Args:
        occ_margin (float, optional): Loss from occupancy is 0 when below
            margin. Defaults to 0.
        positive_loss_scale (float, optional): Influence of positive loss on
            combined loss. Defaults to 0.3.
        negative_loss_scale (float, optional): Influence of negative loss on
            combined loss. Defaults to 0.3.
        similar_occ_only (bool, optional): True to only compare activations
            from points where the ground truth occupancy is true. Defaults to
            False.
        positive_margin (float, optional): margin to use in cosine similarity
            for latent_positive_loss. Defaults to 0.001.
        negative_margin (float, optional): margin to use in cosine similarity
            for latent_negative_loss. Defaults to 0.1

    Returns:
        function(model_ouputs, ground_truth): Loss function that takes model
            outputs and ground truth
    """

    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        Triplet loss enforcing similarity between rotated activations and
        difference between random coords

        Args:
            model_outputs (dict): Dictionary containing 'standard', 'rot',
                'standard_act_hat', 'rot_act_hat'
            ground_truth (dict): Dictionary containing 'occ'
            occ_margin (float, optional): Lower value makes occupancy
                prediction better

        Returns:
            dict: dict containing 'occ'
        """

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        # Get outputs from dict
        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']
        rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        # Mask all occ that are not in the shape
        if similar_occ_only:
            non_zero_label = label.unsqueeze(-1)
            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label
            rot_negative_act_hat *= non_zero_label

        standard_act_hat = torch.flatten(standard_act_hat, start_dim=1)
        rot_act_hat = torch.flatten(rot_act_hat, start_dim=1)
        rot_negative_act_hat = torch.flatten(rot_negative_act_hat, start_dim=1)

        # Calculate loss of occupancy
        standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
        rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

        occ_loss = (standard_loss_occ + rot_loss_occ) / 2

        device = standard_act_hat.get_device()
        if positive_loss_scale > 0:
            # Calculate loss from similarity between latent descriptors
            latent_positive_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat,
                torch.ones(standard_act_hat.shape[0]).to(device), margin=positive_margin)
            latent_positive_loss = latent_positive_loss.mean()
        else:
            latent_positive_loss = 0

        if negative_loss_scale > 0:
            # Calculate loss from difference between unrelated latent descriptors
            latent_negative_loss = F.cosine_embedding_loss(standard_act_hat, rot_negative_act_hat,
                -torch.ones(standard_act_hat.shape[0]).to(device), margin=negative_margin)
            latent_negative_loss = latent_negative_loss.mean()
        else:
            latent_negative_loss = 0

        loss_dict['occ'] = max(occ_loss - occ_margin, 0) \
            + positive_loss_scale * latent_positive_loss \
            + negative_loss_scale * latent_negative_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_positive_loss)
        print('latent neg loss: ', latent_negative_loss)

        return loss_dict

    return loss_fn


def simple_loss(positive_loss_scale: int = 1, negative_loss_scale: int = 1,
    num_negative_samples: int=100, type: str = 'cos_single'):
    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        L2 loss enforcing similarity between rotated activations and
        difference between random coords

        Args:
            model_outputs (dict): Dictionary containing 'standard', 'rot',
                'standard_act_hat', 'rot_act_hat'
            ground_truth (dict): Dictionary containing 'occ'
            occ_margin (float, optional): Lower value makes occupancy
                prediction better

        Returns:
            dict: dict containing 'occ'
        """

        similar_occ_only = True

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']

        rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        if similar_occ_only:
            # Create bool tensor to mask unoccupied stuff
            non_zero_label = torch.tensor(label.unsqueeze(-1)).bool()

            # print('nz_label: ', non_zero_label)
            # print('Shape: ', non_zero_label.size())  # [5, 1500, 1]
            # print('Sum:', non_zero_label.sum())
            # # standard_act_hat = torch.masked_select(standard_act_hat, non_zero_label, keep_dims=True)
            # standard_act_hat = standard_act_hat[:, non_zero_label[:-1], :]
            # rot_act_hat = torch.masked_select(rot_act_hat, non_zero_label)
            # rot_negative_act_hat = torch.masked_select(rot_negative_act_hat, non_zero_label)

            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label
            rot_negative_act_hat *= non_zero_label

        # print('Standard shape: ', standard_act_hat.size())  # [6, 1500, 32]

        # print('Flattened standard size: ', standard_act_hat.size())  # Was [6, 48000]
        # print('Flattened rot size: ', rot_act_hat.size())  # Was [6, 48000]

        # Calculate loss of occupancy
        standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
        rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
            + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

        occ_loss = (standard_loss_occ + rot_loss_occ) / 2

        #-- MSE loss -- #
        if type == 'mse':
            latent_positive_loss = F.mse_loss(standard_act_hat, rot_act_hat, reduction='mean')

            latent_negative_loss = F.mse_loss(rot_act_hat[:num_negative_samples, :],
                rot_negative_act_hat[:num_negative_samples, :], reduction='mean')

            # -- Overall loss -- #
            overall_loss = occ_loss \
                + positive_loss_scale * latent_positive_loss \
                + negative_loss_scale * latent_negative_loss

        # # -- l1 loss -- #
        elif type == 'l1':
            latent_positive_loss = F.l1_loss(standard_act_hat, rot_act_hat, reduction='mean')

            # latent_negative_loss = F.l1_loss(rot_act_hat[:num_negative_samples, :],
            #     rot_negative_act_hat[:num_negative_samples, :], reduction='mean')

            latent_negative_loss = F.l1_loss(rot_act_hat[:num_negative_samples, :],
                rot_negative_act_hat[:num_negative_samples, :], reduction='none')
            latent_negative_loss = latent_negative_loss.sort(dim=0)[0]
            latent_negative_loss = -latent_negative_loss[:50, :].mean()

            # -- Overall loss -- #
            overall_loss = occ_loss \
                + positive_loss_scale * latent_positive_loss \
                + negative_loss_scale * latent_negative_loss

        # -- Cosine loss -- #
        elif type == 'cos':
            latent_positive_loss = F.cosine_similarity(standard_act_hat, rot_act_hat, dim=2)

            # latent_negative_loss = F.cosine_similarity(rot_act_hat[:num_negative_samples, :],
            #     rot_negative_act_hat[:num_negative_samples, :], dim=2)

            r_idx1 = torch.randperm(standard_act_hat.shape[1])
            r_idx2 = torch.randperm(standard_act_hat.shape[1])

            # latent_negative_loss = F.cosine_similarity(rot_act_hat[:, :num_negative_samples, :],
            #     rot_negative_act_hat[:, :num_negative_samples, :], dim=2)

            latent_negative_loss = F.cosine_similarity(rot_act_hat[:, r_idx1, :],
               rot_act_hat[:, r_idx2, :], dim=2)

            # print(latent_negative_loss.sum(dim=1))
            # print(rot_act_hat[:, r_idx2, :].shape)

            latent_positive_loss = 1 - latent_positive_loss[latent_positive_loss != 0].mean()
            latent_negative_loss = latent_negative_loss[latent_negative_loss != 0].mean()

            # -- Overall loss -- #
            overall_loss = occ_loss \
                + positive_loss_scale * latent_positive_loss \
                + negative_loss_scale * latent_negative_loss

        # -- Cosine loss -- #
        elif type == 'cos_rdif':
            # Correlation
            latent_positive_loss = F.cosine_similarity(standard_act_hat, rot_act_hat, dim=2)

            # Break symmetry
            latent_negative_loss = F.cosine_similarity(rot_act_hat,
                rot_negative_act_hat, dim=2)

            # Spread out field
            r_idx1 = torch.randperm(standard_act_hat.shape[1])
            r_idx2 = torch.randperm(standard_act_hat.shape[1])

            latent_spread_loss = F.cosine_similarity(rot_act_hat[:, r_idx1, :],
               rot_act_hat[:, r_idx2, :], dim=2)

            latent_positive_loss = 1 - latent_positive_loss[latent_positive_loss != 0].mean()
            latent_negative_loss = latent_negative_loss[latent_negative_loss != 0].mean()
            latent_spread_loss = latent_spread_loss[latent_spread_loss != 0].mean()

            # -- Overall loss -- #
            overall_loss = occ_loss \
                + positive_loss_scale * latent_positive_loss \
                + negative_loss_scale * latent_negative_loss \
                + 0.1 * negative_loss_scale * latent_spread_loss

        elif type == 'cos_single':
            non_zero_label = torch.tensor(label).int()

            dev = non_zero_label.device

            # r_idx = torch.randperm(non_zero_idx.shape[0])
            # non_zero_idx = torch.nonzero(non_zero_label)
            # non_zero_idx = non_zero_idx[r_idx]

            non_zero_idx = torch.arange(0, label.shape[1])[None, :].repeat(6, 1).to(dev)
            non_zero_idx = non_zero_label * non_zero_idx
            non_zero_idx = non_zero_idx.sort(dim=1, descending=True)[0]

            n_sim_samples = 1
            n_diff_samples = 512
            n_diff_loss_samples = 512
            # n_diff_loss_samples = 32
            n_diff_repeats = 512  # Should be n_diff_samples / n_sim_samples

            sim_idxs = non_zero_idx[:, :n_sim_samples][:, :, None]
            sim_idxs = sim_idxs.repeat(1, 1, standard_act_hat.shape[-1])

            diff_idxs = non_zero_idx[:, n_sim_samples:n_diff_samples + n_sim_samples][:, :, None]
            diff_idxs = diff_idxs.repeat(1, 1, standard_act_hat.shape[-1])

            # print(diff_idxs)
            # print(diff_idxs.shape)

            # Correlation
            latent_positive_loss = F.cosine_similarity(standard_act_hat.gather(1, sim_idxs),
                rot_act_hat.gather(1, sim_idxs), dim=2)

            # Difference
            latent_negative_loss = F.cosine_similarity(rot_act_hat.gather(1, sim_idxs).repeat(1, n_diff_repeats, 1),
               rot_act_hat.gather(1, diff_idxs), dim=2)

            latent_positive_loss = 1 - latent_positive_loss.mean()
            latent_negative_loss = latent_negative_loss.sort(dim=1, descending=True)[0]
            latent_negative_loss = latent_negative_loss[:n_diff_loss_samples].mean()

            overall_loss = occ_loss \
                + positive_loss_scale * latent_positive_loss \
                + negative_loss_scale * latent_negative_loss \

            # # sim_idx = non_zero_idx[
            # # diff


            # n_samples = 100  # Oversample to start since many samples may be invalid

            # n_sim_samples = 1
            # n_diff_samples = 10

            # sim_idxs = torch.randint(0, standard_act_hat.shape[1], (n_samples,))

            # # Correlation
            # latent_positive_loss = F.cosine_similarity(standard_act_hat[:, sim_idxs, :], rot_act_hat[:, sim_idxs, :], dim=2)

            # # # Break symmetry
            # # latent_negative_loss = F.cosine_similarity(rot_act_hat[],
            # #     rot_negative_act_hat, dim=2)

            # # Spread out field
            # r_idx = torch.randperm(standard_act_hat.shape[1])
            # # r_idx2 = torch.randperm(standard_act_hat.shape[1])

            # latent_negative_loss = F.cosine_similarity(rot_act_hat[:, sim_idxs, :],
            #    rot_act_hat[:, r_idx[:n_samples], :], dim=2)

            # latent_positive_loss = 1 - latent_positive_loss[latent_positive_loss != 0][:n_sim_samples].mean()
            # latent_negative_loss = latent_negative_loss[latent_negative_loss != 0][:n_diff_samples].mean()

            # # latent_spread_loss = latent_spread_loss[latent_spread_loss != 0].mean()

            # # -- Overall loss -- #
            # overall_loss = occ_loss \
            #     + positive_loss_scale * latent_positive_loss \
            #     + negative_loss_scale * latent_negative_loss \

        loss_dict['occ'] = overall_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_positive_loss)
        print('latent neg loss: ', latent_negative_loss)
        # if latent_spread_loss:
        #     print('latent spread loss: ', latent_spread_loss)
        print('overall loss: ', overall_loss)

        return loss_dict

    return loss_fn


def cos_contrast(positive_loss_scale: int = 1, negative_loss_scale: int = 1,
    diff_loss_sample_rate: int = 0.0625):
    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        L2 loss enforcing similarity between rotated activations and
        difference between random coords

        Args:
            model_outputs (dict): Dictionary containing 'standard', 'rot',
                'standard_act_hat', 'rot_act_hat'
            ground_truth (dict): Dictionary containing 'occ'
            occ_margin (float, optional): Lower value makes occupancy
                prediction better

        Returns:
            dict: dict containing 'occ'
        """

        similar_occ_only = True

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']

        rot_negative_act_hat = model_outputs['rot_negative_act_hat']

        if similar_occ_only:
            # Create bool tensor to mask unoccupied stuff
            non_zero_label = torch.tensor(label.unsqueeze(-1)).bool()

            standard_act_hat *= non_zero_label
            rot_act_hat *= non_zero_label
            rot_negative_act_hat *= non_zero_label

        # print('Standard shape: ', standard_act_hat.size())  # [6, 1500, 32]

        # print('Flattened standard size: ', standard_act_hat.size())  # Was [6, 48000]
        # print('Flattened rot size: ', rot_act_hat.size())  # Was [6, 48000]

        # Calculate loss of occupancy
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
        n_diff_loss_samples = math.floor(n_diff_samples * diff_loss_sample_rate)
        # n_diff_loss_samples = 32  # Set in header
        n_diff_repeats = 256  # Should be n_diff_samples / n_sim_samples

        # print('n_sum: ', non_zero_label.sum(dim=1))

        # -- Select indicies to use as similar and different samples -- $
        # Shuffle good indices
        valid_idxs = non_zero_idx[:, :n_sim_samples + n_diff_samples]
        r_perm = torch.randperm(valid_idxs.shape[-1])
        valid_idxs = non_zero_idx[:, r_perm]

        sim_idxs = valid_idxs[:, :n_sim_samples][:, :, None]
        sim_idxs = sim_idxs.repeat(1, 1, standard_act_hat.shape[-1])

        diff_idxs = valid_idxs[:, n_sim_samples:n_diff_samples + n_sim_samples][:, :, None]
        diff_idxs = diff_idxs.repeat(1, 1, standard_act_hat.shape[-1])

        # Correlation
        latent_positive_loss = F.cosine_similarity(standard_act_hat.gather(1, sim_idxs),
            rot_act_hat.gather(1, sim_idxs), dim=2)

        # Difference
        # sim_tensor = rot_act_hat.gather(1, sim_idxs).repeat(1, n_diff_repeats, 1)
        sim_tensor = standard_act_hat.gather(1, sim_idxs).repeat(1, n_diff_repeats, 1)
        latent_negative_loss = F.cosine_similarity(sim_tensor,
            standard_act_hat.gather(1, diff_idxs), dim=2)

        latent_positive_loss = 1 - latent_positive_loss.mean()
        latent_negative_loss = latent_negative_loss.sort(dim=1, descending=True)[0]
        # print(latent_negative_loss)
        print('DEBUG: ', latent_negative_loss[:n_diff_loss_samples])
        print('DEBUG: ', latent_negative_loss[:, :n_diff_loss_samples])
        latent_negative_loss = latent_negative_loss[:, :n_diff_loss_samples].mean()  # Add colon

        overall_loss = occ_loss \
            + positive_loss_scale * latent_positive_loss \
            + negative_loss_scale * latent_negative_loss \

        loss_dict['occ'] = overall_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_positive_loss)
        print('latent neg loss: ', latent_negative_loss)
        print('overall loss: ', overall_loss)

        return loss_dict

    return loss_fn


def cos_relative(latent_loss_scale: int = 1):
    def loss_fn(model_outputs, ground_truth, val=False, **kwargs):
        """
        L2 loss enforcing similarity between rotated activations and
        difference between random coords

        Args:
            model_outputs (dict): Dictionary containing 'standard', 'rot',
                'standard_act_hat', 'rot_act_hat'
            ground_truth (dict): Dictionary containing 'occ'
            occ_margin (float, optional): Lower value makes occupancy
                prediction better

        Returns:
            dict: dict containing 'occ'
        """

        similar_occ_only = True

        loss_dict = dict()
        label = ground_truth['occ'].squeeze()
        label = (label + 1) / 2.

        standard_outputs = model_outputs['standard']
        rot_outputs = model_outputs['rot']

        standard_act_hat = model_outputs['standard_act_hat']
        rot_act_hat = model_outputs['rot_act_hat']

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
        target = torch.zeros(relative_sim.shape).to(dev)
        target[:, :n_sim_samples] = 1

        # weight = torch.ones(relative_sim.shape[1])
        # weight[:n_sim_samples] *= 10

        latent_loss = F.cross_entropy(relative_sim, target)

        # -- With binary cross entropy -- #
        # probs = F.softmax(relative_sim, dim=1)  # (6, 257)
        # probs = probs[:, :n_sim_samples]  # (6 x 1)
        # latent_loss = F.binary_cross_entropy(probs, torch.ones(probs.shape).to(dev))
        # latent_loss = probs[:, :n_sim_samples].mean()
        # latent_loss = 1 - latent_loss

        overall_loss = occ_loss \
            + latent_loss_scale * latent_loss

        loss_dict['occ'] = overall_loss

        print('occ loss: ', occ_loss)
        print('latent pos loss: ', latent_loss)
        print('overall loss: ', overall_loss)

        return loss_dict

    return loss_fn


def cos_distance(latent_loss_scale: int = 1, dis_offset: float = 0.002):
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

        # target = torch.zeros(relative_sim.shape[1]).to(dev)
        # target[:n_sim_samples] = 1

        # Get weight based on coordinates
        target_sim_idxs = sim_idxs[:, :, :3]  # Only take the first three repeats for gather
        target_diff_idxs = diff_idxs[:, :, :3]
        sim_coords = coords.gather(1, target_sim_idxs)
        diff_coords = coords.gather(1, target_diff_idxs)
        all_coords = torch.cat((sim_coords, diff_coords), dim=1)
        distances = ((all_coords - sim_coords)**2).sum(dim=-1).sqrt() # (6, 257)

        # Comment out if using max
        target = 1 / (distances + dis_offset)

        # Comment out if using pure offset
        # target = 1 / (dis_scale * torch.maximum(distances, torch.tensor(dis_offset)))

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


def cos_distance_with_l2(latent_loss_scale: int = 1, radius: float=0.1):
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


def rotated_triplet_log(model_outputs, ground_truth, **kwargs):
    """
    Joint loss of occupancy and log of similiarty between rotated and unrotated
    coordinates
    Args:
        model_outputs (dict): Dictionary containing 'standard', 'rot',
            'standard_act_hat', 'rot_act_hat'
        ground_truth (dict): Dictionary containing 'occ'
    Returns:
        dict: dict containing 'occ'
    """
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)
    rot_negative_act_hat = torch.flatten(model_outputs['rot_negative_act_hat'],
        start_dim=1)

    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

    occ_loss = (standard_loss_occ + rot_loss_occ) / 2

    # Calculate loss from similarity between latent descriptors
    negative_margin = 10**-3
    positive_margin = 10**-8
    device = standard_act_hat.get_device()
    latent_positive_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat,
        torch.ones(standard_act_hat.shape[0]).to(device), margin=positive_margin)

    latent_negative_loss = F.cosine_embedding_loss(standard_act_hat, rot_negative_act_hat,
        -torch.ones(standard_act_hat.shape[0]).to(device), margin=negative_margin)

    latent_positive_loss = latent_positive_loss.mean()
    latent_negative_loss = latent_negative_loss.mean()

    negative_loss_scale = 0.5
    positive_loss_scale = 0.05


    # loss_dict['occ'] = occ_loss \
    #     + positive_loss_scale * torch.log(positive_pad + latent_positive_loss) \
    #     + negative_loss_scale * torch.log(torch.tensor(negative_pad)
    #         + max(latent_negative_loss - negative_margin, 0))

    # The negative loss should prevent all the activations from becoming similar
    # to each other while the positive loss encourages rotation invariance
    loss_dict['occ'] = occ_loss \
        + positive_loss_scale * torch.log(10**-5 + latent_positive_loss) \
        + negative_loss_scale * latent_negative_loss \

    print('occ loss: ', occ_loss)
    # print('latent_scale: ', latent_loss_scale)
    print('latent pos loss: ', latent_positive_loss)
    print('latent neg loss: ', latent_negative_loss)

    return loss_dict

def rotated_log(model_outputs, ground_truth, it=-1):
    """
    Joint loss of occupancy and log of similiarty between rotated and unrotated
    coordinates

    Appears to overfit and reduce magnitude of all activations

    Args:
        model_outputs (dict): Dictionary containing 'standard', 'rot',
            'standard_act_hat', 'rot_act_hat'
        ground_truth (dict): Dictionary containing 'occ'
        it (int, optional): current number of iterations. Defaults to -1.
    """
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    # Get outputs from dict
    standard_outputs = model_outputs['standard']
    rot_outputs = model_outputs['rot']
    standard_act_hat = torch.flatten(model_outputs['standard_act_hat'], start_dim=1)
    rot_act_hat = torch.flatten(model_outputs['rot_act_hat'], start_dim=1)

    # rot_negative_act_hat = torch.flatten(model_outputs['rot_negative_act_hat'],
    #     start_dim=1)

    # print(standard_act_hat[0, :5])
    # print(rot_negative_act_hat[0, :5])

    # Calculate loss of occupancy
    standard_loss_occ = -1 * (label * torch.log(standard_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - standard_outputs['occ'] + 1e-5)).mean()
    rot_loss_occ = -1 * (label * torch.log(rot_outputs['occ'] + 1e-5)
        + (1 - label) * torch.log(1 - rot_outputs['occ'] + 1e-5)).mean()

    occ_loss = (standard_loss_occ + rot_loss_occ) / 2

    # Calculate loss from similarity between latent descriptors
    device = standard_act_hat.get_device()
    latent_positive_loss = F.cosine_embedding_loss(standard_act_hat, rot_act_hat,
        torch.ones(standard_act_hat.shape[0]).to(device))

    latent_positive_loss = latent_positive_loss.mean()
    positive_loss_scale = 1  # Higher scale makes slope less steep
    positive_loss_log_scale = 0.1
    # margin = 4 * 10 ** -8
    margin = 10 ** -9

    loss_dict['occ'] = occ_loss + positive_loss_log_scale * torch.log(
        margin + positive_loss_scale * latent_positive_loss)

    print('occ loss: ', occ_loss)
    print('latent pos loss: ', latent_positive_loss)

    return loss_dict
