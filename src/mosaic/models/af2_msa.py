from __future__ import annotations
import numpy as np 
from contextlib import contextmanager
import re 
import torch # check if we should move to complete jax
import torch.nn as nn
from dataclasses import dataclass

_restypes = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",]
_restypes_with_x = _restypes + ["X"]
_restypes_with_x_and_gap = _restypes_with_x + ["-"]

restype_order_with_x = {res:idx for idx, res in enumerate(_restypes_with_x)}
restype_order_with_x_and_gap = {res:idx for idx, res in enumerate(_restypes_with_x_and_gap)}


@dataclass(frozen=True)
class RawMSA:
    tokens: np.ndarray #int32 [N, L]
    deletion_matrix: np.ndarray #float32 [N, L]

@dataclass(frozen=True)
class AF2MSAFeatures:
    msa_feat: np.ndarray
    msa_mask: np.ndarray

    extra_msa: np.ndarray
    extra_deletion_value: np.ndarray
    extra_has_deletion: np.ndarray
    extra_msa_mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        pass

# start define how to parse the whole msa: 

def load_a3m_file(file_name: str):
    """
    Loads an A3M (multiple sequence alignment) file and extracts the raw amino acid sequences.

    Args:
        file_name: Path to the A3M file.

    Returns:
        A list of strings where each string represents an individual protein sequence from the input MSA.
    """

    seqs = None

    current_id = None
    seqs = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
                seqs[current_id] = ""
            elif current_id:
                seqs[current_id] = line

    sequences = [value for value in seqs.values()]
    return sequences 


def onehot_encode_aa_type(seq:str, include_gap_token=False):
    """
    Converts a protein sequence into one-hot encoding. X represents an unkown amino acid.

    Args:
        seq:  A string representing the amino acid sequence using single-letter codes.
        include_gap_token: If True, includes an extra token ('-') in the encoding to
                           represent gaps.

    Returns:
        A PyTorch tensor of shape (N_res, 22) if `include_gap_token` is True,
        or shape (N_res, 21) otherwise.  Here, N_res is the length of the sequence.
    """
    restype_order = restype_order_with_x if not include_gap_token else restype_order_with_x_and_gap
    encoding = None
    seq = [restype_order[s] for s in seq]
    encoding = nn.functional.one_hot(torch.LongTensor(seq), num_classes=len(restype_order))
    return encoding


def initial_data_from_seqs(seqs:list):
    """
    Processes raw sequences from an A3M file to extract initial feature representations.

    Args:
        seqs: A list of amino acid sequences loaded from the A3M file.
              Sequences are represented with single-letter amino acid codes.
              Lowercase letters represent deletions.

    Returns:
        A dictionary containing:
            * msa_aatype: A PyTorch tensor of one-hot encoded amino acid sequences
                  of shape (N_seq, N_res, 22), where N_seq is the number of unique
                  sequences (with deletions removed) and N_res is the length of the sequences.
                  The dimension 22 corresponds to the 20 amino acids, an unknown amino acid
                  token, and a gap token.
            * msa_deletion_count: A tensor of shape (N_seq, N_res) where
                  each element represents the number of deletions occurring before
                  the corresponding residue in the MSA.
            * aa_distribution: A tensor of shape (N_res, 22) containing the
                  overall amino acid distribution at each residue position
                  across the MSA.
    """

    unique_seqs = []
    deletion_count_matrix = []
    aa_distribution = None

    for seq in seqs:
        delitions_count_seq = []
        aligned_seq = []
        count = 0
        for e in seq:
            if e.islower():
               count +=1
            else:
                delitions_count_seq.append(count)
                count = 0
                aligned_seq.append(e)
        aligned_seq = ''.join(aligned_seq)
        if aligned_seq not in unique_seqs:
            unique_seqs.append(aligned_seq)
            deletion_count_matrix.append(delitions_count_seq)

    deletion_count_matrix = torch.Tensor(deletion_count_matrix)
    unique_seqs = torch.stack([onehot_encode_aa_type(seq, include_gap_token=True).float() for seq in unique_seqs])
    aa_distribution = unique_seqs.mean(dim=0)

    return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_distribution}


def select_cluster_centers(features, max_msa_clusters=512, seed=None):
    """
    Selects representative sequences as cluster centers from the MSA to
    reduce redundancy.

    Args:
        features: A dictionary containing feature representations of the MSA.
        max_msa_clusters: The maximum number of cluster centers to select.
        seed: An optional integer seed for the random number generator.
              Use this to ensure reproducibility.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' and 'msa_deletion_count' features to contain
              data for the cluster centers only.
            * Adding 'extra_msa_aatype' and 'extra_msa_deletion_count' features
              to hold the data for the remaining (non-center) sequences.
    """

    N_seq, N_res = features['msa_aatype'].shape[:2]
    MSA_FEATURE_NAMES = ['msa_aatype', 'msa_deletion_count']
    max_msa_clusters = min(max_msa_clusters, N_seq)

    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)

    idxs = torch.randperm(N_seq - 1, generator=gen) + 1
    target = torch.tensor([0], dtype=torch.long, device=idxs.device)
    #ensure that 0 is present
    shuffled_cluster_idxs = torch.cat((target, idxs)).int()
    idxs_selected = shuffled_cluster_idxs[:max_msa_clusters]
    idxs_extra = shuffled_cluster_idxs[max_msa_clusters:]

    msa_selected = features['msa_aatype'][idxs_selected]
    msa_extra = features['msa_aatype'][idxs_extra]

    deletion_selected = features['msa_deletion_count'][idxs_selected]
    deletion_extra = features['msa_deletion_count'][idxs_extra]

    features['msa_aatype'] = msa_selected
    features['extra_msa_aatype'] = msa_extra
    features['msa_deletion_count'] = deletion_selected
    features['extra_msa_deletion_count'] = deletion_extra
    return features

def mask_cluster_centers(features, mask_probability=0.15, seed=None):
    """
    Introduces random masking in the cluster center sequences for data augmentation.

    This function modifies the 'msa_aatype' feature within the 'features' dictionary to improve
    model robustness in the presence of noisy or missing input data.  Masking is inspired by
    the AlphaFold architecture.

    Args:
        features: A dictionary containing feature representations of the MSA. It is assumed
                  that cluster centers have already been selected.
        mask_probability: The probability of masking out an individual amino acid
                          in a cluster center sequence.
        seed: An optional integer seed for the random number generator.
              Use this to ensure reproducibility.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' feature with masked-out tokens as well as possible
              replacements based on defined probabilities.
            * Creating a copy of the original 'msa_aatype' feature with the key 'true_msa_aatype'.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_aa_categories = 23 # 20 Amino Acids, Unknown AA, Gap, masked_msa_token
    odds = {
        'uniform_replacement': 0.1,
        'replacement_from_distribution': 0.1,
        'no_replacement': 0.1,
        'masked_out': 0.7,
    }
    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)
        torch.manual_seed(seed)

    uniform_replacement = torch.tensor([1/20]*20+[0,0]) * odds['uniform_replacement']
    # replacement_from_distribution has shape (N_res, 22)
    replacement_from_distribution = features['aa_distribution'] * odds['replacement_from_distribution']
    # no_replacement has shape (N_clust, N_res, 22)
    no_replacement = features['msa_aatype'] * odds['no_replacement']
    # masked_out has shape (N_clust, N_res, 1)
    masked_out = torch.ones((N_clust, N_res, 1)) * odds['masked_out']

    uniform_replacement = uniform_replacement[None, None, ...].broadcast_to(no_replacement.shape)
    replacement_from_distribution = replacement_from_distribution[None, ...].broadcast_to(no_replacement.shape)

    categories_without_mask_token = uniform_replacement + replacement_from_distribution + no_replacement
    categories_with_mask_token = torch.cat((categories_without_mask_token, masked_out), dim=-1)
    categories_with_mask_token = categories_with_mask_token.reshape(-1, N_aa_categories)

    replace_with = torch.distributions.Categorical(categories_with_mask_token).sample()
    replace_with = nn.functional.one_hot(replace_with, num_classes=N_aa_categories)
    replace_with = replace_with.reshape(N_clust, N_res, N_aa_categories)
    replace_with = replace_with.float()

    replace_mask = torch.rand((N_clust, N_res), generator=gen) < mask_probability

    features['true_msa_aatype'] = features['msa_aatype'].clone()
    aatype_padding = torch.zeros((N_clust, N_res, 1))
    features['msa_aatype'] = torch.cat((features['msa_aatype'], aatype_padding), dim=-1)
    features['msa_aatype'][replace_mask] = replace_with[replace_mask]
    return features 


def cluster_assignment(features):
    """
    Assigns sequences in the extra MSA to their closest cluster centers based on Hamming distance.

    Args:
        features: A dictionary containing feature representations of the MSA.
                  It is assumed that cluster centers have already been selected.

    Returns:
        The updated 'features' dictionary with the following additions:
            * cluster_assignment:  A tensor of shape (N_extra,) containing the indices
                                  of the assigned cluster centers for each extra sequence.
            * cluster_assignment_counts: A tensor of shape (N_clust,)  where each element indicates
                                        the number of extra sequences assigned to a cluster center
                                        (excluding the cluster center itself).
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    sliced_msa_aatype = features['msa_aatype'][..., :21]
    sliced_extra_msa_type = features['extra_msa_aatype'][..., :21]
    #c:clust, r = res, e=enc, x=extra,
    agreement_tensor = torch.einsum('cre,xre->cx', sliced_msa_aatype, sliced_extra_msa_type)
    assignment = torch.argmax(agreement_tensor,dim=0)
    features['cluster_assignment'] = assignment

    assignment_counts = torch.bincount(assignment, minlength=N_clust)
    features['cluster_assignment_counts'] = assignment_counts
    return features 


def cluster_average(feature, extra_feature, cluster_assignment, cluster_assignment_count):
    """
    Calculates the average representation of each cluster center by aggregating features
    from the assigned extra sequences.

    Args:
        feature: A tensor containing feature representations for the cluster centers.
                 Shape: (N_clust, N_res, *)
        extra_feature: A tensor containing feature representations for extra sequences.
                       Shape: (N_extra, N_res, *).  The trailing dimensions (*) must
                       be smaller or equal to those of the 'feature' tensor.
        cluster_assignment: A tensor indicating the cluster assignment of each extra sequence.
                            Shape: (N_extra,)
        cluster_assignment_count: A tensor containing the number of extra
                                 sequences assigned to each cluster center.
                                 Shape: (N_clust,)

    Returns:
        A tensor containing the average feature representation for each cluster.
        Shape: (N_clust, N_res, *)
    """
    N_clust, N_res = feature.shape[:2]
    N_extra = extra_feature.shape[0]


    usz_extra_shape = (N_extra,) + (1,) * (extra_feature.dim()-1)
    usz_cluster_shape = (N_clust,) + (1,)*(feature.dim()-1)

    cluster_assignment = cluster_assignment.view(usz_extra_shape).broadcast_to(extra_feature.shape)
    cluster_sum = torch.scatter_add(feature, dim=0, index=cluster_assignment, src=extra_feature)
    cluster_assignment_count = cluster_assignment_count.view(usz_cluster_shape).broadcast_to(feature.shape)
    cluster_average = cluster_sum / (cluster_assignment_count+1)

    return cluster_average


def summarize_clusters(features):
    """
    Calculates cluster summaries by applying cluster averaging to the MSA amino acid
    representations and deletion counts.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Modifies:
        The 'features' dictionary in-place by adding the following:
            * cluster_deletion_mean: Average deletion counts for each cluster center,
                                     scaled for numerical stability.
            * cluster_profile: Average amino acid representations for each cluster center.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]

    cluster_deletion_means = cluster_average(features['msa_deletion_count'], features['extra_msa_deletion_count'],
                          features['cluster_assignment'], features['cluster_assignment_counts'])
    features['cluster_deletion_mean'] = 2/torch.pi * torch.arctan(cluster_deletion_means/3)


    features['cluster_profile'] = cluster_average(features['msa_aatype'], features['extra_msa_aatype'],
                                         features['cluster_assignment'], features['cluster_assignment_counts'])
    return features 


def crop_extra_msa(features, max_extra_msa_count=5120, seed=None):
    """
    Reduces the number of extra sequences in the MSA to a fixed size for computational efficiency.

    Args:
        features: A dictionary containing feature representations of the MSA.
        max_extra_msa_count: The maximum number of extra sequences to retain.
        seed: An optional integer seed for the random number generator.
              Use this to ensure reproducibility.

    Modifies:
        The  'features' dictionary in-place by cropping the following keys to include
        only the first 'max_extra_msa_count' sequences:
            * Any key starting with 'extra_'
    """

    N_extra = features['extra_msa_aatype'].shape[0]
    gen = None
    if seed is not None:
        gen = torch.Generator(features['extra_msa_aatype'].device)
        gen.manual_seed(seed)

    max_extra_msa_count = min(max_extra_msa_count, N_extra)

    idxs = torch.randperm(N_extra, generator=gen)
    sliced_perm = idxs[:max_extra_msa_count]
    for key, value in features.items():
        if 'extra' in key:
            features[key] = value[sliced_perm]
    return features

def calculate_msa_feat(features):
    """
    Prepares the final MSA feature representation for protein structure prediction.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_clust, N_res, 49) representing the final MSA features,
        formed by concatenating processed cluster information and deletion-related values.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    msa_feat = None

    cluster_msa = features['msa_aatype']

    cluster_has_deletion = (features['msa_deletion_count'] > 0).float().unsqueeze(-1)

    cluster_deletion_value = 2/torch.pi * torch.arctan(features['msa_deletion_count'] / 3)
    cluster_deletion_value = cluster_deletion_value.unsqueeze(-1)

    cluster_deletion_mean = features['cluster_deletion_mean'].unsqueeze(-1)
    cluster_profile = features['cluster_profile']

    msa_feat = torch.cat((cluster_msa, cluster_has_deletion, cluster_deletion_value, cluster_profile, cluster_deletion_mean), dim=-1)
    return msa_feat


def calculate_extra_msa_feat(features):
    """
    Prepares the extra MSA feature representation for protein structure prediction.
    This function is similar to 'calculate_msa_feat' but operates on  extra MSA sequences
    and includes padding of extra_msa_aatype to match the shape of msa_aatype.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_extra, N_res, 25) representing the final extra MSA features.
    """

    N_extra, N_res = features['extra_msa_aatype'].shape[:2]
    extra_msa_feat = None

    extra_msa_aatype = features['extra_msa_aatype']
    extra_msa_deletion_count = features['extra_msa_deletion_count']

    extra_msa_has_deletion = (extra_msa_deletion_count > 0).float().unsqueeze(-1)
    extra_msa_deletion_value = 2 / torch.pi * torch.arctan(extra_msa_deletion_count/3)
    extra_msa_deletion_value = extra_msa_deletion_value.unsqueeze(-1)

    extra_pad_t = torch.zeros((N_extra, N_res, 1))
    extra_msa_aatype = torch.cat((extra_msa_aatype, extra_pad_t), -1)
    extra_msa_feat = torch.cat((extra_msa_aatype, extra_msa_has_deletion, extra_msa_deletion_value), -1)

    return extra_msa_feat


def create_features_from_a3m(file_name, seed=None):
    """
    Creates feature representations for an MSA from its A3M file.

    This function orchestrates a sequence of transformations on the raw MSA sequences to
    produce features suitable for protein structure prediction.

    Args:
        file_name: Path to the A3M file containing the MSA sequences.

    Returns:
        A dictionary containing the following feature representations for the MSA:
           * msa_feat: A tensor containing the final MSA feature representation.
           * extra_msa_feat: A tensor containing the final extra MSA feature representation.
           * target_feat: A tensor containing a one-hot encoded representation of the
                          target protein sequence (excluding gaps and masked tokens).
           * residue_index: A tensor containing the residue indices (0, 1, ..., N_res-1).
    """

    msa_feat = None
    extra_msa_feat = None
    target_feat = None
    residue_index = None
    select_clusters_seed = None
    mask_clusters_seed = None
    crop_extra_seed = None
    if seed is not None:
        select_clusters_seed = seed
        mask_clusters_seed = seed+1
        crop_extra_seed = seed+2

    sequences = load_a3m_file(file_name)
    features = initial_data_from_seqs(sequences)
    transformation_list = [lambda x :select_cluster_centers(x, seed=select_clusters_seed),
                           lambda x: mask_cluster_centers(x, seed=mask_clusters_seed),
                           cluster_assignment,
                           summarize_clusters,
                           lambda x : crop_extra_msa(x, seed=crop_extra_seed)]
    for f in transformation_list:
        features = f(features)

    # enforce float
    for key, value in features.items():
        features[key] = value.float()
    #final features
    msa_feat = calculate_msa_feat(features)
    extra_msa_feat = calculate_extra_msa_feat(features)

    # target features and idxs

    target_feat = onehot_encode_aa_type(sequences[0], include_gap_token=False)
    target_feat = target_feat.float()
    residue_index = torch.arange(len(sequences[0]))
    return {
        'msa_feat': msa_feat,
        'extra_msa_feat': extra_msa_feat,
        'target_feat': target_feat,
        'residue_index': residue_index
    }




