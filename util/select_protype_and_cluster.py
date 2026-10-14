import os

import einops

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def compute_class_prototypes_fast(
        features,
        labels,
        num_clusters=10,
        batch_size=4096):

    features = features.cpu()
    labels = labels.cpu()

    unique_labels = torch.unique(labels)
    unique_labels, _ = torch.sort(unique_labels)

    prototypes = []

    for c in unique_labels:

        print(f"\nProcessing class {c.item()}")

        mask = labels == c
        class_feats_origin = features[mask]   # [Nc, D]
        class_feats = torch.mean(einops.rearrange(class_feats_origin, 'b d n m -> b (n m) d'), dim=1)

        Nc = class_feats.shape[0]

        if Nc < num_clusters:
            raise ValueError(f"class {c} has fewer samples than clusters")

        class_feats_norm = F.normalize(class_feats, dim=1)

        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=batch_size,
            random_state=0,
            n_init=3,
            max_iter=100
        )

        cluster_ids = kmeans.fit_predict(class_feats_norm.numpy())

        cluster_centers = []

        for k in range(num_clusters):
            cluster_mask = cluster_ids == k

            if cluster_mask.sum() == 0:
                assert 0
            else:
                center = class_feats_origin[cluster_mask].mean(dim=0)

            cluster_centers.append(center)

        cluster_centers = torch.stack(cluster_centers)  # [K, D]

        prototypes.append(cluster_centers)

    prototypes = torch.stack(prototypes)  # [N_class, K, D]

    return prototypes

def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    B, N = logits.shape
    assert labels.ndim == 1 and labels.shape[0] == B
    assert labels.dtype == torch.long or labels.dtype == torch.int64
    preds = logits.argmax(dim=1)  # [B]
    correct = (preds == labels)  # [B], bool

    num_classes = N

    # one-hot labels: [B, N]
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # [B, N]
    correct_one_hot = labels_one_hot * correct.unsqueeze(1).float()  # [B, N]

    total_per_class = labels_one_hot.sum(dim=0)  # [N]
    correct_per_class = correct_one_hot.sum(dim=0)  # [N]
    acc_per_class = torch.zeros_like(total_per_class)

    nonzero = total_per_class > 0
    acc_per_class[nonzero] = correct_per_class[nonzero] / total_per_class[nonzero]

    return acc_per_class  # shape [N]
dir = 'work_dir/faster_rcnn_increase/NS-DS/SCL-IOD/nighttimeSunny1_4/prototype'
roi_features = torch.load(dir + '/protype.pth').cpu()
roi_labels = torch.load(dir + '/protype_label.pth').cpu()

prototypes_cluster = compute_class_prototypes_fast(roi_features, roi_labels)
torch.save(prototypes_cluster, dir + "/class_centroids_10_per_class.pt")


