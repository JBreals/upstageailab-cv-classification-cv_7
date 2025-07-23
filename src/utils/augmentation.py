import torch
import numpy as np
import random
import torch.nn.functional as F 

def mixup_batch(images, labels, idx1, idx2, alpha):
    if len(idx1) == 0:
        return torch.empty(0), torch.empty(0), []

    lam = torch.tensor(np.random.beta(alpha, alpha, size=len(idx1)), dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(images.device)
    img1, img2 = images[idx1], images[idx2]
    lbl1, lbl2 = labels[idx1], labels[idx2]

    mixed_img = lam * img1 + (1 - lam) * img2
    lam = lam.squeeze()  # shape: (N,) for label mix
    mixed_label = lam.unsqueeze(1) * lbl1 + (1 - lam.unsqueeze(1)) * lbl2
    return mixed_img, mixed_label, ['mixup'] * len(idx1)

def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    if cut_w == 0 or cut_h == 0:
        # 너무 작게 잘리는 경우 전체로 설정 (fail-safe)
        return 0, 0, W, H

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_batch(images, labels, idx1, idx2, alpha):
    if len(idx1) == 0:
        return torch.empty(0), torch.empty(0), []

    img1, img2 = images[idx1], images[idx2]
    lbl1, lbl2 = labels[idx1], labels[idx2]

    lam = np.random.beta(alpha, alpha, size=len(idx1))
    bbxs = [rand_bbox(img1.shape, l) for l in lam]
    mixed_imgs = img1.clone()

    for i, (x1, y1, x2, y2) in enumerate(bbxs):
        mixed_imgs[i, :, y1:y2, x1:x2] = img2[i, :, y1:y2, x1:x2]
        patch_area = (x2 - x1) * (y2 - y1)
        lam_adj = 1 - patch_area / (images.shape[2] * images.shape[3])
        lam[i] = lam_adj

    lam = torch.tensor(lam, dtype=torch.float32).to(images.device)
    mixed_label = lam.unsqueeze(1) * lbl1 + (1 - lam.unsqueeze(1)) * lbl2
    return mixed_imgs, mixed_label, ['cutmix'] * len(idx1)

def select_pair(pool, k):
    if len(pool) >= k:
        return random.sample(pool, k)
    else:
        return random.choices(pool, k=k)

def custom_collate_fn(batch, alpha=1.0, aug_ratio=[0.1, 0.1, 0.8]):
    
    images, labels = zip(*batch)
    images = torch.stack(images)   # (B, C, H, W)
    labels = torch.stack(labels)   # (B, num_classes) — one-hot

    B = images.size(0)
    assert len(aug_ratio) == 3, "aug_ratio는 [mixup, cutmix, none]이어야 함"
    assert abs(sum(aug_ratio) - 1.0) < 1e-5, "aug_ratio의 합은 1이어야 함"

    # 개수 계산
    n_mixup = int(B * aug_ratio[0])
    n_cutmix = int(B * aug_ratio[1])

    # 인덱스 랜덤 분할
    indices = torch.randperm(B)
    mixup_idx = indices[:n_mixup].tolist() if n_mixup > 0 else []
    cutmix_idx = indices[n_mixup:n_mixup + n_cutmix].tolist() if n_cutmix > 0 else []
    none_idx = indices[n_mixup + n_cutmix:].tolist()

    
    mixup_pairs = select_pair(none_idx, len(mixup_idx))
    cutmix_pairs = select_pair(none_idx, len(cutmix_idx))
    
    mixed_images, mixed_labels, mixup_modes = mixup_batch(images, labels, mixup_idx, mixup_pairs, alpha)
    cutmix_images, cutmix_labels, cutmix_modes = cutmix_batch(images, labels, cutmix_idx, cutmix_pairs, alpha)

    none_img = images[none_idx]
    none_label = labels[none_idx]
    none_mode = ['none'] * len(none_idx)

    # 전체 합치기
    all_images = torch.cat([mixed_images, cutmix_images, none_img], dim=0)
    all_labels = torch.cat([mixed_labels, cutmix_labels, none_label], dim=0)
    all_modes = mixup_modes + cutmix_modes + none_mode

    return all_images, all_labels, all_modes


if __name__ == "__main__":
    # 가짜 이미지 데이터 (batch=4, channel=3, height=224, width=224)
    images = torch.randn(32, 3, 224, 224)
    labels = torch.randint(0, 17, (32,))
    one_hot = F.one_hot(labels, num_classes=17).float()

    # batch 형태로 만듦 ([(img, label), ...])
    batch = [(img, label) for img, label in zip(images, one_hot)]

    print("=== mixup 테스트 ===")
    images, labels, modes = custom_collate_fn(batch, aug_ratio=[0.1, 0.1, 0.8], alpha=0.4)
    print(f"mixup mode: {modes}")
    print(f"images shape: {images.shape}, labels shape: {labels.shape}\n")

    print("=== cutmix 테스트 ===")
    images, labels, modes = custom_collate_fn(batch, aug_ratio=[0.1, 0.1, 0.8], alpha=0.4)
    print(f"cutmix mode: {modes}")
    print(f"images shape: {images.shape}, labels shape: {labels.shape}\n")

    print("=== dual 모드 랜덤 적용 테스트 ===")
    for _ in range(3):
        images, labels, modes = custom_collate_fn(batch, aug_ratio=[0.1, 0.1, 0.8], alpha=0.4)
        print(f"dual mode: {modes}")
        print(f"images shape: {images.shape}, labels shape: {labels.shape}\n")