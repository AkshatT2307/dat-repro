"""Explicit protocols: compact is a compute-reduced adaptation, not paper-exact."""
PROFILES = {
    'paper': dict(steps=50000, crop_size=256, output_size=256, milestones=[25000, 40000]),
    'compact': dict(steps=10000, crop_size=256, output_size=128, milestones=[5000, 8000]),
}


def default_output(method, profile):
    return f'runs/{method}' if profile == 'paper' else f'runs/compact10k_128/{method}'


def validate_resume(previous, current):
    defaults = dict(profile='paper', crop=256, output_size=256, dataset='imagenet', epochs=None, channels=3, noise_max=50/255, device='cuda', samples_per_epoch=50000, validate=False, patience=5, min_delta=1e-5, validation_count=1000, split_seed=2026, split_fingerprint=None, validation_seed=2026)
    for key in ('batch', 'manifest_sha256', 'method', 'seed', 'workers',
                'profile', 'crop', 'output_size', 'milestones', 'dataset', 'epochs', 'channels', 'noise_max', 'device', 'images', 'samples_per_epoch', 'validate', 'patience', 'min_delta', 'validation_count', 'split_seed', 'split_fingerprint', 'validation_seed'):
        if previous.get(key, defaults.get(key)) != current.get(key, defaults.get(key)):
            raise ValueError(f'Resume mismatch: {key}; start a separate run for a different protocol.')


def make_scheduler(optimizer, profile):
    from torch.optim.lr_scheduler import MultiStepLR
    return MultiStepLR(optimizer, PROFILES[profile]['milestones'], gamma=0.5)
