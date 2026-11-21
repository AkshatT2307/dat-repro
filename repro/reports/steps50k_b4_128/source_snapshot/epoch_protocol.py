"""Complete-epoch budgets with no dropped training samples."""
import math


def plan(samples, batch, epochs):
    if min(samples, batch, epochs) < 1:
        raise ValueError('samples, batch, epochs must be positive')
    steps_per_epoch = math.ceil(samples / batch)
    total = epochs * steps_per_epoch
    milestones = sorted(set(max(1, math.ceil(f * total)) for f in (0.5, 0.8)))
    return dict(samples_per_epoch=samples, batch=batch, epochs=epochs,
                steps_per_epoch=steps_per_epoch, steps=total,
                samples_total=samples * epochs, milestones=milestones)
