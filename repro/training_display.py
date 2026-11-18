"""Plain-text training summaries: readable in terminals, tee output, and log files."""


def duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class TrainingDisplay:
    def __init__(self, config, resumed_step=0):
        self.rows = 0
        method = 'Fast IDR' if config['method'] == 'idr' else 'Supervised U-Net'
        print('\n' + '=' * 108, flush=True)
        print(f"  {method} | {config['gpu']} | {config['precision']}", flush=True)
        print(f"  Samples/epoch: {config['images']:,}   Batch: {config['batch']}   "
              f"Iterations: {config['steps']:,}   Start: {resumed_step:,}   Seed: {config['seed']}", flush=True)
        if config.get('epochs'):
            print(f"  Epoch limit: {config['epochs']} | up to {config['budget']['samples_total']:,} samples", flush=True)
            if config.get('validate'):
                print(f"  Validation: every epoch | early-stop patience {config['patience']} | min delta {config['min_delta']:g}", flush=True)
        milestones = ' and '.join(f'{value:,}' for value in config['milestones'])
        print(f"  Adam | L1 loss | current LR {config.get('effective_start_lr', config['learning_rate']):.1e} | schedule milestones {milestones}", flush=True)
        print(f"  Profile: {config.get('profile', 'paper')} | clean crop {config['crop']}x{config['crop']}"
              f" -> {config.get('output_size', config['crop'])}x{config.get('output_size', config['crop'])} before noise", flush=True)
        print(f"  Output: {config.get('output') or 'runs/' + config['method']} | "
              'metrics.jsonl: every step | console.log: displayed progress', flush=True)
        print('=' * 108, flush=True)

    def progress(self, record):
        if self.rows % 20 == 0:
            print(f"{'Step':>15} {'Done':>7} {'Epoch':>6} {'L1 loss':>11} {'LR':>10} "
                  f"{'s/step':>8} {'img/s':>8} {'Elapsed':>10} {'ETA':>10} {'Peak GiB':>9}", flush=True)
            print('-' * 108, flush=True)
        step, total = record['step'], record['total_steps']
        print(f"{f'{step:,}/{total:,}':>15} {100*step/total:6.1f}% {record['epoch']+1:6d} "
              f"{record['loss']:11.6f} {record['learning_rate']:10.2e} "
              f"{record['seconds_per_step']:8.3f} {record['images_per_second']:8.1f} "
              f"{duration(record['elapsed_seconds']):>10} "
              f"{duration(record['estimated_remaining_hours']*3600):>10} "
              f"{record['gpu_peak_allocated_mib']/1024:9.2f}", flush=True)
        self.rows += 1
