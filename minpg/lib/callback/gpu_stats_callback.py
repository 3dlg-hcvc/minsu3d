from pytorch_lightning.callbacks import GPUStatsMonitor


def init_gpu_stats_monitor():
    monitor = GPUStatsMonitor(
        memory_utilization=True,
        gpu_utilization=True,
        fan_speed=True,
        temperature=True
    )
    return monitor
