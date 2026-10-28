from dataclasses import dataclass, replace

@dataclass
class TinyConfig:
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    sequence_len: int = 512
    vocab_size: int = 50257
    dropout: float = 0.1
    window_pattern: str = "SSSL"
    xsa_mode: str = "all"
    num_iterations: int = 2
    iteration_transition_ratio: float = 0.3
    train_backprop_iterations: int = 1
    mtp_predict: int = 3
    mtp_anneal_frac: float = 0.66
    device_batch_size: int = 2
    effective_tokens: int = 65536
    train_tokens: int = 1_000_000
    val_tokens: int = 100_000
    num_epochs: int = 15
    max_steps: int = 0
    matrix_lr: float = 0.04
    scalar_lr: float = 0.25
    embedding_lr: float = 0.15
    unembedding_lr: float = 0.001
    lr_multiplier: float = 0.8
    weight_decay: float = 0.8
    wd_phase1_epoch: float = 2
    wd_phase2_epoch: float = 8
    wd_mid: float = 0.1
    wd_end: float = 0.93
    warmdown_ratio: float = 0.4
    swa_last_epochs: int = 4
    update_ema_every: int = 10
    ema_decay_per_epoch: float = 0.15
    mg_every: int = 1
    mg_step_norm: float = 0.5
    muon_eq_r: bool = False
    precision: str = "auto"
    device: str = "auto"
    use_compile: bool = False
    ema_device: str = "cpu"
    seed: int = 42
    eval_batches: int = 0
    run_dir: str = "runs/tiny_current_colab"

    def validate(self):
        for name in ("n_layer", "n_embd", "n_head", "sequence_len", "vocab_size",
                     "device_batch_size", "effective_tokens", "num_epochs", "train_tokens", "val_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        
        if self.n_embd < 32 or self.n_embd % self.n_head or (self.n_embd // self.n_head) % 4:
            raise ValueError("Require width >= 32 and a head dimension divisible by 4")

        if self.effective_tokens % (self.device_batch_size * self.sequence_len):
            raise ValueError("effective_tokens must divide into whole microbatches")
        
        if self.xsa_mode not in ("off", "first6", "all"):
            raise ValueError("xsa_mode must be off, first6 or all")
        
        if not self.window_pattern or set(self.window_pattern) - set("SL"):
            raise ValueError("window_pattern uses S and L")
        
        if not 1 <= self.mtp_predict <= 4 or self.num_iterations < 1 or self.train_backprop_iterations < 1:
            raise ValueError("Invalid MTP / recurrence count")
        
        if not 0 <= self.iteration_transition_ratio <= 1 or not 0 < self.mtp_anneal_frac <= 1:
            raise ValueError("Invalid schedule fractions")
        
        if not 0 <= self.warmdown_ratio <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid warmdown / dropout")
        
        if not 0 < self.ema_decay_per_epoch < 1 or self.update_ema_every < 0:
            raise ValueError("Invalid EMA settings")
        
        if self.wd_phase1_epoch < 0 or self.wd_phase2_epoch <= self.wd_phase1_epoch:
            raise ValueError("WD phases must increase")
        
        if self.max_steps < 0 or self.eval_batches < 0 or self.mg_every < 0 or self.swa_last_epochs < 0:
            raise ValueError("Counts must be nonnegative")
        
        if self.mg_every and self.effective_tokens // self.sequence_len < 2:
            raise ValueError("Meta-gradient requires at least two sequences per effective batch")
        
        return self

def preset(name="colab_t4", **overrides):
    cfg = TinyConfig()

    if name == "smoke":
        cfg = replace(cfg, n_layer=4, n_embd=128, n_head=4, sequence_len=128,
                      effective_tokens=2048, train_tokens=32768, val_tokens=8192,
                      num_epochs=3, update_ema_every=2, swa_last_epochs=2)
    
    elif name == "colab_l4":
        cfg = replace(cfg, n_layer=8, n_embd=512, n_head=8, device_batch_size=4)
    
    elif name == "colab_a100":
        cfg = replace(cfg, n_layer=10, n_embd=640, n_head=10, sequence_len=1024,
                      device_batch_size=4)
    
    elif name == "upstream_shape":
        cfg = replace(cfg, n_layer=16, n_embd=1024, n_head=8, sequence_len=2048,
                      device_batch_size=32, effective_tokens=524288,
                      train_tokens=100_000_000, val_tokens=10_000_000)
    
    elif name != "colab_t4":
        raise ValueError(f"Unknown preset {name}")
    
    return replace(cfg, **overrides).validate()