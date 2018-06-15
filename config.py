class Config():
    def __init__(self):
        self.n_embed = 0
        self.d_embed = 0
        self.vocab_size = 15000
        self.max_grad_norm = 5
        self.d_hidden = 300
        self.lr_decay = 0.95
        self.batch_size = 5
        self.lr = 0.001
        self.epoch = 40
        self.steps = 5
        self.fix_embed = False
        self.d_out = 2
        self.padding_idx = 0
        self.log_interval = 50
