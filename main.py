
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing

# ==========================================
# 1. هایپرپارامترها - بهینه برای داده محدود
# ==========================================
batch_size = 16        # کمتر = regularization بیشتر
block_size = 64        # کوتاه‌تر = یادگیری بهتر با داده کم
max_iters = 20000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 256        # از 192 بیشتر کن (داده بیشتر داری)
n_head = 8          # از 6 بیشتر کن
n_layer = 4         # همین خوبه
dropout = 0.3       # از 0.4 کمتر کن (داده بیشتر = نیاز کمتر به dropout)
patience = 7           # early stopping

torch.manual_seed(1405)

# ==========================================
# 2. داده و Tokenizer
# ==========================================
sample_text = """
توانا بود هر که دانا بود ز دانش دل پیر برنا بود
به نام خداوند جان و خرد کزین برتر اندیشه برنگذرد
خداوند نام و خداوند جای خداوند روزی ده رهنمای
""" * 200  # تکرار بیشتر برای داده بیشتر

if os.path.exists('farsi_data_clean.txt'):
    with open('farsi_data_clean.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # اگه داده کمه، تکرارش کن
    if len(text) < 50000:
        text = text * (50000 // len(text) + 1)
        print(f"داده کم بود، تکرار شد. حجم جدید: {len(text)} کاراکتر")
else:
    text = sample_text
    with open('farsi_data_clean.txt', 'w', encoding='utf-8') as f:
        f.write(text)

print(f"حجم داده: {len(text):,} کاراکتر")

# ساخت BPE Tokenizer
print("در حال ساخت tokenizer...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["farsi_data_clean.txt"],
    vocab_size=6000,   # کوچک‌تر برای داده محدود
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", 1), ("</s>", 2)]
)

tokenizer.save_model(".", "farsi_tokenizer")
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer آماده شد. اندازه vocab: {vocab_size}")

# تبدیل متن به توکن
encoded = tokenizer.encode(text)
data = torch.tensor(encoded.ids, dtype=torch.long)
print(f"تعداد توکن‌ها: {len(data):,}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ==========================================
# 3. معماری
# ==========================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FarsiLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Tied embeddings
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================================
# 4. آموزش با Early Stopping
# ==========================================

model = FarsiLanguageModel().to(device)
param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"تعداد پارامترها: {param_count:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


print("شروع آموزش...")
best_val_loss = float('inf')
patience_counter = 0

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        lr_now = scheduler.get_last_lr()[0]
        print(f"مرحله {iter}: train={losses['train']:.4f}, val={losses['val']:.4f}, lr={lr_now:.2e}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'iter': iter,
                'val_loss': best_val_loss,
                'vocab_size': vocab_size,
            }, 'farsi_model_best.pth')
            print(f"  ✓ مدل بهتر ذخیره شد (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping در مرحله {iter}")
                break

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

print(f"\nآموزش تمام شد. بهترین val_loss: {best_val_loss:.4f}")

# ==========================================
# 5. لود بهترین مدل و تولید متن
# ==========================================

checkpoint = torch.load('farsi_model_best.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
print(f"بهترین مدل از مرحله {checkpoint['iter']} لود شد.")

while True:
    user_input = input("\nپرامپت (bye برای خروج): ")
    if user_input.strip().lower() == 'bye':
        break

    encoded_input = tokenizer.encode(user_input)
    context = torch.tensor([encoded_input.ids], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=150, temperature=0.8, top_k=40)
    print(f"\n{tokenizer.decode(generated[0].tolist())}")
