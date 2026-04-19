Persian GPT - مدل تولید متن فارسی با PyTorch
یک مدل زبان Transformer از صفر پیاده‌سازی شده برای تولید متن به زبان فارسی، با قابلیت آموزش روی داده‌های محدود و Early Stopping.

✨ ویژگی‌ها
🚀 پیاده‌سازی کامل معماری Transformer (GPT-like) از صفر

🎯 استفاده از Byte-Level BPE Tokenizer مخصوص زبان فارسی

📊 Early Stopping برای جلوگیری از overfitting

🔧 هایپرپارامترهای بهینه‌شده برای داده‌های محدود

💾 ذخیره خودکار بهترین مدل بر اساس validation loss

🌡️ قابلیت تنظیم Temperature و Top-K Sampling برای تولید متن متنوع

🔄 Weight Tying بین embedding و output layer

📋 پیش‌نیازها
bash
pip install torch tokenizers
🏗️ معماری مدل
پارامتر	مقدار	توضیح
لایه‌ها (n_layer)	4	تعداد بلوک‌های Transformer
هدها (n_head)	8	تعداد توجه‌های چندسر
ابعاد embedding	256	ابعاد برداری توکن‌ها
طول بلوک (block_size)	64	حداکثر طول دنباله
Dropout	0.3	نرخ regularization
Vocab size	6000	اندازه دیکشنری BPE
📁 ساختار فایل‌ها
text
├── farsi_model.py          # کد اصلی مدل و آموزش
├── farsi_tokenizer.json    # فایل tokenizer آموزش دیده
├── farsi_tokenizer-merges.txt
├── farsi_tokenizer-vocab.json
├── farsi_model_best.pth    # بهترین مدل ذخیره شده
└── farsi_data_clean.txt    # داده‌های آموزشی
🚀 نحوه استفاده
1. آماده‌سازی داده
داده فارسی خود را در فایل farsi_data_clean.txt قرار دهید. اگر فایل وجود نداشته باشد، از متن نمونه استفاده می‌شود.

2. اجرای آموزش
bash
python farsi_model.py
مدل به صورت خودکار:

Tokenizer را روی داده شما آموزش می‌دهد

داده را به train/val تقسیم می‌کند (90%/10%)

مدل را با Early Stopping آموزش می‌دهد

بهترین مدل را ذخیره می‌کند

3. تولید متن
بعد از اتمام آموزش، می‌توانید به صورت تعاملی متن تولید کنید:

text
پرامپت (bye برای خروج): در روزهای بارانی
[مدل متن را ادامه می‌دهد...]
4. تنظیمات تولید متن
python
generated = model.generate(
    context, 
    max_new_tokens=150,  # تعداد توکن‌های جدید
    temperature=0.8,     # خلاقیت (کمتر = محافظه‌کارتر)
    top_k=40            # فقط از k توکن برتر نمونه برداری کن
)
📊 آموزش با داده محدود
این کد برای شرایطی که داده کم دارید بهینه شده:

block_size کوتاه (64): یادگیری بهتر الگوهای کوتاه

dropout بالاتر (0.3): کاهش overfitting

batch_size کوچک (16): regularization طبیعی

Early Stopping (patience=7): توقف خودکار

CosineAnnealingLR: نرخ یادگیری نزولی ملایم

افزایش حجم داده
اگر فایل داده شما کوچک است، کد به صورت خودکار آن را تکرار می‌کند.

🧪 نتایج نمونه
python
# پرامپت: "روز خوبی بود و"
# خروجی: روز خوبی بود و هوا آفتابی بود. به پارک رفتم و ...
⚙️ تنظیمات پیشرفته
می‌توانید هایپرپارامترها را در ابتدای کد تغییر دهید:

python
batch_size = 16        # افزایش برای داده بیشتر
block_size = 64        # افزایش برای متن‌های بلندتر
n_embd = 256          # افزایش برای مدل قوی‌تر
n_head = 8            # باید بر n_embd بخش‌پذیر باشد
n_layer = 4           # افزایش برای داده بیشتر
dropout = 0.3         # کاهش اگر داده زیاد دارید
patience = 7          # تعداد مراحل بدون بهبود
📝 توضیحات فنی
Attention Mechanism
Causal self-attention (masked) برای تولید متن دنباله‌ای

Multi-head attention با head_size = n_embd // n_head

Scaled dot-product attention

Positional Encoding
از embedding قابل یادگیری برای موقعیت‌ها استفاده می‌شود

حداکثر طول: block_size

Regularization
Dropout در تمام لایه‌ها

Weight decay در بهینه‌ساز

Gradient clipping

🔄 ذخیره و بارگذاری مدل
python
# ذخیره
torch.save({
    'model': model.state_dict(),
    'iter': iter,
    'val_loss': best_val_loss,
    'vocab_size': vocab_size,
}, 'farsi_model_best.pth')

# بارگذاری
checkpoint = torch.load('farsi_model_best.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
🎯 نکات کلیدی برای زبان فارسی
استفاده از ByteLevelBPETokenizer که با کاراکترهای فارسی به خوبی کار می‌کند

دیکشنری با اندازه 6000 که برای اکثر متون فارسی کافی است

پشتیبانی از راست به چپ در خروجی (بسته به ترمینال)

📈 بهبودهای احتمالی
افزودن Flash Attention برای سرعت بیشتر

پشتیبانی از RoPE positional encoding

افزودن beam search برای تولید بهتر

پشتیبانی از fine-tuning روی داده خاص

📜 مجوز
این پروژه تحت مجوز MIT منتشر شده است.

🤝 مشارکت
کاملاً باز برای مشارکت! لطفاً issue یا pull request خود را ارسال کنید.

ساخته شده با ❤️ برای زبان فارسی

