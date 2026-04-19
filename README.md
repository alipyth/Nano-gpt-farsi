# Persian GPT - مدل تولید متن فارسی با PyTorch

یک مدل زبان Transformer از صفر پیاده‌سازی شده برای تولید متن به زبان فارسی، با قابلیت آموزش روی داده‌های محدود و Early Stopping.

## ✨ ویژگی‌ها

- 🚀 پیاده‌سازی کامل معماری Transformer (GPT-like) از صفر
- 🎯 استفاده از **Byte-Level BPE Tokenizer** مخصوص زبان فارسی
- 📊 **Early Stopping** برای جلوگیری از overfitting
- 🔧 هایپرپارامترهای بهینه‌شده برای داده‌های محدود
- 💾 ذخیره خودکار بهترین مدل بر اساس validation loss
- 🌡️ قابلیت تنظیم **Temperature** و **Top-K Sampling** برای تولید متن متنوع
- 🔄 **Weight Tying** بین embedding و output layer

