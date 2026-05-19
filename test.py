from transformers import pipeline

# 加上 device_map="auto" 自动适配硬件
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="auto")

messages = [
    {"role": "user", "content": "Who are you?"},
]

print("\n" + "="*10 + " 正在让 DeepSeek 思考中，请稍候... " + "="*10)

# 1. 运行并设置 max_new_tokens，防止回答被截断
outputs = pipe(messages, max_new_tokens=256)

print("\n" + "="*10 + " 模型回答如下 " + "="*10)

# 2. 从返回的复杂字典结构中，把最终的文本内容 print 出来
print(outputs[0]["generated_text"][-1]["content"])

print("="*34 + "\n")