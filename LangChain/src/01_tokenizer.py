from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
text = "你好，我是Victor。"
bpe_codes = tokenizer.tokenize(text)
print(bpe_codes)
decoded_result = []
for bpe_code in bpe_codes:
    id = tokenizer.convert_tokens_to_ids(bpe_code)
    decoded = tokenizer.decode([id])
    decoded_result.append(decoded)
print("分词结果：", decoded_result)

token_ids = tokenizer.convert_tokens_to_ids(bpe_codes)
print("向量ID：", token_ids)
count = len(token_ids)
print("Token总数：", count)
print("解码结果：", tokenizer.decode(token_ids))