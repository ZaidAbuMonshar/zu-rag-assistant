from FlagEmbedding import BGEM3FlagModel

# First run downloads the model (~1.1GB) and caches it locally
# in ~/.cache/huggingface — subsequent runs load from disk, no re-download
print("Loading BGE-M3... (first run will download ~1.1GB, be patient)")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # use_fp16 = faster, negligible quality loss

# Two Arabic sentences with related meaning, to sanity-check semantic similarity
sentences = [
    "ما هي متطلبات القبول في تخصص علوم الحاسوب؟",   # "What are the admission requirements for CS?"
    "الأوراق المطلوبة للتسجيل في كلية علوم الحاسوب"    # "Documents required to register in the CS college"
]

output = model.encode(sentences, return_dense=True)
embeddings = output['dense_vecs']

print(f"\nEmbedding shape: {embeddings.shape}")   # expect (2, 1024)
print(f"Embedding dtype: {embeddings.dtype}")

# Cosine similarity between the two — should be reasonably high (>0.5-0.6)
# since they're semantically related despite different wording
similarity = embeddings[0] @ embeddings[1].T
print(f"\nSimilarity between the two sentences: {similarity:.4f}")