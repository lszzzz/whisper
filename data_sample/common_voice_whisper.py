import numpy as np

import whisper
import torch
from huggingface_hub import login

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

model = whisper.load_model("small")

login('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')

from datasets import load_dataset, Audio

# common_voice_test = load_dataset(
#     "mozilla-foundation/common_voice_13_0", "dv", split="test[:1%]"
# )

common_voice_test = load_dataset(
    "mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True
)

common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16000))

from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

all_predictions = []

for data in common_voice_test:
    result = model.transcribe(data["audio"]["array"].astype(np.float32))
    print(result)
    all_predictions.append(result["text"])

from evaluate import load

wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=common_voice_test["sentence"], predictions=all_predictions
)
print(wer_ortho)

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

# 计算规范化 WER
all_predictions_norm = [normalizer(pred) for pred in all_predictions]
all_references_norm = [normalizer(label) for label in common_voice_test["sentence"]]

# 过滤掉参考文本被规范化后为零值的样本
all_predictions_norm = [
    all_predictions_norm[i]
    for i in range(len(all_predictions_norm))
    if len(all_references_norm[i]) > 0
]
all_references_norm = [
    all_references_norm[i]
    for i in range(len(all_references_norm))
    if len(all_references_norm[i]) > 0
]

wer = 100 * wer_metric.compute(
    references=all_references_norm, predictions=all_predictions_norm
)

print(wer)
