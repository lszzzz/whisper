import whisper

model = whisper.load_model("base")
# result = model.transcribe("/Users/samlee/Documents/sample/asr/longwav_2.wav")
# print(result["text"])
# print(result["segments"])

result = model.transcribe("/Users/samlee/Documents/sample/asr/asr_example_zh.wav")
print(result)
