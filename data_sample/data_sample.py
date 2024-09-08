from evaluate import load

reference = "the cat sat on the mat"
prediction = "the cat sit on the"

wer_metric = load("wer")
wer = wer_metric.compute(references=[reference], predictions=[prediction])
print(wer)
