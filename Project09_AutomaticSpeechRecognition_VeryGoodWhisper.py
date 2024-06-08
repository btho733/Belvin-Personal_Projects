#  Re-encode Large audio files and compress and use whisper to do ASR 
#  Top Tip :  (This works really well !!!)

# First, uncomment and use the ffmpeg code below to increase transcriptable length by re-encoding and compressing file 
# ffmpeg -i /path/to/file/to/be/compressed.mp3 -vn -map_metadata -1 -ac 1 -c:a libopus -b:a 12k -application voip /path/to/compressedfile/compressed_re-encode.ogg


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

result = pipe("/path/to/Audiofile.ogg",generate_kwargs={"language" : "english"})
print(result["text"])
