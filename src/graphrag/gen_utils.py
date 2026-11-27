import nltk
import re

def initialize_nltk():
  """Ensure the required NLTK tokenizer data is available."""
  try:
    nltk.data.find("tokenizers/punkt_tab")
  except LookupError:
    nltk.download("punkt_tab", quiet=True)

DEF_TARGET_CHUNCK_IN_CHARS = 1200
DEF_MAX_TARGET_CHUNCK_IN_CHARS = 1400
DEF_OVERLAP_SENTENCES = 1

def create_chunks(text, 
                  target_chars=DEF_TARGET_CHUNCK_IN_CHARS, 
                  max_chars=DEF_MAX_TARGET_CHUNCK_IN_CHARS, 
                  overlap_sentences=DEF_OVERLAP_SENTENCES):
  sentences = nltk.sent_tokenize(re.sub(r"\s+", " ", text).strip())
  chunks = []
  start = 0

  while start < len(sentences):
    end = start
    current = ""

    while end < len(sentences):
      candidate = (current + " " + sentences[end]).strip()
      if len(candidate) > max_chars and current:
        break
      current = candidate
      end += 1
      if len(current) >= target_chars:
        break
      
    chunks.append(current)
    if end >= len(sentences):
      break
    # Start the next chunk with overlapping sentences
    start = max(start + 1, end - overlap_sentences)

  return chunks

