import tiktoken
from chatgpt import ChatGPT


class TokenizeChunker:
    def __init__(self):
        pass
    def tokenize_and_chunk(self, text, chunk_size=1000, max_tokens = 150000):
        """Tokenize and chunk text into chunks of size chunk_size."""
        # Get the encoding for the text.
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = len(encoding.encode(text, disallowed_special=()))
        if total_tokens > max_tokens:
            return ["The file is too large to analyze"]
        chunks = []
        current_chunk = ""
        current_token_count = 0
        print(text)
        for line in text.split('\n'):
            line_tokens = len(encoding.encode(line, disallowed_special=()))
            if current_token_count + line_tokens <= chunk_size:
                current_chunk += line + '\n'
                current_token_count += line_tokens
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
                current_token_count = line_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def summarize_chunks(self, chunks):
        """Summarize chunks 10 at a time with gpt3"""
        gpt = ChatGPT()
        # Send 10 chunks at a time to gpt3 for summarization
        # example: 10chunks = gpt.chat_with_gpt3("Summarize 10 chunks of text:", chunks)
        summaries = []
        for i in range(0, len(chunks), 10):
            ten_chunks = chunks[i:i+10]
            ten_chunks_str = " ".join(ten_chunks)
            summary = gpt.chat_with_gpt3("Summarize 10 chunks of text:", ten_chunks_str)
            summaries.append(summary)
        return summaries