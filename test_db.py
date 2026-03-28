import uuid
from typing import List, Optional, Iterable

class Test:
    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[str]] = None) -> List[str]:
        embeddings = []
        urls = []
        text_chunk = []
        for index, text in enumerate(texts):
            chunks, embd = ([(1, 2), (3, 4)], [0.1, 0.2, 0.3])
            urls.append(metadatas[index] if metadatas is not None else None)
            embeddings.append(embd)
            
            text_tokens = []
            for chunk in chunks:
                text_tokens.extend(chunk)
            text_chunk.append(text_tokens)

        print(f"embeddings: {len(embeddings)}, urls: {len(urls)}, text_chunk: {len(text_chunk)}")
        return text_chunk

t = Test()
print(t.add_texts(["hello", "world"], metadatas=["url1", "url2"]))
