import os
import torch
import numpy as np
import faiss
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel

# PDF 파일 목록
pdf_folder = "./static/documents"  # PDF 파일들이 들어있는 폴더
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# PDF 파일에서 텍스트를 추출하는 함수
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# 문단들을 BERT 임베딩으로 변환하는 함수
def embed_text(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS 토큰의 출력을 임베딩으로 사용
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding)
    return embeddings

# FAISS 인덱스 생성 (벡터 차원은 BERT의 출력 크기와 동일하게 설정)
embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)

# 모든 PDF 파일에서 텍스트 추출 및 임베딩 생성 후 저장
document_titles = []

for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)
    paragraphs = text.split('\n\n')  # 문단 단위로 나누기
    document_titles.extend([os.path.basename(pdf_file)] * len(paragraphs))

    # 문단 임베딩 생성
    paragraph_embeddings = embed_text(paragraphs)
    
    # FAISS 인덱스에 벡터 추가
    paragraph_embeddings_np = np.vstack(paragraph_embeddings).astype(np.float32)  # numpy 배열로 변환
    index.add(paragraph_embeddings_np)

# FAISS 인덱스 및 제목 목록 저장
faiss.write_index(index, "document_faiss_index.idx")
np.save("document_titles.npy", np.array(document_titles))
print("FAISS 인덱스와 제목 저장 완료")