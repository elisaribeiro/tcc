# pdf_indexer.py
import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import io
from typing import Union, List, Sequence 
from edital_manager import load_cached_grants
import unicodedata

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def safe_str_strip(value):
    if value is None:
        return ""
    return str(value).strip()

def clean_cell(cell):
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").replace("\r", " ").strip()

def _normalize_filename_for_match(filename):
    # Remove extensão, normaliza acentuação, minúsculas, remove espaços e pontuação
    name = os.path.splitext(filename)[0]
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII').lower()
    name = ''.join(c for c in name if c.isalnum())
    return name

def _find_edital_metadata_for_file(file_name, grants):
    file_norm = _normalize_filename_for_match(file_name)
    for edital in grants:
        title_norm = _normalize_filename_for_match(edital.get('title', ''))
        if file_norm == title_norm or file_norm in title_norm or title_norm in file_norm:
            return {
                'title': edital.get('title', ''),
                'agency': edital.get('agency', ''),
                'deadline': edital.get('deadline', ''),
                'url': edital.get('url', '')
            }
    return {}

def process_pdfs_into_documents(pdf_sources: Union[str, Sequence[Union[str, io.BytesIO]]]) -> List[Document]:
    """
    Processa arquivos PDF de um diretório OU uma lista de caminhos de arquivo OU uma lista de objetos BytesIO.
    Extrai texto e tabelas (cronogramas), e retorna uma lista de objetos Document para indexação.

    Args:
        pdf_sources: Um caminho de diretório (str) ou uma sequência de caminhos de arquivo (Sequence[str])
                     ou uma sequência de objetos BytesIO (Sequence[io.BytesIO]).
    """
    all_documents_for_indexing: List[Document] = []
    
    files_to_process = [] # Lista de tuplas (nome_do_arquivo, objeto_arquivo_BytesIO)

    if isinstance(pdf_sources, str): # Se for um caminho de diretório (str)
        print(f"\n--- Iniciando processamento de PDFs do diretório: {pdf_sources} ---")
        pdf_paths = [os.path.join(pdf_sources, f) for f in os.listdir(pdf_sources) if f.endswith(".pdf")]
        for path in pdf_paths:
            files_to_process.append((os.path.basename(path), open(path, 'rb'))) # Abre o arquivo do disco
        
    elif isinstance(pdf_sources, (list, tuple)): # Se for uma lista/tupla (Sequence)
        print("\n--- Iniciando processamento de PDFs da lista fornecida ---")
        for item in pdf_sources:
            if isinstance(item, str): # Se o item é um CAMINHO DE ARQUIVO (str)
                files_to_process.append((os.path.basename(item), open(item, 'rb')))
            elif isinstance(item, io.BytesIO): # Se o item é um BytesIO
                file_name = getattr(item, 'name', f"uploaded_file_{len(files_to_process)}.pdf")
                files_to_process.append((file_name, item))
            else:
                raise TypeError(f"Tipo de item de PDF não suportado na lista: {type(item)}")
    else:
        raise ValueError("pdf_sources deve ser um caminho de diretório (str) ou uma lista/tupla de caminhos/BytesIO.")

    grants = load_cached_grants()

    for file_name, file_obj in files_to_process:
        edital_meta = _find_edital_metadata_for_file(file_name, grants)
        try:
            with pdfplumber.open(file_obj) as pdf:
                for i, page in enumerate(pdf.pages):
                    chunk_id_base = f"{file_name.replace('.', '_')}_page_{i+1}"
                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    # Excluir páginas institucionais (case-insensitive, ignora acentuação, busca palavra inteira)
                    def normalize(text):
                        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').lower()
                    normalized_text = normalize(text)
                    exclude_terms = [
                        "sumario", "indice", "expediente", "apresentacao", "carta",
                        "ouvidoria", "endereco", "contato", "biblioteca",
                        "lei", "presidente", "diretor", "conselho",
                        "sic", "fala.sp.gov.br", "canais", "rua", "Práticas"
                    ]
                    import re
                    pattern = r'\\b(' + '|'.join(re.escape(term) for term in exclude_terms) + r')\\b'
                    if re.search(pattern, normalized_text):
                        print(f"Página {i+1} do arquivo {file_name} descartada por conter termo institucional (regex palavra inteira).")
                        continue

                    # Função utilitária para checar termos institucionais em qualquer texto
                    def chunk_has_excluded_term(text):
                        normalized = normalize(text)
                        return re.search(pattern, normalized)

                    # Só indexe se contiver palavras-chave de edital (lista ampliada)
                    edital_keywords = [
                        "edital", "chamada", "chamada pública", "propostas", "inscrições", "submissão", "financiamento", "bolsa",
                        "projeto", "fomento", "pesquisa", "seleção", "resultado", "cronograma", "objetivo",
                        "valor", "recurso", "vigência", "anexo", "regulamento", "critério", "apresentação de propostas",
                        "funding", "deadline", "apoio", "concessão", "proponente", "instituição executora", "instituição parceira",
                        "contrapartida", "documentação", "requisitos", "submissão de propostas", "proponente responsável",
                        "área temática", "área de conhecimento", "projetos contemplados", "projetos aprovados", "projetos selecionados",
                        "cronograma de atividades", "cronograma de execução", "cronograma financeiro", "recursos financeiros",
                        "valor global", "valor total", "valor financiado", "vigência do projeto", "vigência da bolsa", "vigência do edital"
                    ]
                    if not any(word in text.lower() for word in edital_keywords):
                        print(f"Página {i+1} do arquivo {file_name} descartada por não conter palavras-chave de edital.")
                        continue

                    # Se a página contém alguma palavra-chave de edital, indexe o texto inteiro da página
                    if any(word in text.lower() for word in edital_keywords):
                        if chunk_has_excluded_term(text):
                            print(f"Página {i+1} do arquivo {file_name} (chunk inteiro) descartada por conter termo institucional.")
                            continue
                        meta = {"source": file_name, "page": i + 1, "type": "page_relevante"}
                        meta.update(edital_meta)
                        all_documents_for_indexing.append(Document(
                            page_content=text,
                            metadata=meta,
                            id=f"{chunk_id_base}_page_relevante"
                        ))
                        continue

                    # --- NOVO: Priorize seções relevantes do edital ---
                    # Se encontrar uma seção que começa com palavras-chave típicas de edital, priorize esse trecho
                    section_keywords = [
                        "objetivo", "finalidade", "propostas", "inscrições", "submissão", "cronograma", "prazo", "valor", "recurso", "financiamento", "bolsa", "seleção", "resultado", "vigência", "anexo", "regulamento", "critério", "apresentação de propostas"
                    ]
                    relevant_sections = []
                    for line in text.split('\n'):
                        if any(line.lower().strip().startswith(kw) for kw in section_keywords):
                            relevant_sections.append(line.strip())
                    # Se encontrar seções relevantes, indexe apenas elas
                    if relevant_sections:
                        for section in relevant_sections:
                            if chunk_has_excluded_term(section):
                                print(f"Seção relevante da página {i+1} do arquivo {file_name} descartada por conter termo institucional.")
                                continue
                            meta = {"source": file_name, "page": i + 1, "type": "section_relevante"}
                            meta.update(edital_meta)
                            all_documents_for_indexing.append(Document(
                                page_content=section,
                                metadata=meta,
                                id=f"{chunk_id_base}_section_{section_keywords[0]}"
                            ))
                        continue

                    # --- Aprimora heurística: sempre indexe seções com termos de elegibilidade, modalidades, requisitos, apoio ---
                    prioridade_keywords = [
                        "elegibilidade", "quem pode participar", "requisitos", "modalidade de apoio", "modalidades de apoio", "financiamento", "submissão", "participação", "condições", "critério de participação", "critério de elegibilidade", "proponente", "instituição executora", "instituição parceira", "expression of interest", "EOI", "horizon europe", "NSF", "ANR", "colaboração internacional"
                    ]
                    for line in text.split('\n'):
                        for kw in prioridade_keywords:
                            if kw in line.lower():
                                meta_prior = {"source": file_name, "page": i + 1, "type": "prioridade", "keyword": kw}
                                meta_prior.update(edital_meta)
                                all_documents_for_indexing.append(Document(
                                    page_content=line.strip(),
                                    metadata=meta_prior,
                                    id=f"{chunk_id_base}_prioridade_{kw}"
                                ))
                                break

                    structured_cronograma_content = ""
                    found_main_cronograma_table = False
                    
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        headers_row = table[0]
                        cleaned_headers = [
                            cleaned_cell for h in headers_row if h is not None and (cleaned_cell := str(h).replace("\n", " ").replace("\r", " ").strip())
                        ]
                        
                        if "ETAPAS" in cleaned_headers and "DATAS" in cleaned_headers:
                            found_main_cronograma_table = True
                            structured_cronograma_content += "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES:\n"
                            
                            for row_idx, row in enumerate(table):
                                if row_idx == 0:
                                    continue
                                if len(row) < 2 or row[0] is None or row[1] is None:
                                    continue

                                item = str(row[0]).replace("\n", " ").replace("\r", " ").strip()
                                value = str(row[1]).replace("\n", " ").replace("\r", " ").strip()
                                
                                structured_cronograma_content += f"O evento '{item}' tem a data ou período de {value}.\n"
                    
                    # Crie um ID único para cada chunk para evitar duplicatas no Chroma
                    
                    if found_main_cronograma_table and structured_cronograma_content:
                        if chunk_has_excluded_term(structured_cronograma_content):
                            print(f"Chunk de cronograma da página {i+1} do arquivo {file_name} descartado por conter termo institucional.")
                        else:
                            meta = {"source": file_name, "page": i + 1, "type": "cronograma_principal"}
                            meta.update(edital_meta)
                            all_documents_for_indexing.append(Document(
                                page_content=structured_cronograma_content,
                                metadata=meta,
                                id=f"{chunk_id_base}_cronograma"
                            ))
                    elif text and not found_main_cronograma_table:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        text_chunks = text_splitter.split_text(text)
                        for chunk_idx, txt_chunk in enumerate(text_chunks):
                            if chunk_has_excluded_term(txt_chunk):
                                print(f"Chunk {chunk_idx} da página {i+1} do arquivo {file_name} descartado por conter termo institucional.")
                                continue
                            meta = {"source": file_name, "page": i + 1, "type": "page_text", "chunk_idx": chunk_idx}
                            meta.update(edital_meta)
                            all_documents_for_indexing.append(Document(
                                page_content=txt_chunk,
                                metadata=meta,
                                id=f"{chunk_id_base}_text_{chunk_idx}"
                            ))
            
            print(f"PDF '{file_name}' processado.")
        except Exception as e:
            print(f"Erro ao processar o arquivo '{file_name}': {e}")
            continue
        finally:
            # Apenas feche o file_obj se ele foi aberto por nós (do diretório local ou de um caminho str)
            # Objetos BytesIO passados como lista não precisam ser fechados aqui.
            if isinstance(pdf_sources, str) or (isinstance(pdf_sources, (list, tuple)) and all(isinstance(item, str) for item in pdf_sources)):
                file_obj.close()


    text_splitter_general = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks: List[Document] = text_splitter_general.split_documents(all_documents_for_indexing)

    print(f"✅ PDFs processados. Gerados {len(chunks)} chunks.")
    return chunks