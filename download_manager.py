# download_manager.py
import os
import requests
import re
import json
from urllib.parse import urljoin, urlparse
import certifi # Para garantir a verificação SSL
from typing import List, Dict, Any, Union

from bs4 import BeautifulSoup # Importar BeautifulSoup (pip install beautifulsoup4)

# --- Funções para Download de PDF ---

# Função auxiliar para sanitizar nomes de arquivos
def _sanitize_filename(title: str) -> str:
    """Remove caracteres inválidos de nome de arquivo."""
    return re.sub(r'[\\/:*?"<>|]', '', title).strip()[:150] # Trunca para evitar nomes muito longos

def download_pdf(url: str, filename: str) -> bool:
    """
    Baixa um PDF de uma URL para um arquivo.
    Retorna True em caso de sucesso, False em caso de falha.
    """
    # Adicionando tratamento para URLs que podem ser 'Link Permanente' ou vazias
    if not url or url.lower() == 'link permanente' or url.lower() == 'url desconhecida':
        print(f"AVISO: URL inválida ou genérica para '{filename}'. Pulando download.")
        return False

    original_url = url # Guardar a URL original para logs
    if not url.startswith('http'):
        print(f"AVISO: URL '{url}' não é absoluta. Tentando resolver...")
        # Heurísticas para resolver URLs relativas (se o LLM as extrair)
        inferred_base_url = None
        # Tentar usar o domínio do site principal da agência se o link for relativo
        # Priorize os domínios mais comuns para editais
        if "cnpq.br" in original_url: inferred_base_url = "https://www.gov.br/cnpq/pt-br/"
        elif "capes.gov.br" in original_url: inferred_base_url = "https://www.gov.br/capes/pt-br/"
        elif "finep.gov.br" in original_url: inferred_base_url = "https://www.finep.gov.br/"
        elif "fapesp.br" in original_url: inferred_base_url = "https://fapesp.br/"

        if inferred_base_url:
            url = urljoin(inferred_base_url, url)
            print(f"DEBUG: URL relativa resolvida para: {url}")
        else:
            print(f"AVISO: Não foi possível inferir a URL base para '{filename}' com link relativo '{original_url}'. Pulando.")
            return False

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Adiciona verify=certifi.where() para usar o bundle de certificados mais atualizado
        # Isso ajuda a resolver SSLError
        response = requests.get(url, stream=True, timeout=60, headers=headers, verify=certifi.where())
        response.raise_for_status() # Lança um erro para status HTTP 4xx/5xx

        content_type = response.headers.get('Content-Type', '').lower()

        if 'application/pdf' in content_type:
            # Se já é um PDF direto, baixa
            with open(filename, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)
            print(f"INFO: PDF '{filename}' baixado com sucesso de {url}.")
            return True
        elif 'text/html' in content_type:
            # Se é uma página HTML, tenta encontrar links para PDF dentro dela
            print(f"DEBUG: URL '{url}' é uma página HTML. Procurando links PDF...")
            pdf_links_on_page = _find_pdf_links_on_page(response.text, url)
            
            if pdf_links_on_page:
                print(f"DEBUG: Encontrado(s) {len(pdf_links_on_page)} link(s) PDF na página. Tentando baixar o primeiro...")
                # Tenta baixar o primeiro PDF encontrado na página
                for pdf_link in pdf_links_on_page:
                    try:
                        pdf_response = requests.get(pdf_link, stream=True, timeout=60, headers=headers, verify=certifi.where())
                        pdf_response.raise_for_status()
                        if 'application/pdf' in pdf_response.headers.get('Content-Type', '').lower():
                            with open(filename, 'wb') as pdf_file:
                                for chunk in pdf_response.iter_content(chunk_size=8192):
                                    pdf_file.write(chunk)
                            print(f"INFO: PDF '{filename}' baixado com sucesso de {pdf_link} (encontrado na página).")
                            return True
                        else:
                            print(f"AVISO: Link '{pdf_link}' não é PDF. (Content-Type: {pdf_response.headers.get('Content-Type','')}).")
                    except requests.exceptions.RequestException as e:
                        print(f"ERRO: Falha ao baixar PDF do link '{pdf_link}' (na página {url}): {e}")
                print(f"AVISO: Nenhum PDF válido pôde ser baixado dos links encontrados na página {url}.")
                return False # Falhou em baixar dos links da página
            else:
                print(f"AVISO: Nenhum link PDF encontrado na página HTML: {url}. Pulando download de {filename}.")
                return False
        else:
            print(f"AVISO: URL '{url}' tem Content-Type desconhecido: {content_type}. Pulando download de {filename}.")
            return False

    except requests.exceptions.SSLError as ssl_err:
        print(f"ERRO SSL: Falha ao baixar '{filename}' de '{url}': {ssl_err}")
        print("Causa provável: Problema na verificação do certificado SSL. Tente rodar 'Install Certificates.command' no seu ambiente Python.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"ERRO: Falha ao baixar '{filename}' de '{url}': {e}")
        return False
    except Exception as e:
        print(f"ERRO: Erro inesperado ao processar '{filename}' de '{url}': {e}")
        return False

def _find_pdf_links_on_page(html_content: str, base_url: str) -> List[str]:
    """
    Analisa o conteúdo HTML de uma página e procura por links para arquivos PDF.
    Retorna uma lista de URLs absolutas de PDFs.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    pdf_links = []
    
    # Procurar por links <a href="..."> que contenham ".pdf"
    from bs4 import Tag
    for a_tag in soup.find_all('a', href=True):
        if isinstance(a_tag, Tag):
            href = a_tag.get('href')
            if isinstance(href, str) and href.lower().endswith('.pdf'):
                absolute_url = urljoin(base_url, href)
                pdf_links.append(absolute_url)
    
    # Você pode adicionar mais heurísticas aqui, como procurar por links em <div>s com "download"
    # ou links que o texto seja "baixe o edital" etc., mas isso seria mais complexo.
    
    return list(set(pdf_links)) # Retorna links únicos


def download_pdfs_from_editals_json(
    editals_json_list: List[Dict[str, Any]], # Recebe uma lista de dicionários (JSON parseado)
    download_dir: str = "pdfs_baixados"
) -> List[str]: # <<< CORREÇÃO AQUI: O tipo de retorno deve ser List[str]
    """
    Baixa os PDFs dos editais fornecidos em uma lista de dicionários JSON.
    Cria o diretório de download se não existir.
    Retorna a lista dos caminhos dos PDFs baixados com sucesso.
    """
    print(f"\n--- Iniciando o download de PDFs dos editais em JSON ---")

    if not editals_json_list:
        print("Nenhum edital encontrado na lista JSON para baixar PDFs.")
        return [] # Retorna uma lista vazia se não houver editais

    os.makedirs(download_dir, exist_ok=True)
    print(f"DEBUG: Diretório de download '{download_dir}' assegurado.")

    successful_downloads_paths: List[str] = [] # Garante tipagem explícita para o Pylance
    # failed_downloads (removido, pois não é retornado pela função)

    for edital in editals_json_list:
        title = edital.get('title', 'titulo_desconhecido')
        url = edital.get('url') 

        final_title_sanitized = _sanitize_filename(title)
        
        if not url or url.lower() == 'link permanente' or url.lower() == 'url desconhecida':
            print(f"AVISO: Edital '{final_title_sanitized}' não possui URL válida. Pulando download.")
            continue

        file_path = os.path.join(download_dir, f"{final_title_sanitized}.pdf")

        if os.path.exists(file_path):
            print(f"DEBUG: PDF '{final_title_sanitized}.pdf' já existe. Pulando download.")
            successful_downloads_paths.append(file_path) # Adiciona à lista de sucessos, mesmo se já existia
            continue

        print(f"DEBUG: Baixando '{final_title_sanitized}.pdf' de '{url}'...")
        if download_pdf(url, file_path):
            successful_downloads_paths.append(file_path) # Adiciona o caminho do novo PDF baixado
        else:
            # Não é mais necessário adicionar a uma lista de falhas interna, 
            # pois a função retorna apenas os caminhos dos sucessos.
            pass

    print(f"\n--- Download de PDFs Concluído ---")
    print(f"Total de PDFs baixados (nesta execução): {len(successful_downloads_paths)}")
    # A lista de falhas não é mais impressa aqui, pois a função agora retorna a lista de caminhos.
    # Se quiser as falhas, a função download_pdf já imprime o ERRO/AVISO individualmente.
    
    return successful_downloads_paths 