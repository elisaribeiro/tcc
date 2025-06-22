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
) -> int:
    """
    Baixa os PDFs dos editais fornecidos em uma lista de dicionários JSON.
    Cria o diretório de download se não existir.
    Retorna o número de PDFs baixados com sucesso.
    """
    print(f"\n--- Iniciando o download de PDFs dos editais em JSON ---")

    if not editals_json_list:
        print("Nenhum edital encontrado na lista JSON para baixar PDFs.")
        return 0

    os.makedirs(download_dir, exist_ok=True)
    print(f"DEBUG: Diretório de download '{download_dir}' assegurado.")

    download_count = 0
    failed_downloads = []

    for edital in editals_json_list:
        title = edital.get('title', 'titulo_desconhecido')
        url = edital.get('url') 

        final_title_sanitized = _sanitize_filename(title)
        
        # O download_pdf já lida com URLs inválidas/genéricas
        if not url:
            print(f"AVISO: Edital '{final_title_sanitized}' não possui URL. Pulando download.")
            continue

        file_path = os.path.join(download_dir, f"{final_title_sanitized}.pdf")

        if os.path.exists(file_path):
            print(f"DEBUG: PDF '{final_title_sanitized}.pdf' já existe. Pulando download.")
            download_count += 1
            continue

        print(f"DEBUG: Baixando '{final_title_sanitized}.pdf' de '{url}'...")
        if download_pdf(url, file_path): # Chama a função auxiliar download_pdf
            download_count += 1
        else:
            failed_downloads.append(f"- {final_title_sanitized} (URL: {url})")

    print(f"\n--- Download de PDFs Concluído ---")
    print(f"Total de PDFs baixados: {download_count}")
    if failed_downloads:
        print("Falhas no download:")
        for fail in failed_downloads:
            print(fail)
    else:
        print("Todos os PDFs foram baixados ou já existiam no diretório.")
    
    return download_count


def download_pdfs_from_editals_from_txt(
    txt_file_path: str = "editais_encontrados.txt",
    download_dir: str = "pdfs_baixados"
) -> int:
    """
    Lê a lista de editais de um arquivo TXT formatado para humanos,
    baixa os PDFs e os salva em um diretório especificado.
    Esta função é mais sensível a mudanças no formato do TXT.
    """
    print(f"\n--- Iniciando o download de PDFs dos editais do arquivo TXT ---")

    if not os.path.exists(txt_file_path):
        print(f"AVISO: Arquivo TXT '{txt_file_path}' não encontrado. Execute a opção 2 primeiro para gerá-lo.")
        return 0

    editals_data: List[Dict[str, Any]] = []
    current_edital: Dict[str, Any] = {}
    
    title_re = re.compile(r"^\s*Título:\s*(.*)$")
    agency_re = re.compile(r"^\s*Agência:\s*(.*)$")
    deadline_re = re.compile(r"^\s*Prazo Final:\s*(.*)$")
    url_re = re.compile(r"^\s*URL:\s*(.*)$")
    number_re = re.compile(r"^\s*Número:\s*(.*)$")

    line_num = 0
    try:
        with open(txt_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()

                if stripped_line.startswith("Edital "):
                    if current_edital:
                        editals_data.append(current_edital)
                    current_edital = {}
                    continue

                match_title = title_re.match(stripped_line)
                if match_title:
                    current_edital['title'] = match_title.group(1).strip()
                    continue
                
                match_agency = agency_re.match(stripped_line)
                if match_agency:
                    current_edital['agency'] = match_agency.group(1).strip()
                    continue

                match_deadline = deadline_re.match(stripped_line)
                if match_deadline:
                    current_edital['deadline'] = match_deadline.group(1).strip()
                    continue

                match_url = url_re.match(stripped_line)
                if match_url:
                    current_edital['url'] = match_url.group(1).strip()
                    continue

                match_number = number_re.match(stripped_line)
                if match_number:
                    current_edital['number'] = match_number.group(1).strip()
                    continue

                if re.match(r"^-+$", stripped_line): # Procura por uma linha com apenas hifens
                    if current_edital:
                        editals_data.append(current_edital)
                    current_edital = {}
                    continue

        if current_edital:
            editals_data.append(current_edital)

        print(f"DEBUG: Parseados {len(editals_data)} editais do arquivo TXT para download.")

    except Exception as e:
        print(f"ERRO: Ocorreu um erro ao ler ou parsear o arquivo TXT na linha {line_num}: {e}. Não foi possível baixar PDFs.")
        return 0

    if not editals_data:
        print("Nenhum edital encontrado no arquivo TXT para baixar PDFs.")
        return 0

    os.makedirs(download_dir, exist_ok=True)
    print(f"DEBUG: Diretório de download '{download_dir}' assegurado.")

    download_count = 0
    failed_downloads = []

    for edital in editals_data:
        title = edital.get('title', 'titulo_desconhecido')
        url = edital.get('url') 

        final_title_sanitized = _sanitize_filename(title)
        
        if not url or url.lower() == 'url desconhecida':
            print(f"AVISO: Edital '{final_title_sanitized}' não possui URL válida. Pulando.")
            continue

        file_path = os.path.join(download_dir, f"{final_title_sanitized}.pdf")

        if os.path.exists(file_path):
            print(f"DEBUG: PDF '{final_title_sanitized}.pdf' já existe. Pulando download.")
            download_count += 1
            continue

        print(f"DEBUG: Baixando '{final_title_sanitized}.pdf' de '{url}'...")
        if download_pdf(url, file_path):
            download_count += 1
        else:
            failed_downloads.append(f"{final_title_sanitized} (URL: {url})")

    print(f"\n--- Download de PDFs Concluído ---")
    print(f"Total de PDFs processados (baixados ou já existentes): {download_count}")
    if failed_downloads:
        print("Falhas no download:")
        for fail in failed_downloads:
            print(f"- {fail}")
    else:
        print("Todos os PDFs foram baixados ou já existiam no diretório.")
    
    return download_count