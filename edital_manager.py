# edital_manager.py (Ajuste das funções de prazo)

import json
import os
from datetime import datetime, timedelta
import re

CACHE_FILE = "cached_online_grants.json"

def _parse_deadline(deadline_str: str) -> list[datetime]:
    """
    Tenta parsear uma string de prazo em uma ou mais datas datetime.
    Retorna uma lista de objetos datetime. Lida com múltiplos formatos e "sempre aberto".
    """
    parsed_dates = []
    current_year = datetime.now().year # Usar o ano atual para prazos sem ano
    
    # Normalizar e verificar termos de "sempre aberto"
    normalized_str = deadline_str.lower().strip()
    always_open_keywords = ['n/a', 'not specified', 'fluxo contínuo', 'aberta o ano todo', 'não especificado', 'null']
    if any(keyword in normalized_str for keyword in always_open_keywords):
        # Para "sempre aberto", definimos uma data muito distante no futuro para fácil comparação
        parsed_dates.append(datetime.max - timedelta(days=1))
        return parsed_dates # Retorna imediatamente se for "sempre aberto"

    # Mapeamento de nomes de meses para números
    month_map = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6,
        'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    # Regex mais abrangente para encontrar todas as datas possíveis na string
    # Procura por DD/MM/YYYY ou Mes DD, YYYY ou DD de Mes [de] YYYY ou Mes AAAA
    # Considera múltiplas datas separadas por vírgulas, parênteses, "e", etc.
    date_patterns_re = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{4})' + # DD/MM/YYYY
        r'|(\d{1,2})\s+(?:de\s+)?([a-zçãõ]+)(?:\s+de\s+(\d{4}))?' + # DD de Mês [de] YYYY
        r'|([a-zçãõ]+)\s+(\d{4})' + # Mês AAAA
        r'|([a-z]+)\s+(\d{1,2}),?\s*(\d{4})', # Mês DD, YYYY (inglês)
        re.IGNORECASE | re.DOTALL
    )
    
    # Iterar sobre todas as ocorrências de datas na string (lidando com múltiplos formatos/ciclos)
    potential_date_strings = []
    # Usar finditer para pegar todos os grupos
    for match in date_patterns_re.finditer(normalized_str):
        # match.groups() retorna uma tupla com todos os grupos capturados.
        # Filtramos None e juntamos para formar uma string de data para o parseador
        # Ex: ('30/06/2025', None, None, None, None, None, None, None) -> '30/06/2025'
        # Ex: (None, '30', 'junho', '2025', None, None, None, None) -> '30 junho 2025'
        date_parts = [part for part in match.groups() if part is not None]
        if date_parts:
            potential_date_strings.append(' '.join(date_parts).strip())
    
    # Se a regex não encontrou nada, mas a string original não está vazia, tente com a string inteira.
    if not potential_date_strings and normalized_str:
        potential_date_strings.append(normalized_str)


    for part in potential_date_strings:
        try:
            # Tentar DD/MM/YYYY
            if re.match(r'^\d{2}/\d{2}/\d{4}$', part):
                parsed_dates.append(datetime.strptime(part, '%d/%m/%Y'))
                continue
            
            # Tentar DD de Mês de AAAA ou DD de Mês (com ano inferido)
            match_full_pt_date = re.match(r'^(\d{1,2})\s+de\s+([a-zçãõ]+)(?:\s+de\s+(\d{4}))?$', part)
            if match_full_pt_date:
                day_str, month_name_pt, year_str = match_full_pt_date.groups()
                month_num = month_map.get(month_name_pt.replace('ç', 'c').replace('ã', 'a').replace('õ', 'o'))
                if month_num:
                    year = int(year_str) if year_str else current_year
                    parsed_dates.append(datetime(year, month_num, int(day_str)))
                    continue

            # Tentar Mês AAAA (ex: "julho 2025", assumir último dia do mês)
            match_month_year = re.match(r'^([a-zçãõ]+)\s+(\d{4})$', part)
            if match_month_year:
                month_name_pt, year_str = match_month_year.groups()
                month_num = month_map.get(month_name_pt.replace('ç', 'c').replace('ã', 'a').replace('õ', 'o'))
                if month_num:
                    last_day_of_month = (datetime(int(year_str), month_num % 12 + 1, 1) - timedelta(days=1)).day
                    parsed_dates.append(datetime(int(year_str), month_num, last_day_of_month))
                    continue
            
            # Tentar Mês DD, YYYY (inglês)
            match_en_date = re.match(r'([a-z]+)\s+(\d{1,2}),?\s*(\d{4})', part)
            if match_en_date:
                month_name_en, day_str_en, year_str_en = match_en_date.groups()
                month_num_en = month_map.get(month_name_en.lower())
                if month_num_en:
                    parsed_dates.append(datetime(int(year_str_en), month_num_en, int(day_str_en)))
                    continue

        except ValueError:
            pass # Ignora e tenta o próximo formato/parte
    
    # Se nenhuma data foi parseada, mas o original tinha termos como "N/A" que deveriam ser tratados por is_edital_open
    # A is_edital_open já lida com os termos de "sempre aberto" no início, então se chegamos aqui
    # e parsed_dates está vazia, significa que não há data válida para comparar.
    if not parsed_dates:
        print(f"AVISO: _parse_deadline não conseguiu extrair datas válidas da string: '{deadline_str}'")

    return parsed_dates

def is_edital_open(edital: dict) -> bool:
    """
    Verifica se um edital está em aberto com base na data atual e no prazo.
    Assume que o prazo está na chave 'deadline'.
    """
    # Garante que deadline_str é sempre uma string
    deadline_value = edital.get('deadline')
    deadline_str = str(deadline_value).strip() if deadline_value is not None else ''
    
    # Cenários de "sempre aberto" ou "não especificado"
    always_open_keywords = ['n/a', 'not specified', 'fluxo contínuo', 'aberta o ano todo', 'não especificado', 'null']
    if any(keyword in deadline_str.lower() for keyword in always_open_keywords):
        print(f"DEBUG: Prazo '{deadline_str}' é 'sempre aberto'.")
        return True

    today = datetime.now().date()
    parsed_deadlines = _parse_deadline(deadline_str)
    
    if not parsed_deadlines:
        print(f"AVISO: Prazo '{deadline_str}' do edital '{edital.get('title', 'N/A')}' não pôde ser parseado em datas. Edital considerado fechado.")
        return False
    
    for d in parsed_deadlines:
        # Se for a data muito distante no futuro (indicando "sempre aberto" do _parse_deadline)
        if d == datetime.max - timedelta(days=1):
            return True
        # Se a data parseada for no futuro (ou hoje), o edital está aberto
        if d.date() >= today:
            print(f"DEBUG: Prazo '{deadline_str}' parseado para {d.date()}, ainda aberto.")
            return True
            
    print(f"DEBUG: Todos os prazos parseados para '{deadline_str}' estão no passado. Edital considerado fechado.")
    return False

def load_cached_grants() -> list:
    """Carrega os editais do arquivo de cache."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"AVISO: Arquivo de cache '{CACHE_FILE}' corrompido ou vazio. Criando um novo.")
            return []
        except Exception as e:
            print(f"AVISO: Erro ao carregar cache de editais: {e}")
            return [] # Retorna vazio em caso de erro
    return []

def save_cached_grants(grants: list):
    """Salva os editais no arquivo de cache."""
    if grants:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(grants, f, indent=2, ensure_ascii=False)
            print(f"INFO: {len(grants)} editais abertos salvos em '{CACHE_FILE}'.")
        except Exception as e:
            print(f"AVISO: Falha ao salvar o cache de editais: {e}")
    else:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print(f"INFO: Nenhum edital aberto. Arquivo de cache '{CACHE_FILE}' removido.")

def manage_editals_cache(newly_extracted_editals: list[dict]) -> list[dict]:
    """
    Gerencia o cache de editais: carrega existentes, mescla com novos,
    deduplica, filtra editais expirados e salva o cache atualizado.
    Retorna a lista de editais abertos e atualizados.
    """
    print("Iniciando gerenciamento do cache de editais...")
    
    existing_grants = load_cached_grants()
    
    all_grants_map = {} 

    # Adicionar editais existentes que ainda estão abertos
    for edital in existing_grants:
        if edital.get('url') and is_edital_open(edital):
            all_grants_map[edital['url']] = edital
        else:
            # DEBUG: Edital existente descartado
            if edital.get('url'):
                print(f"DEBUG: Edital existente '{edital.get('title', 'N/A')}' (URL: {edital['url']}) descartado (expirado ou inválido).")

    # Adicionar/atualizar novos editais no mapa
    for grant in newly_extracted_editals:
        if grant.get('url'):
            if is_edital_open(grant):
                all_grants_map[grant['url']] = grant
            else:
                print(f"DEBUG: Novo edital '{grant.get('title', 'N/A')}' (URL: {grant['url']}) descartado (expirado ou inválido).")
        else:
            # Editais sem URL são tratados com cuidado para deduplicação
            # Criamos uma chave temporária baseada no título e agência
            temp_key = f"NOURL_{grant.get('title', 'untitled').replace(' ', '_')}_{grant.get('agency', 'unknown').replace(' ', '_')}"
            
            # Apenas adicione se não houver um com a mesma chave temporária e se estiver aberto
            if is_edital_open(grant):
                if temp_key not in all_grants_map:
                    all_grants_map[temp_key] = grant
                    print(f"AVISO: Edital '{grant.get('title', 'N/A')}' sem URL. Usando chave temporária para cache. Adicionado.")
                else:
                    print(f"DEBUG: Edital '{grant.get('title', 'N/A')}' sem URL e duplicado (chave temp). Sobrescrevendo/ignorado.")
            else:
                print(f"DEBUG: Novo edital '{grant.get('title', 'N/A')}' (sem URL) descartado (expirado ou inválido).")


    # Converte o mapa de volta para uma lista
    final_cached_and_filtered_editals = list(all_grants_map.values())

    # Salva a lista filtrada de volta no cache
    save_cached_grants(final_cached_and_filtered_editals)
    
    print(f"Gerenciamento do cache concluído. Total de editais abertos e no cache: {len(final_cached_and_filtered_editals)}")
    return final_cached_and_filtered_editals