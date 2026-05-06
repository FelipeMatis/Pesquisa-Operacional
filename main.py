import math
import re
import numpy as np
import random

random.seed()

# --- UTILITÁRIOS ---
def clean_value(value, decimals=6):
    """Evita ruídos de precisão decimal (ex: 0.0000001 vira 0.0)."""
    if abs(value) < (10 ** -decimals):
        return 0.0
    return round(float(value), decimals)

def extract_coefficient(coefficient):
    """Trata sinais e coeficientes vazios vindos do Regex."""
    coefficient = coefficient.replace(" ", "")
    if coefficient == "" or coefficient == "+": return 1.0
    if coefficient == "-": return -1.0
    return float(coefficient)

# --- LEITURA E FORMA PADRÃO ---
def carregar_modelo_txt(file_path):
    """Extrai os dados do TXT e identifica o número total de variáveis."""
    pattern = re.compile(r'([+-]?\s*\d*(?:\.\d+)?)\s*[xX](\d+)')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_lines = []
    max_index = 0
    for line in lines:
        norm = line.replace('−', '-').replace('≤', '<=').replace('≥', '>=').strip()
        if not norm or (re.search(r'>=\s*0', norm) and ',' in norm): continue
        matches = pattern.findall(norm)
        for _, idx_str in matches:
            max_index = max(max_index, int(idx_str))
        valid_lines.append(norm)

    n_vars = max_index
    tipo = "max"
    c = np.zeros(n_vars)
    matrix_A_list, vector_b_list, sinals = [], [], []

    # Processa Função Objetivo
    line_obj = valid_lines[0]
    if "min" in line_obj.lower(): tipo = "min"
    if "=" in line_obj:
        eq = line_obj.split("=")[1]
        for coef, idx in pattern.findall(eq):
            c[int(idx) - 1] = extract_coefficient(coef)

    # Processa Restrições
    for line in valid_lines[1:]:
        sinal = next((s for s in ['<=', '>=', '='] if s in line), None)
        if not sinal: continue
        left, right = line.split(sinal)
        line_A = np.zeros(n_vars)
        for coef, idx in pattern.findall(left):
            line_A[int(idx) - 1] = extract_coefficient(coef)
        matrix_A_list.append(line_A)
        vector_b_list.append(float(right.strip()))
        sinals.append(sinal)

    return tipo, c, np.array(matrix_A_list), np.array(vector_b_list), sinals

def normalized_func(A, sinals, c):
    """Adiciona variáveis de folga (+1) ou excesso (-1) para criar igualdades.""" #trata "<" e ">"
    m = A.shape[0]
    n_folgas = sum(1 for s in sinals if s in ['<=', '>='])
    matrix_folgas = np.zeros((m, n_folgas))
    c_p = np.hstack((c, np.zeros(n_folgas))) # Variáveis de folga têm custo zero
    
    idx = 0
    for i in range(m):
        if sinals[i] == '>=':
            matrix_folgas[i, idx] = -1
            idx += 1
        elif sinals[i] == '<=':
            matrix_folgas[i, idx] = 1
            idx += 1
    return np.hstack((A, matrix_folgas)), c_p

# --- ÁLGEBRA E NÚCLEO SIMPLEX ---
def inverter_matriz(M):
    """Inversão via Gauss-Jordan."""
    n = len(M)
    mat = np.hstack((M, np.eye(n)))
    for i in range(n):
        pivo_linha = i + np.argmax(np.abs(mat[i:, i]))
        mat[[i, pivo_linha]] = mat[[pivo_linha, i]]
        pivo_val = mat[i, i]
        if abs(pivo_val) < 1e-10: raise ValueError("Matriz Singular!") #se a determinante for = 0, matriz singular
        mat[i] /= pivo_val
        for j in range(n):
            if i != j: mat[j] -= mat[j, i] * mat[i]
    return np.vectorize(clean_value)(mat[:, n:])

def basic_nonBasic(A, b):
    """Sorteia colunas para tentar formar uma base inicial."""
    m, n = A.shape
    s_vals = random.sample(range(n), m)
    nb_vals = [i for i in range(n) if i not in s_vals]
    return A[:, s_vals], A[:, nb_vals], s_vals, nb_vals

def simplex2(c, B, NB, B_idx, NB_idx, inv_B, b, tipo):
    """Passos lógicos do Simplex."""
    cB, cN = c[B_idx], c[NB_idx]
    
    # 1. Vetor Multiplicador (Dual): pi = cB * B^-1
    pi = np.dot(cB, inv_B)
    print(f"Vetor Multiplicador (Pi): {[clean_value(v) for v in pi]}")

    # 2. Custos Reduzidos: c_reduzido = cN - (pi * N)
    rel_cN = [clean_value(v) for v in (cN - np.dot(pi, NB))]
    print(f"Custos Reduzidos: {rel_cN}")

    # 3. Teste de Otimalidade (Se tudo >= 0, fim)
    min_c = min(rel_cN)
    if min_c >= 0:
        xB = np.dot(inv_B, b)
        sol = np.zeros(len(c))
        for i, idx in enumerate(B_idx): sol[idx] = clean_value(xB[i])
        z = np.dot(cB, xB)
        if tipo == "max": z *= -1
        print(f"\nSolução Completa: {sol}\nValor Ótimo (Z): {clean_value(z)}")
        return True, None, None

    # 4. Variável de Entrada (Mais negativa) e Direção Simplex (y)
    ent_idx = rel_cN.index(min_c)
    y = np.dot(inv_B, NB[:, ent_idx])
    xB = np.dot(inv_B, b)
    
    # 5. Teste da Razão Mínima (Quem sai da base?)
    razoes = [ (xB[i]/y[i] if y[i] > 0 else float('inf')) for i in range(len(y)) ]
    min_raz = min(razoes)
    if min_raz == float('inf'):
        print("[ALERTA] Problema ILIMITADO."); return True, None, None

    return False, ent_idx, razoes.index(min_raz)

# --- EXECUÇÃO ---
if __name__ == "__main__":
    try:
        tipo, c_orig, A_orig, b_orig, sinais = carregar_modelo_txt('func.txt')
        A_mat, c_p = normalized_func(A_orig, sinais, c_orig)
        if tipo == "max": c_p *= -1 # Transforma Max em Min

        # Busca Base Inicial Factível (xB >= 0)
        tested_bases = set()
        while True:
            B_np, NB_np, s_vals, nb_vals = basic_nonBasic(A_mat, b_orig)
            base_t = tuple(sorted(s_vals))
            if base_t in tested_bases: continue
            tested_bases.add(base_t)

            try:
                inv_temp = inverter_matriz(B_np)
                xB_t = np.dot(inv_temp, b_orig)
                if all(clean_value(v) >= 0 for v in xB_t): # Garante solução positiva
                    new_B, new_NB, sorted_values, nb_values, inv_B = B_np, NB_np, s_vals, nb_vals, inv_temp
                    break
            except: continue
            if len(tested_bases) > 100: exit("[ERRO] Nenhuma base factível encontrada.")

        great, it = False, 1
        while not great and it <= 20:
            print(f"\n{'='*40}\nITERAÇÃO {it}\n{'='*40}")
            great, e_idx, l_idx = simplex2(c_p, new_B, new_NB, sorted_values, nb_values, inv_B, b_orig, tipo)
            if not great:
                v_ent, v_sai = nb_values[e_idx], sorted_values[l_idx]
                print(f"Troca: x{v_sai+1} sai e x{v_ent+1} entra.")
                
                # Atualização dos índices e matrizes
                sorted_values[l_idx], nb_values[e_idx] = v_ent, v_sai
                new_B, new_NB = A_mat[:, sorted_values], A_mat[:, nb_values]
                inv_B = inverter_matriz(new_B)
                it += 1
    except Exception as e: print(f"[ERRO] {e}")
