#!/usr/bin/env python3
"""
Teste de Sincronização Palavra-Imagem
Este arquivo testa e valida o sistema de sincronização entre palavras-chave e imagens.
"""

import json
import os
from datetime import datetime

def testar_sincronizacao():
    """Testa o sistema de sincronização palavra-imagem"""

    print("🔍 TESTE DE SINCRONIZAÇÃO PALAVRA-IMAGEM")
    print("=" * 50)

    # Caminhos dos arquivos
    pasta_temp = "temp"
    arquivo_palavras = os.path.join(pasta_temp, "palavras_narracao.json")
    arquivo_roteiro = os.path.join(pasta_temp, "roteiro_estruturado.json")

    # Verificar se os arquivos existem
    if not os.path.exists(arquivo_palavras):
        print(f"❌ Arquivo não encontrado: {arquivo_palavras}")
        print("Execute o pipeline principal primeiro!")
        return False

    if not os.path.exists(arquivo_roteiro):
        print(f"❌ Arquivo não encontrado: {arquivo_roteiro}")
        print("Execute o pipeline principal primeiro!")
        return False

    try:
        # Carregar dados
        with open(arquivo_palavras, 'r', encoding='utf-8') as f:
            palavras = json.load(f)

        with open(arquivo_roteiro, 'r', encoding='utf-8') as f:
            roteiro = json.load(f)

        print(f"✅ Dados carregados:")
        print(f"   - {len(palavras)} palavras com timestamps")
        print(f"   - {len(roteiro)} cenas no roteiro")
        print()

        # Análise de sincronização
        print("📊 ANÁLISE DE SINCRONIZAÇÃO")
        print("-" * 30)

        cenas_com_palavra_chave = 0
        cenas_sincronizadas = 0
        problemas = []

        for i, cena in enumerate(roteiro):
            print(f"\n🎬 Cena {i+1}:")
            print(f"   Texto: '{cena.get('text', 'N/A')[:50]}...'")

            if 'palavra_chave_visual' in cena:
                cenas_com_palavra_chave += 1
                palavra_chave = cena['palavra_chave_visual']
                print(f"   🔑 Palavra-chave: '{palavra_chave}'")

                if 'timestamp_palavra_chave' in cena:
                    cenas_sincronizadas += 1
                    timestamp = cena['timestamp_palavra_chave']
                    print(f"   ⏰ Timestamp: {timestamp:.3f}s")

                    # Verificar se a palavra realmente existe nos dados
                    palavra_encontrada = False
                    for palavra_info in palavras:
                        if palavra_chave.lower() in palavra_info['word'].lower():
                            palavra_encontrada = True
                            diferenca = abs(timestamp - palavra_info['start'])
                            if diferenca > 0.1:  # Mais de 100ms de diferença
                                problemas.append(f"Cena {i+1}: Diferença de timing suspeita ({diferenca:.3f}s)")
                            break

                    if not palavra_encontrada:
                        problemas.append(f"Cena {i+1}: Palavra-chave '{palavra_chave}' não encontrada no áudio")
                        print(f"   ⚠️  Palavra não encontrada no áudio transcrito")
                    else:
                        print(f"   ✅ Palavra encontrada e sincronizada")
                else:
                    problemas.append(f"Cena {i+1}: Palavra-chave sem timestamp")
                    print(f"   ❌ Timestamp não encontrado")
            else:
                problemas.append(f"Cena {i+1}: Sem palavra-chave definida")
                print(f"   ❌ Sem palavra-chave visual")

        # Relatório final
        print("\n" + "=" * 50)
        print("📈 RELATÓRIO FINAL")
        print("=" * 50)

        taxa_palavras_chave = (cenas_com_palavra_chave / len(roteiro)) * 100
        taxa_sincronizacao = (cenas_sincronizadas / len(roteiro)) * 100

        print(f"📊 Estatísticas:")
        print(f"   - Total de cenas: {len(roteiro)}")
        print(f"   - Cenas com palavra-chave: {cenas_com_palavra_chave} ({taxa_palavras_chave:.1f}%)")
        print(f"   - Cenas sincronizadas: {cenas_sincronizadas} ({taxa_sincronizacao:.1f}%)")
        print(f"   - Palavras únicas no áudio: {len(set(p['word'] for p in palavras))}")

        if problemas:
            print(f"\n⚠️  PROBLEMAS ENCONTRADOS ({len(problemas)}):")
            for problema in problemas:
                print(f"   - {problema}")
        else:
            print(f"\n✅ Nenhum problema encontrado! Sistema funcionando perfeitamente.")

        # Teste de precisão temporal
        print(f"\n🎯 TESTE DE PRECISÃO TEMPORAL")
        print("-" * 30)

        tempos_palavra = [p['start'] for p in palavras]
        if tempos_palavra:
            duracao_total = max(tempos_palavra) - min(tempos_palavra)
            densidade_palavras = len(palavras) / duracao_total if duracao_total > 0 else 0

            print(f"   - Duração do áudio: {duracao_total:.2f}s")
            print(f"   - Densidade de palavras: {densidade_palavras:.1f} palavras/segundo")
            print(f"   - Precisão mínima: {min(p['end'] - p['start'] for p in palavras if p['end'] > p['start']):.3f}s")
            print(f"   - Precisão média: {sum(p['end'] - p['start'] for p in palavras if p['end'] > p['start']) / len([p for p in palavras if p['end'] > p['start']]):.3f}s")

        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES")
        print("-" * 30)

        if taxa_sincronizacao < 80:
            print("   - ⚠️  Taxa de sincronização baixa. Considere revisar o prompt de geração do roteiro.")

        if len(problemas) > len(roteiro) * 0.2:
            print("   - ⚠️  Muitos problemas detectados. Verifique a qualidade da transcrição.")

        if densidade_palavras > 3:
            print("   - 💡 Alta densidade de palavras. Considere aumentar o atraso visual para melhor sincronização.")
        elif densidade_palavras < 1:
            print("   - 💡 Baixa densidade de palavras. Você pode diminuir o atraso visual.")

        print("   - 🎛️  Ajuste o atraso em milissegundos conforme necessário no arquivo main.py")
        print("   - 🎥 Teste diferentes valores entre 100-500ms dependendo do ritmo da narração")

        return len(problemas) == 0

    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        return False

def mostrar_timeline():
    """Mostra uma timeline visual da sincronização"""

    pasta_temp = "temp"
    arquivo_roteiro = os.path.join(pasta_temp, "roteiro_estruturado.json")

    if not os.path.exists(arquivo_roteiro):
        print("❌ Arquivo do roteiro não encontrado!")
        return

    try:
        with open(arquivo_roteiro, 'r', encoding='utf-8') as f:
            roteiro = json.load(f)

        print("\n🕐 TIMELINE DE SINCRONIZAÇÃO")
        print("=" * 50)

        eventos = []
        for i, cena in enumerate(roteiro):
            if 'timestamp_palavra_chave' in cena:
                eventos.append({
                    'tempo': cena['timestamp_palavra_chave'],
                    'tipo': 'palavra',
                    'cena': i + 1,
                    'palavra': cena.get('palavra_chave_visual', 'N/A'),
                    'texto': cena.get('text', '')[:30] + '...'
                })

                # Calcular quando a imagem aparecerá (assumindo 200ms de atraso padrão)
                atraso = 0.2  # 200ms
                eventos.append({
                    'tempo': cena['timestamp_palavra_chave'] + atraso,
                    'tipo': 'imagem',
                    'cena': i + 1,
                    'palavra': cena.get('palavra_chave_visual', 'N/A'),
                    'descricao': cena.get('descricao_visual', 'N/A')[:40] + '...'
                })

        # Ordenar por tempo
        eventos.sort(key=lambda x: x['tempo'])

        print("Tempo    | Evento | Cena | Detalhes")
        print("-" * 50)

        for evento in eventos:
            tempo_str = f"{evento['tempo']:6.2f}s"
            if evento['tipo'] == 'palavra':
                print(f"{tempo_str} | 🗣️ FALA | C{evento['cena']:2d}  | '{evento['palavra']}' - {evento['texto']}")
            else:
                print(f"{tempo_str} | 🖼️ IMG  | C{evento['cena']:2d}  | {evento['descricao']}")

        print("\n📝 Legenda:")
        print("   🗣️ FALA = Momento que a palavra-chave é falada")
        print("   🖼️ IMG  = Momento que a imagem correspondente aparece")

    except Exception as e:
        print(f"❌ Erro ao mostrar timeline: {e}")

if __name__ == "__main__":
    print(f"Iniciando teste em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sucesso = testar_sincronizacao()
    mostrar_timeline()

    if sucesso:
        print(f"\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print("Sistema de sincronização palavra-imagem funcionando corretamente.")
    else:
        print(f"\n⚠️  TESTE CONCLUÍDO COM PROBLEMAS")
        print("Verifique os problemas reportados acima.")

    print(f"\nTeste finalizado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
