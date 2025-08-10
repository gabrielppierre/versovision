#!/usr/bin/env python3
"""
Script de teste para identificar problema de frame preto na montagem de vídeo
"""

import os
import json
from PIL import Image
from moviepy.editor import ImageSequenceClip, ImageClip, concatenate_videoclips
import moviepy.video.fx.all as vfx

def test_video_assembly():
    """Testa diferentes abordagens de montagem de vídeo"""

    PASTA_TEMP = "temp"

    # Carrega o roteiro
    with open(os.path.join(PASTA_TEMP, "roteiro_estruturado.json"), "r", encoding="utf-8") as f:
        cenas_roteiro = json.load(f)

    print("=== TESTE DE MONTAGEM DE VÍDEO ===")
    print(f"Total de cenas no roteiro: {len(cenas_roteiro)}")

    # Verifica quais imagens existem
    imagens_disponiveis = []
    for i in range(1, len(cenas_roteiro) + 1):
        caminho_img = os.path.join(PASTA_TEMP, f"img_{i}.jpg")
        if os.path.exists(caminho_img):
            size = os.path.getsize(caminho_img)
            img = Image.open(caminho_img)
            imagens_disponiveis.append({
                'indice': i,
                'caminho': caminho_img,
                'size': img.size,
                'tamanho_arquivo': size,
                'cena': cenas_roteiro[i-1]
            })
            print(f"✓ img_{i}.jpg ({img.size}, {size} bytes)")
        else:
            print(f"✗ img_{i}.jpg (não encontrada)")

    print(f"\nImagens disponíveis: {len(imagens_disponiveis)} de {len(cenas_roteiro)}")

    # Verifica transições
    print("\n=== TRANSIÇÕES DISPONÍVEIS ===")
    for i, img_info in enumerate(imagens_disponiveis):
        if i < len(imagens_disponiveis) - 1:
            img_atual = img_info['indice']
            img_proxima = imagens_disponiveis[i + 1]['indice']

            # Conta quantas transições existem
            count_trans = 0
            j = 1
            while True:
                caminho_trans = os.path.join(PASTA_TEMP, f"trans_{img_atual}_{j:02d}.jpg")
                if os.path.exists(caminho_trans):
                    count_trans += 1
                    j += 1
                else:
                    break

            print(f"img_{img_atual} -> img_{img_proxima}: {count_trans} frames de transição")

    # Teste 1: Vídeo simples sem transições
    print("\n=== TESTE 1: VÍDEO SIMPLES (SEM TRANSIÇÕES) ===")
    try:
        clipes_simples = []
        for img_info in imagens_disponiveis:
            duracao = img_info['cena']['end'] - img_info['cena']['start']
            if duracao <= 0:
                duracao = 1.0

            clipe = ImageClip(img_info['caminho'], duration=duracao)
            clipes_simples.append(clipe)
            print(f"Clipe img_{img_info['indice']}: {duracao:.2f}s")

        video_simples = concatenate_videoclips(clipes_simples)
        video_simples.write_videofile("saida/teste_simples.mp4", fps=24, verbose=False, logger=None)
        print("✓ Vídeo simples criado com sucesso!")

    except Exception as e:
        print(f"✗ Erro no teste simples: {e}")

    # Teste 2: Vídeo com transições válidas apenas
    print("\n=== TESTE 2: VÍDEO COM TRANSIÇÕES ===")
    try:
        clipes_com_transicoes = []

        for i, img_info in enumerate(imagens_disponiveis):
            frames_cena = [img_info['caminho']]

            # Adiciona transições se existirem e a próxima imagem for sequencial
            if i < len(imagens_disponiveis) - 1:
                img_atual = img_info['indice']
                img_proxima = imagens_disponiveis[i + 1]['indice']

                # Só adiciona transições se as imagens forem consecutivas ou se existirem
                j = 1
                while True:
                    caminho_trans = os.path.join(PASTA_TEMP, f"trans_{img_atual}_{j:02d}.jpg")
                    if os.path.exists(caminho_trans):
                        frames_cena.append(caminho_trans)
                        j += 1
                    else:
                        break

            if len(frames_cena) > 1:
                print(f"Cena {img_info['indice']}: {len(frames_cena)} frames total")

                # Calcula duração e FPS
                duracao = img_info['cena']['end'] - img_info['cena']['start']
                if duracao <= 0:
                    duracao = 1.0

                fps = min(len(frames_cena) / duracao, 30)  # Máximo 30 FPS
                fps = max(fps, 1)  # Mínimo 1 FPS

                clipe_animado = ImageSequenceClip(frames_cena, fps=fps)

                # Se o clipe é mais curto que deveria, congela o último frame
                if clipe_animado.duration < duracao:
                    tempo_extra = duracao - clipe_animado.duration
                    clipe_final = clipe_animado.fx(vfx.freeze, t='end', freeze_duration=tempo_extra)
                else:
                    clipe_final = clipe_animado.subclip(0, duracao)

                clipes_com_transicoes.append(clipe_final)
                print(f"  -> Clipe final: {clipe_final.duration:.2f}s, FPS: {fps:.1f}")
            else:
                # Sem transições, usa imagem estática
                duracao = img_info['cena']['end'] - img_info['cena']['start']
                if duracao <= 0:
                    duracao = 1.0
                clipe = ImageClip(img_info['caminho'], duration=duracao)
                clipes_com_transicoes.append(clipe)
                print(f"Cena {img_info['indice']}: imagem estática {duracao:.2f}s")

        video_transicoes = concatenate_videoclips(clipes_com_transicoes)
        video_transicoes.write_videofile("saida/teste_transicoes.mp4", fps=24, verbose=False, logger=None)
        print("✓ Vídeo com transições criado com sucesso!")

    except Exception as e:
        print(f"✗ Erro no teste com transições: {e}")
        import traceback
        traceback.print_exc()

    # Teste 3: Análise de frames individuais
    print("\n=== TESTE 3: ANÁLISE DE FRAMES INDIVIDUAIS ===")

    # Verifica se algum frame pode estar gerando problema
    frames_problema = []

    for img_info in imagens_disponiveis:
        try:
            img = Image.open(img_info['caminho'])
            # Verifica se a imagem é muito escura (pode parecer preta)
            import numpy as np
            img_array = np.array(img)
            media_brilho = np.mean(img_array)

            if media_brilho < 50:  # Muito escuro
                frames_problema.append((img_info['caminho'], media_brilho))
                print(f"⚠ img_{img_info['indice']}: muito escura (brilho médio: {media_brilho:.1f})")
            else:
                print(f"✓ img_{img_info['indice']}: OK (brilho médio: {media_brilho:.1f})")

        except Exception as e:
            print(f"✗ Erro ao analisar img_{img_info['indice']}: {e}")

    if frames_problema:
        print(f"\n⚠ Encontradas {len(frames_problema)} imagens muito escuras que podem parecer pretas:")
        for caminho, brilho in frames_problema:
            print(f"  {caminho}: brilho médio {brilho:.1f}")
    else:
        print("\n✓ Nenhuma imagem excessivamente escura encontrada")

    print("\n=== RESUMO DOS TESTES ===")
    print("Verifique os arquivos gerados em saida/:")
    print("- teste_simples.mp4: vídeo básico sem transições")
    print("- teste_transicoes.mp4: vídeo com transições (se criado)")
    print("\nCompare estes vídeos com o original para identificar onde aparece o frame preto.")

if __name__ == "__main__":
    # Cria pasta de saída se não existir
    os.makedirs("saida", exist_ok=True)
    test_video_assembly()
