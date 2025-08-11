import os
import gradio as gr
import whisper
import google.generativeai as genai
from google.cloud import texttospeech
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from spleeter.separator import Separator
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import time
import gc
import tempfile
import shutil
from moviepy.editor import (ImageSequenceClip, AudioFileClip, CompositeAudioClip, 
                            afx, ImageClip, concatenate_videoclips, CompositeVideoClip)

# --- Configurações e Checagens Iniciais ---

# Adiciona o caminho do modelo RIFE ao sistema
sys.path.append('rife_model')
try:
    from train_log.RIFE_HDv3 import Model as RIFE_Model
except ImportError:
    raise ImportError("Não foi possível encontrar o modelo RIFE. Certifique-se de que a pasta 'rife_model' está no mesmo diretório que app.py.")

if hasattr(Image, "Resampling") and not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# --- Funções Auxiliares ---

def criar_prompt_para_imagem_com_estilo(descricao_visual, estilo_usuario, trecho_letra=None):
    """
    Usa o Gemini para criar um prompt de imagem detalhado.
    """
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')

        contexto_letra = f"\n\n**Trecho da Letra (contexto):**\n\"{trecho_letra}\"" if trecho_letra else ""
        prompt_template = f"""
        Crie um prompt de imagem em inglês para uma IA generativa.
        O prompt deve ser detalhado, cinematográfico e capturar a essência da seguinte descrição, estilo e contexto da letra.
        **Descrição Visual:** "{descricao_visual}"
        **Estilo Artístico:** "{estilo_usuario}"{contexto_letra}
        **Instruções:** Combine tudo em um prompt coeso, em inglês, focado na emoção e na atmosfera. Retorne APENAS o prompt.
        """
        response = model.generate_content(prompt_template)
        prompt_final = response.text.strip().replace('"', '').replace("Prompt:", "").strip()
        return prompt_final
    except Exception as e:
        print(f"Aviso: Falha ao usar Gemini para prompt, usando fallback. Erro: {e}")
        return f"{descricao_visual}, in the style of {estilo_usuario}"

# --- Função Principal do Pipeline para o Gradio ---

def gerar_video_completo(arquivo_musica, estilo_visual, atraso_ms, progress=gr.Progress(track_tqdm=True)):
    """
    Função principal que encapsula todo o pipeline de geração de vídeo.
    Recebe as entradas do Gradio e retorna o caminho do vídeo final.
    """
    if not arquivo_musica:
        raise gr.Error("Por favor, faça o upload de um arquivo de música.")

    # Cria um diretório temporário único para esta execução
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Definição de todos os caminhos dentro do diretório temporário
        progress(0, desc="Configurando ambiente...")
        PASTA_ENTRADA = os.path.join(temp_dir, "entrada")
        PASTA_SAIDA = os.path.join(temp_dir, "saida")
        PASTA_TEMP = os.path.join(temp_dir, "temp")
        os.makedirs(PASTA_ENTRADA, exist_ok=True)
        os.makedirs(PASTA_SAIDA, exist_ok=True)
        os.makedirs(PASTA_TEMP, exist_ok=True)

        nome_arquivo_musica = os.path.basename(arquivo_musica.name)
        caminho_musica = os.path.join(PASTA_ENTRADA, nome_arquivo_musica)
        shutil.copy(arquivo_musica.name, caminho_musica)

        ATRASO_VISUAL_SEGUNDOS = atraso_ms / 1000.0
        TEMPO_RESPIRO_SEGUNDOS = 3

        caminho_letra = os.path.join(PASTA_TEMP, "letra.txt")
        caminho_roteiro_narracao = os.path.join(PASTA_TEMP, "roteiro_narracao.txt")
        caminho_narracao_audio = os.path.join(PASTA_TEMP, "narracao.mp3")
        caminho_segmentos_narracao = os.path.join(PASTA_TEMP, "segmentos_narracao.json")
        caminho_palavras_narracao = os.path.join(PASTA_TEMP, "palavras_narracao.json")
        caminho_roteiro_estruturado = os.path.join(PASTA_TEMP, "roteiro_estruturado.json")
        nome_base_musica = os.path.splitext(nome_arquivo_musica)[0]
        caminho_musica_instrumental = os.path.join(PASTA_TEMP, nome_base_musica, "accompaniment.wav")
        caminho_video_final = os.path.join(PASTA_SAIDA, "video_final.mp4")

        # --- INÍCIO DO PIPELINE ---

        # Etapa 2: Transcrição da Letra com Whisper
        progress(0.05, desc="Etapa 1/9: Transcrevendo letra...")
        try:
            modelo_whisper = whisper.load_model("medium", device="cpu")
            resultado = modelo_whisper.transcribe(caminho_musica, language="pt", verbose=False)
            with open(caminho_letra, "w", encoding="utf-8") as f:
                f.write(resultado['text'])
            del modelo_whisper; gc.collect()
        except Exception as e:
            raise gr.Error(f"Erro no Whisper (transcrição): {e}")

        # Etapa 3: Criação do roteiro da narração com Gemini
        progress(0.15, desc="Etapa 2/9: Criando roteiro da narração...")
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash')
            with open(caminho_letra, "r", encoding="utf-8") as f:
                letra_musica = f.read()
            prompt_narracao = f"""Com base na letra de música a seguir, crie uma narração curta, poética e analítica (máximo de 100 palavras) em português. Retorne APENAS a narração. Letra: --- {letra_musica} ---"""
            response_narracao = model.generate_content(prompt_narracao)
            with open(caminho_roteiro_narracao, "w", encoding="utf-8") as f:
                f.write(response_narracao.text)
        except Exception as e:
            raise gr.Error(f"Erro no Gemini (criação de roteiro): {e}")

        # Etapa 4: Geração do áudio da narração com Google TTS
        progress(0.25, desc="Etapa 3/9: Gerando áudio da narração...")
        try:
            with open(caminho_roteiro_narracao, "r", encoding="utf-8") as f:
                texto_narracao = f.read()
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=texto_narracao)
            voice = texttospeech.VoiceSelectionParams(language_code="pt-BR", name="pt-BR-Wavenet-D")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            with open(caminho_narracao_audio, "wb") as out:
                out.write(response.audio_content)
        except Exception as e:
            raise gr.Error(f"Erro no Google Text-to-Speech: {e}")

        # Etapa 5: Análise do áudio da narração com timestamps de palavras
        progress(0.35, desc="Etapa 4/9: Sincronizando palavras da narração...")
        try:
            modelo_whisper = whisper.load_model("medium", device="cpu")
            resultado = modelo_whisper.transcribe(caminho_narracao_audio, language="pt", word_timestamps=True, verbose=False)
            
            with open(caminho_segmentos_narracao, 'w', encoding='utf-8') as f:
                json.dump(resultado['segments'], f, ensure_ascii=False, indent=4)

            palavras_com_tempo = []
            for segmento in resultado['segments']:
                if 'words' in segmento:
                    for palavra_info in segmento['words']:
                        palavras_com_tempo.append({
                            'word': palavra_info['word'].strip(),
                            'start': palavra_info['start'],
                            'end': palavra_info['end'],
                            'segment_id': segmento['id']
                        })
            with open(caminho_palavras_narracao, 'w', encoding='utf-8') as f:
                json.dump(palavras_com_tempo, f, ensure_ascii=False, indent=4)
            del modelo_whisper; gc.collect()
        except Exception as e:
            raise gr.Error(f"Erro no Whisper (timestamps): {e}")

        # Etapa 6: Criação do roteiro visual estruturado com Gemini
        progress(0.45, desc="Etapa 5/9: Criando roteiro visual...")
        try:
            with open(caminho_segmentos_narracao, "r", encoding="utf-8") as f:
                segmentos_narracao = json.load(f)
            with open(caminho_palavras_narracao, "r", encoding="utf-8") as f:
                palavras_com_tempo = json.load(f)

            prompt_visual_estruturado = f"""
            Como diretor de arte, crie um roteiro visual em JSON. Para cada segmento da narração, gere um objeto com "id", "start", "end", "text" (copiados do original), e adicione:
            1. "descricao_visual": Uma descrição literal e direta da cena. Seja concreto.
            2. "palavra_chave_visual": A palavra mais importante do texto para a imagem.
            Narração: --- {json.dumps(segmentos_narracao, indent=2, ensure_ascii=False)} ---
            Retorne APENAS o JSON.
            """
            model = genai.GenerativeModel('gemini-1.5-flash')
            response_visual = model.generate_content(prompt_visual_estruturado)
            texto_json = response_visual.text.strip().replace("```json", "").replace("```", "")
            roteiro_json = json.loads(texto_json)
            
            # Adicionar timestamps de palavras-chave
            for cena in roteiro_json:
                palavra_chave = cena.get('palavra_chave_visual', '').lower()
                timestamp = cena['start'] # Fallback
                for p in palavras_com_tempo:
                    if p['segment_id'] == cena['id'] and palavra_chave in p['word'].lower():
                        timestamp = p['start']
                        break
                cena['timestamp_palavra_chave'] = timestamp

            with open(caminho_roteiro_estruturado, "w", encoding="utf-8") as f:
                json.dump(roteiro_json, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise gr.Error(f"Erro no Gemini (roteiro visual): {e}")

        # Etapa 7: Geração de imagens com Imagen/Vertex AI
        progress(0.55, desc="Etapa 6/9: Gerando imagens...")
        try:
            vertexai.init(project=os.getenv("GCP_PROJECT_ID"), location="us-central1")
            generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            
            with open(caminho_roteiro_estruturado, "r", encoding="utf-8") as f:
                cenas_roteiro = json.load(f)

            for i, cena in enumerate(progress.tqdm(cenas_roteiro, desc="Gerando imagens...")):
                prompt = criar_prompt_para_imagem_com_estilo(cena['descricao_visual'], estilo_visual, cena['text'])
                response = generation_model.generate_images(prompt=prompt, number_of_images=1)
                caminho_imagem_cena = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
                response.images[0].save(location=caminho_imagem_cena)
                time.sleep(5) # Pausa para evitar erros de "rate limiting"
        except Exception as e:
            raise gr.Error(f"Erro no Vertex AI Imagen: {e}. Verifique as permissões e cotas.")

        # Etapa 8: Criação de transições com RIFE
        progress(0.70, desc="Etapa 7/9: Gerando transições de vídeo...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            rife_model = RIFE_Model()
            rife_model.load_model('rife_model/train_log', -1)
            rife_model.eval()
            rife_model.device()
            
            transform = transforms.ToTensor()
            
            with open(caminho_roteiro_estruturado, "r", encoding="utf-8") as f:
                cenas_roteiro = json.load(f)

            for i in progress.tqdm(range(len(cenas_roteiro) - 1), desc="Criando transições..."):
                img1_path = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
                img2_path = os.path.join(PASTA_TEMP, f"img_{i+2}.jpg")
                if not (os.path.exists(img1_path) and os.path.exists(img2_path)): continue

                img1 = Image.open(img1_path).convert("RGB"); img2 = Image.open(img2_path).convert("RGB")
                w, h = img1.size; w_new = (w // 32) * 32; h_new = (h // 32) * 32
                img1 = img1.resize((w_new, h_new)); img2 = img2.resize((w_new, h_new))
                img1_tensor = transform(img1).unsqueeze(0).to(device); img2_tensor = transform(img2).unsqueeze(0).to(device)
                
                # Gerador para interpolação
                def make_inference_generator(img1, img2, n_recursions=5): # N recursions mais baixo para performance
                    if n_recursions > 0:
                        middle_frame = rife_model.inference(img1, img2)
                        yield from make_inference_generator(img1, middle_frame, n_recursions - 1)
                        yield middle_frame
                        yield from make_inference_generator(middle_frame, img2, n_recursions - 1)

                frame_generator = make_inference_generator(img1_tensor, img2_tensor)
                for j, frame_tensor in enumerate(frame_generator):
                    frame_image = transforms.ToPILImage()(frame_tensor.squeeze(0).cpu())
                    frame_image.save(os.path.join(PASTA_TEMP, f"trans_{i+1}_{j+1:02d}.jpg"))
                del img1, img2, img1_tensor, img2_tensor; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"Aviso: Falha ao gerar transições RIFE. O vídeo será criado com cortes secos. Erro: {e}")

        # Etapa 8.5: Separação de vocais com Spleeter
        progress(0.85, desc="Etapa 8/9: Separando vocais da música...")
        try:
            separator = Separator('spleeter:2stems', multiprocess=False) # multiprocess=False para compatibilidade
            separator.separate_to_file(caminho_musica, PASTA_TEMP, codec='wav')
            if not os.path.exists(caminho_musica_instrumental):
                raise FileNotFoundError("Spleeter não gerou o arquivo de acompanhamento.")
        except Exception as e:
            print(f"Aviso: Falha no Spleeter. Usando música original. Erro: {e}")
            shutil.copy(caminho_musica, caminho_musica_instrumental) # Fallback

        # Etapa 9: Montagem do vídeo final com MoviePy
        progress(0.90, desc="Etapa 9/9: Montando o vídeo final...")
        try:
            audio_narracao = AudioFileClip(caminho_narracao_audio)
            audio_musica = AudioFileClip(caminho_musica_instrumental).volumex(0.15)
            with open(caminho_roteiro_estruturado, "r", encoding="utf-8") as f:
                cenas_roteiro = json.load(f)

            # Encontrar o tamanho padrão e a primeira imagem
            tamanho_padrao, primeira_imagem_encontrada = None, None
            for i in range(len(cenas_roteiro)):
                caminho_img = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
                if os.path.exists(caminho_img):
                    tamanho_padrao = Image.open(caminho_img).size
                    primeira_imagem_encontrada = caminho_img
                    break
            if not tamanho_padrao: raise gr.Error("Nenhuma imagem foi gerada. Impossível criar o vídeo.")
            
            # Montar clipes
            clipes_visuais = []
            for i, cena in enumerate(cenas_roteiro):
                frames_da_cena = []
                caminho_img_principal = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
                if not os.path.exists(caminho_img_principal): continue
                
                frames_da_cena.append(caminho_img_principal)
                # Adicionar transições se existirem
                j = 1
                while True:
                    caminho_trans = os.path.join(PASTA_TEMP, f"trans_{i}_{j:02d}.jpg")
                    if os.path.exists(caminho_trans):
                        frames_da_cena.append(caminho_trans)
                        j += 1
                    else:
                        break
                
                duracao_cena = cena['end'] - cena['start']
                if duracao_cena <= 0: duracao_cena = 0.5 # Duração mínima
                
                # Garante que todos os frames tenham o mesmo tamanho
                clip_cena = ImageSequenceClip(frames_da_cena, fps=24).set_duration(duracao_cena)
                clip_cena_resized = clip_cena.resize(tamanho_padrao)
                clipes_visuais.append(clip_cena_resized)
            
            video_animado = concatenate_videoclips(clipes_visuais)

            # Adicionar respiros no início e fim
            clipe_inicio = ImageClip(primeira_imagem_encontrada, duration=TEMPO_RESPIRO_SEGUNDOS).resize(tamanho_padrao)
            
            ultima_imagem_path = clipes_visuais[-1].get_frame(clipes_visuais[-1].duration - 0.1)
            clipe_fim = ImageClip(ultima_imagem_path, duration=TEMPO_RESPIRO_SEGUNDOS).set_fps(24)

            video_final_visual = concatenate_videoclips([clipe_inicio, video_animado, clipe_fim])
            
            # Composição do áudio
            duracao_total = video_final_visual.duration
            audio_narracao_posicionado = audio_narracao.set_start(TEMPO_RESPIRO_SEGUNDOS)
            audio_musica_fundo = audio_musica.set_duration(duracao_total).fx(afx.audio_fadein, 2).fx(afx.audio_fadeout, 3)
            audio_final = CompositeAudioClip([audio_musica_fundo, audio_narracao_posicionado])
            
            video_com_audio = video_final_visual.set_audio(audio_final)
            
            # Renderização final
            video_com_audio.write_videofile(caminho_video_final, codec='libx264', audio_codec='aac', preset='medium', ffmpeg_params=['-pix_fmt', 'yuv420p'])
            
            # Copia o vídeo final para um local que o Gradio possa acessar antes da limpeza
            caminho_final_acessivel = shutil.copy(caminho_video_final, ".")
            progress(1.0, desc="Concluído!")
            return caminho_final_acessivel

        except Exception as e:
            raise gr.Error(f"Erro na montagem do vídeo com MoviePy: {e}")

# --- Interface Gradio ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as demo:
    gr.Markdown(
        """
        # 🎬 Gerador de Vídeo Musical com IA
        Faça o upload de um arquivo de música em MP3, descreva um estilo visual e veja a mágica acontecer!
        **Atenção:** O processo é intensivo e pode levar vários minutos.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            musica_input = gr.File(label="Arquivo de Música (MP3)", file_types=[".mp3"])
            estilo_input = gr.Textbox(
                label="Estilo Visual Detalhado", 
                value="Fotografia cinematográfica em preto e branco, filme noir, iluminação dramática, granulação de filme, 8k",
                info="Ex: 'Pintura a óleo impressionista', 'Arte de fantasia sombria', 'Cyberpunk neon'."
            )
            atraso_input = gr.Slider(
                minimum=0, 
                maximum=1000, 
                value=200, 
                step=50, 
                label="Atraso Visual (ms)",
                info="Ajuste para sincronizar a imagem com a palavra-chave falada."
            )
            run_button = gr.Button("Gerar Vídeo", variant="primary")
        
        with gr.Column(scale=2):
            video_output = gr.Video(label="Vídeo Final Gerado", interactive=False)

    run_button.click(
        fn=gerar_video_completo,
        inputs=[musica_input, estilo_input, atraso_input],
        outputs=[video_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("#### Pré-requisitos para Execução Local:")
    gr.Markdown("1. Instale todas as dependências do arquivo `requirements.txt`.\n"
                "2. Crie um arquivo `.env` com sua `GOOGLE_API_KEY`.\n"
                "3. Exporte a variável de ambiente `GOOGLE_APPLICATION_CREDENTIALS` apontando para seu arquivo de credenciais JSON.\n"
                "4. Exporte a variável de ambiente `GCP_PROJECT_ID` com o ID do seu projeto Google Cloud.\n"
                "5. Certifique-se que a pasta `rife_model` está presente.")


if __name__ == "__main__":
    # Carrega variáveis de ambiente de um arquivo .env (opcional, bom para desenvolvimento local)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Aviso: 'python-dotenv' não instalado. Lembre-se de definir as variáveis de ambiente manualmente.")

    # Verificação de pré-requisitos essenciais antes de iniciar
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GCP_PROJECT_ID"):
        print("ERRO CRÍTICO: As variáveis de ambiente GOOGLE_API_KEY, GOOGLE_APPLICATION_CREDENTIALS e GCP_PROJECT_ID são obrigatórias.")
        print("Por favor, configure-as e tente novamente.")
    else:
        print("Variáveis de ambiente encontradas. Iniciando a aplicação Gradio...")
        demo.launch(debug=True, share=False) # debug=True ajuda a ver erros no console; share=False para rodar localmente