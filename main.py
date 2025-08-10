import os
import whisper
import google.generativeai as genai
#from openai import OpenAI
from google.cloud import texttospeech
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from spleeter.separator import Separator
import requests
import json
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import io

sys.path.append('rife_model')
from train_log.RIFE_HDv3 import Model
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, afx, ImageClip, concatenate_videoclips, CompositeVideoClip
import moviepy.video.fx.all as vfx # Importa os efeitos de vídeo
import time
import gc

if hasattr(Image, "Resampling") and not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

PASTA_ENTRADA = "entrada"
PASTA_SAIDA = "saida"
PASTA_TEMP = "temp"

REEXECUTAR_TUDO = False

NOME_ARQUIVO_MUSICA = "tontos_kelvins_duran.mp3"
caminho_musica = os.path.join(PASTA_ENTRADA, NOME_ARQUIVO_MUSICA)

TEMPO_RESPIRO_SEGUNDOS = 3

# Configuração de sincronização precisa palavra-imagem
ATRASO_VISUAL_MILISSEGUNDOS = 200  # Atraso em milissegundos após a palavra-chave
ATRASO_VISUAL_SEGUNDOS = ATRASO_VISUAL_MILISSEGUNDOS / 1000.0  # Conversão para segundos

def configurar_sincronizacao():
    """Permite ao usuário configurar o atraso visual"""
    global ATRASO_VISUAL_MILISSEGUNDOS, ATRASO_VISUAL_SEGUNDOS

    print(f"\n--- Configuração de Sincronização Palavra-Imagem ---")
    print(f"Atraso atual: {ATRASO_VISUAL_MILISSEGUNDOS}ms após a palavra-chave")

    resposta = input("Deseja alterar o atraso? (y/N): ").strip().lower()
    if resposta in ['y', 'yes', 's', 'sim']:
        try:
            novo_atraso = int(input("Digite o novo atraso em milissegundos (ex: 150, 300, 500): "))
            if 0 <= novo_atraso <= 2000:  # Limita entre 0 e 2 segundos
                ATRASO_VISUAL_MILISSEGUNDOS = novo_atraso
                ATRASO_VISUAL_SEGUNDOS = novo_atraso / 1000.0
                print(f"Atraso configurado para {novo_atraso}ms")
            else:
                print("Valor fora do intervalo permitido (0-2000ms). Usando padrão.")
        except ValueError:
            print("Valor inválido. Usando configuração padrão.")

    print(f"Atraso final: {ATRASO_VISUAL_MILISSEGUNDOS}ms ({ATRASO_VISUAL_SEGUNDOS:.3f}s)")
    return ATRASO_VISUAL_MILISSEGUNDOS

if not os.path.exists(caminho_musica):
    print(f"Erro: O arquivo de música '{caminho_musica}' não foi encontrado.")
    print("Por favor, adicione o arquivo de música na pasta 'entrada' e tente novamente.")
    exit()

print(f"Arquivo de música '{caminho_musica}' encontrado com sucesso.")
print("Iniciando o pipeline de processamento...")

# Configurar otimização CUDA simples
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configurar sincronização
configurar_sincronizacao()

caminho_letra = os.path.join(PASTA_TEMP, "letra.txt")
if REEXECUTAR_TUDO or not os.path.exists(caminho_letra):
    print("\n--- Etapa 2: Transcrevendo a letra da música para dar contexto à IA ---")
    modelo_whisper = None
    try:
        print("Carregando o modelo Whisper (medium) na CPU...")
        modelo_whisper = whisper.load_model("medium", device="cpu")

        print("Iniciando a transcrição do áudio...")
        resultado = modelo_whisper.transcribe(caminho_musica, language="pt", verbose=False)
        letra = resultado['text']

        with open(caminho_letra, "w", encoding="utf-8") as f:
            f.write(letra)
        print(f"Transcrição de letra concluída. Salva em '{caminho_letra}'")
    except Exception as e:
        print(f"Erro durante a transcrição com Whisper: {e}")
        exit()
    finally:
        if modelo_whisper is not None:
            print("Descarregando o modelo Whisper da memória...")
            del modelo_whisper
            gc.collect()
else:
    print(f"\nArquivo '{caminho_letra}' já existe. Pulando Etapa 2.")

caminho_roteiro_narracao = os.path.join(PASTA_TEMP, "roteiro_narracao.txt")
if REEXECUTAR_TUDO or not os.path.exists(caminho_roteiro_narracao):
    print("\n--- Etapa 3: Criando o roteiro para a narração ---")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
            exit()

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        with open(caminho_letra, "r", encoding="utf-8") as f:
            letra_musica = f.read()

        prompt_narracao = f"""
        Com base na seguinte letra de música, crie uma narração curta e analítica.
        A narração deve refletir sobre os sentimentos, a poesia e o significado da letra, como se fosse um ensaio poético.
        O tom deve ser introspectivo e calmo.
        A narração deve ter no máximo 100 palavras e estar em português. Me retorne APENAS a narração.

        Letra da música:
        ---
        {letra_musica}
        ---

        Roteiro para Narração:
        """
        print("Gerando roteiro para NARRAÇÃO...")

        # Retry logic para erros 500
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_narracao = model.generate_content(prompt_narracao)
                roteiro_narracao = response_narracao.text
                with open(caminho_roteiro_narracao, "w", encoding="utf-8") as f:
                    f.write(roteiro_narracao)
                print(f"Roteiro de narração salvo em '{caminho_roteiro_narracao}'")
                break
            except Exception as e:
                if "500" in str(e) and attempt < max_retries - 1:
                    print(f"Erro 500 na tentativa {attempt + 1}/{max_retries}. Aguardando 15 segundos e tentando novamente...")
                    time.sleep(15)
                else:
                    raise e

    except Exception as e:
        print(f"Erro durante a criação do roteiro de narração: {e}")
        exit()
else:
    print(f"\nArquivo de roteiro de narração já existe. Pulando Etapa 3.")

caminho_narracao_audio = os.path.join(PASTA_TEMP, "narracao.mp3")
if REEXECUTAR_TUDO or not os.path.exists(caminho_narracao_audio):
    print("\n--- Etapa 4: Gerando a narração em áudio ---")
    try:
        def gerar_narracao_com_google(texto_narracao, caminho_saida_audio):
            """Gera o áudio de narração usando a API Google Cloud Text-to-Speech."""
            try:
                print("Enviando texto para a API de TTS do Google Cloud...")

                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text=texto_narracao)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="pt-BR",
                    name="pt-BR-Wavenet-D"
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                with open(caminho_saida_audio, "wb") as out:
                    out.write(response.audio_content)
                    print(f"Áudio da narração salvo em '{caminho_saida_audio}'")

            except Exception as e:
                print(f"Ocorreu um erro durante a geração de áudio com o Google TTS: {e}")
                exit()

        with open(caminho_roteiro_narracao, "r", encoding="utf-8") as f:
            texto_narracao = f.read()

        gerar_narracao_com_google(texto_narracao, caminho_narracao_audio)
        print(f"Áudio da narração gerado com sucesso e salvo em '{caminho_narracao_audio}'")

    except Exception as e:
        print(f"Erro durante a geração do áudio de narração: {e}")
        exit()
else:
    print(f"\nArquivo '{caminho_narracao_audio}' já existe. Pulando Etapa 4.")

def criar_prompt_para_imagem_com_estilo(descricao_visual, estilo_usuario, trecho_letra=None):
    """
    Usa o Gemini para criar um prompt de imagem detalhado a partir de uma descrição visual, um estilo e trecho da letra.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
            return f"{descricao_visual}, in the style of {estilo_usuario}"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Incluir trecho da letra se disponível
        contexto_letra = f"\n\n        **Trecho da Letra da Música (contexto importante):**\n        \"{trecho_letra}\"" if trecho_letra else ""

        prompt_template = f"""
        Crie um prompt de imagem para uma IA generativa de imagens (como Imagen ou Midjourney).
        O prompt deve ser detalhado, cinematográfico e capturar a essência da seguinte descrição visual,
        seguindo o estilo artístico fornecido e refletindo o sentimento do trecho da música.

        **Descrição Visual:**
        "{descricao_visual}"

        **Estilo Artístico Desejado:**
        "{estilo_usuario}"{contexto_letra}

        **Instruções para o Prompt:**
        1. O prompt deve ser em inglês para maximizar a compatibilidade com os modelos de imagem.
        2. Traduza a essência e os elementos principais da descrição para o inglês.
        3. Incorpore o estilo artístico de forma proeminente no prompt.
        4. SE HOUVER um trecho da letra fornecido, use-o como contexto principal para inspirar a imagem, garantindo que a imagem reflita o sentimento e as palavras da música.
        5. Adicione detalhes descritivos que tornem a cena mais viva, como iluminação, composição, emoção e atmosfera.
        6. O resultado final deve ser apenas o prompt de imagem, sem nenhum texto ou explicação adicional.

        **Exemplo de Saída (se a descrição for sobre um mar solitário e o estilo for 'pintura a óleo'):**
        "Oil painting of a vast, lonely sea under a stormy sky, dramatic light breaking through the clouds, crashing waves against black rocks, sense of melancholy and solitude, hyper-detailed, 8k."

        **Sua Tarefa:**
        Agora, gere o prompt de imagem para a descrição e estilo fornecidos.
        """

        response = model.generate_content(prompt_template)

        # Adicionando um fallback e limpeza da resposta
        if response and response.text:
            # Remove aspas ou palavras como "Prompt:" que a IA possa adicionar
            prompt_final = response.text.strip().replace('"', '').replace("Prompt:", "").replace("prompt:", "").strip()
            return prompt_final
        else:
            # Fallback para um prompt mais simples caso a geração falhe
            return f"{descricao_visual}, in the style of {estilo_usuario}"

    except Exception as e:
        print(f"Erro ao criar prompt estilizado: {e}")
        return f"{descricao_visual}, in the style of {estilo_usuario}"

caminho_segmentos_narracao = os.path.join(PASTA_TEMP, "segmentos_narracao.json")
caminho_palavras_narracao = os.path.join(PASTA_TEMP, "palavras_narracao.json")

if REEXECUTAR_TUDO or not os.path.exists(caminho_segmentos_narracao) or not os.path.exists(caminho_palavras_narracao):
    print("\n--- Etapa 5: Analisando o áudio da narração para extrair segmentos e palavras ---")
    modelo_whisper = None
    try:
        print("Carregando o modelo Whisper (medium) na CPU para análise da narração com timestamps de palavras...")
        modelo_whisper = whisper.load_model("medium", device="cpu")
        print("Analisando a narração com timestamps de palavras (isso pode demorar um pouco)...")

        # Transcrição com timestamps de palavras
        resultado = modelo_whisper.transcribe(
            caminho_narracao_audio,
            language="pt",
            word_timestamps=True,
            verbose=False
        )

        segmentos_narracao = resultado['segments']

        # Extrair todas as palavras com timestamps
        palavras_com_tempo = []
        for segmento in segmentos_narracao:
            if 'words' in segmento:
                for palavra_info in segmento['words']:
                    palavras_com_tempo.append({
                        'word': palavra_info['word'].strip(),
                        'start': palavra_info['start'],
                        'end': palavra_info['end'],
                        'segment_id': segmento['id']
                    })

        with open(caminho_segmentos_narracao, 'w', encoding='utf-8') as f:
            json.dump(segmentos_narracao, f, ensure_ascii=False, indent=4)

        with open(caminho_palavras_narracao, 'w', encoding='utf-8') as f:
            json.dump(palavras_com_tempo, f, ensure_ascii=False, indent=4)

        print(f"Segmentos da narração salvos em '{caminho_segmentos_narracao}'")
        print(f"Palavras com timestamps salvos em '{caminho_palavras_narracao}' ({len(palavras_com_tempo)} palavras)")
    except Exception as e:
        print(f"Erro durante a análise da narração: {e}")
        exit()
    finally:
        if modelo_whisper is not None:
            print("Descarregando o modelo Whisper da memória...")
            del modelo_whisper
            gc.collect()
else:
    print(f"\nArquivos de segmentos e palavras já existem. Pulando Etapa 5.")

caminho_roteiro_estruturado = os.path.join(PASTA_TEMP, "roteiro_estruturado.json")
if REEXECUTAR_TUDO or not os.path.exists(caminho_roteiro_estruturado):
    print("\n--- Etapa 6: Criando roteiro visual sincronizado com palavras-chave ---")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
            exit()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        with open(caminho_segmentos_narracao, "r", encoding="utf-8") as f:
            segmentos_narracao = json.load(f)

        with open(caminho_palavras_narracao, "r", encoding="utf-8") as f:
            palavras_com_tempo = json.load(f)

        # Carregar a letra original para contextualizaçao
        with open(caminho_letra, "r", encoding="utf-8") as f:
            letra_original = f.read()

        prompt_visual_estruturado = f"""
        Como diretor de arte, crie descrições visuais que representem LITERALMENTE as palavras ditas na narração.

        INSTRUÇÃO PRINCIPAL: Use as palavras EXATAS da narração para criar as imagens. Se a narração fala "solidão", mostre uma pessoa sozinha. Se fala "memórias", mostre fotografias ou lembranças. Se fala "amor", mostre pessoas se amando. Seja direto e literal, não metafórico.

        Crie um roteiro visual em formato JSON. Para cada segmento, gere um objeto com:
        1. "id", "start", "end", "text": Copie exatamente do segmento original.
        2. "descricao_visual": Descreva uma cena que mostre LITERALMENTE o que está sendo dito no texto da narração. Use os substantivos, verbos e conceitos concretos mencionados. Evite simbolismo - seja direto.
        3. "palavra_chave_visual": A palavra mais importante do segmento para gerar a imagem.

        EXEMPLOS de como ser literal:
        - Narração: "A solidão ecoa no silêncio" → Imagem: "pessoa sozinha em quarto silencioso, expressão melancólica"
        - Narração: "Memórias se desvanecem" → Imagem: "fotografias antigas espalhadas, algumas desbotadas"
        - Narração: "O tempo passa devagar" → Imagem: "relógio antigo, ponteiros se movendo lentamente"

        Segmentos da Narração:
        ---
        {json.dumps(segmentos_narracao, indent=2, ensure_ascii=False)}
        ---

        Retorne APENAS o JSON como uma lista de objetos.
        """

        # Retry logic para erros 500
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Gerando roteiro VISUAL ESTRUTURADO com palavras-chave... (tentativa {attempt + 1}/{max_retries})")
                response_visual = model.generate_content(prompt_visual_estruturado)

                texto_json = response_visual.text.strip()
                if texto_json.startswith("```json"):
                    texto_json = texto_json[len("```json"):]
                if texto_json.endswith("```"):
                    texto_json = texto_json[:-3]

                roteiro_json = json.loads(texto_json.strip())

                # Mapear palavras-chave com seus timestamps
                def encontrar_timestamp_palavra(palavra_chave, segmento_id):
                    """Encontra o timestamp da palavra-chave no segmento"""
                    palavras_segmento = [p for p in palavras_com_tempo if p['segment_id'] == segmento_id]

                    # Busca exata primeiro
                    for palavra_info in palavras_segmento:
                        if palavra_chave.lower() in palavra_info['word'].lower():
                            return palavra_info['start']

                    # Se não encontrar, usa o início do segmento
                    return next((s['start'] for s in segmentos_narracao if s['id'] == segmento_id), 0)

                # Adicionar timestamp da palavra-chave a cada cena
                for cena in roteiro_json:
                    if 'palavra_chave_visual' in cena:
                        timestamp_palavra = encontrar_timestamp_palavra(
                            cena['palavra_chave_visual'],
                            cena['id']
                        )
                        cena['timestamp_palavra_chave'] = timestamp_palavra
                        print(f"Cena {cena['id']}: palavra-chave '{cena['palavra_chave_visual']}' detectada em {timestamp_palavra:.3f}s")

                with open(caminho_roteiro_estruturado, "w", encoding="utf-8") as f:
                    json.dump(roteiro_json, f, ensure_ascii=False, indent=4)
                print(f"Roteiro visual estruturado salvo em '{caminho_roteiro_estruturado}'")
                break
            except Exception as e:
                if "500" in str(e) and attempt < max_retries - 1:
                    print(f"Erro 500 na tentativa {attempt + 1}/{max_retries}. Aguardando 15 segundos e tentando novamente...")
                    time.sleep(15)
                else:
                    raise e

    except Exception as e:
        print(f"Erro durante a criação do roteiro estruturado: {e}")
        exit()
else:
    print(f"\nArquivo '{caminho_roteiro_estruturado}' já existe. Pulando Etapa 6.")

# --- Solicitar Estilo ao Usuário ---
print("\n--- Definição do Estilo Visual ---")
estilo_artistico_usuario = input("Digite o estilo desejado para as imagens (ex: 'Fotografia preto e branco, filme noir', 'Pintura a óleo impressionista', 'Arte digital de fantasia sombria'): ")
if not estilo_artistico_usuario.strip():
    estilo_artistico_usuario = "cinematic, hyper-realistic, dramatic lighting, 8k"  # Estilo padrão
    print(f"Nenhum estilo inserido. Usando estilo padrão: {estilo_artistico_usuario}")
else:
    print(f"Estilo selecionado: {estilo_artistico_usuario}")

# Substitua o bloco da Etapa 7 por este:
print("\n--- Etapa 7: Gerando imagens com o Imagen do Google ---")

try:
    # Substitua pelo ID do seu projeto e uma localização válida
    # A localização precisa ser uma onde o modelo Imagen está disponível, como "us-central1"
    vertexai.init(project="gen-lang-client-0537101315", location="us-central1")

    # Carrega o modelo de geração de imagem
    generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")

    with open(os.path.join(PASTA_TEMP, "roteiro_estruturado.json"), "r", encoding="utf-8") as f:
        cenas_roteiro = json.load(f)

    NUM_IMAGENS = len(cenas_roteiro)
    caminhos_imagens = [os.path.join(PASTA_TEMP, f"img_{i+1}.jpg") for i in range(NUM_IMAGENS)]
    imagens_existem = all(os.path.exists(p) for p in caminhos_imagens)

    if REEXECUTAR_TUDO or not imagens_existem:
        print(f"Serão geradas {NUM_IMAGENS} imagens com o Imagen.")

        for i, cena in enumerate(cenas_roteiro):
            caminho_imagem_cena = caminhos_imagens[i]
            if os.path.exists(caminho_imagem_cena) and not REEXECUTAR_TUDO:
                print(f"Imagem para a cena {i+1} já existe. Pulando.")
                continue

            descricao = cena['descricao_visual']

            # Buscar o trecho correspondente da letra para esta cena
            trecho_letra_cena = None
            if 'trecho_letra' in cena:
                trecho_letra_cena = cena['trecho_letra']
            elif 'text' in cena:
                trecho_letra_cena = cena['text']

            # Usar nossa nova função para criar um prompt estilizado
            prompt = criar_prompt_para_imagem_com_estilo(descricao, estilo_artistico_usuario, trecho_letra_cena)

            print(f"Cena {i+1}/{NUM_IMAGENS} - Descrição: '{descricao}'")
            if trecho_letra_cena:
                print(f"  > Conectando com a letra: '{trecho_letra_cena[:100]}{'...' if len(trecho_letra_cena) > 100 else ''}'")
            print(f"  > Prompt final estilizado: '{prompt[:150]}{'...' if len(prompt) > 150 else ''}'")
            print("Enviando solicitação para a API do Imagen (Vertex AI)...")

            response = generation_model.generate_images(
                prompt=prompt,
                number_of_images=1
            )

            if response.images and len(response.images) > 0:
                response.images[0].save(location=caminho_imagem_cena)
                print(f"Imagem salva em '{caminho_imagem_cena}'.")
            else:
                print(f"Erro: Nenhuma imagem foi gerada para a cena {i+1}. Verifique as credenciais e permissões.")
                continue

            # Aguarda 10 segundos antes da próxima imagem para evitar rate limiting
            if i < len(cenas_roteiro) - 1:  # Não espera após a última imagem
                print(f"Aguardando 10 segundos antes da próxima imagem...")
                time.sleep(10)

    else:
        print("Todas as imagens já existem. Pulando a Etapa 7.")

except Exception as e:
    print(f"Ocorreu um erro durante a geração de imagens com o Imagen: {e}")
    print("Certifique-se de que a variável de ambiente 'GOOGLE_APPLICATION_CREDENTIALS' está definida corretamente,")
    print("apontando para o seu arquivo de chave JSON.")
    exit()


print("\n--- Etapa 8: Criando transições dinâmicas entre as imagens ---")

deve_executar_etapa_8 = True
if not os.path.exists(os.path.join(PASTA_TEMP, "img_1.jpg")):
    print("AVISO: Nenhuma imagem encontrada, impossível criar transições. Pulando Etapa 8.")
    deve_executar_etapa_8 = False
elif os.path.exists(os.path.join(PASTA_TEMP, "trans_1_01.jpg")) and not REEXECUTAR_TUDO:
    print("AVISO: Transições já parecem existir. Pulando Etapa 8.")
    deve_executar_etapa_8 = False

if deve_executar_etapa_8:
    try:
        # Limpeza completa de memória CUDA antes das transições
        print("🧹 Limpando memória CUDA antes das transições...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            memoria_livre = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            print(f"   Memória CUDA livre: {memoria_livre:.2f} GB")

        with open(os.path.join(PASTA_TEMP, "roteiro_estruturado.json"), "r", encoding="utf-8") as f:
            cenas_roteiro = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        model = Model()
        model.load_model('rife_model/train_log', -1)
        model.eval()
        model.device()

        def make_inference_generator(img1, img2, n_recursions):
            if n_recursions > 0:
                middle_frame = model.inference(img1, img2)
                yield from make_inference_generator(img1, middle_frame, n_recursions - 1)
                yield middle_frame
                yield from make_inference_generator(middle_frame, img2, n_recursions - 1)

        transform = transforms.ToTensor()

        for i in range(len(cenas_roteiro) - 1):
            # Limpeza preventiva a cada transição
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            cena_atual = cenas_roteiro[i]
            img1_path = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
            img2_path = os.path.join(PASTA_TEMP, f"img_{i+2}.jpg")

            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"AVISO: Pulando transição da cena {i+1} -> {i+2} por falta de uma das imagens.")
                continue

            duracao_cena = cena_atual['end'] - cena_atual['start']
            # Reduzir recursions para economizar memória
            if duracao_cena < 2.0: n_recursions = 5  # Mais conservador
            elif duracao_cena < 4.0: n_recursions = 6  # Mais conservador
            else: n_recursions = 7  # Mais conservador
            num_frames_a_gerar = (2**n_recursions) - 1

            print(f"\nGerando transição para {os.path.basename(img1_path)} -> {os.path.basename(img2_path)} ({num_frames_a_gerar} quadros)")

            img1 = Image.open(img1_path).convert("RGB"); img2 = Image.open(img2_path).convert("RGB")
            w, h = img1.size; w_new = (w // 32) * 32; h_new = (h // 32) * 32
            img1 = img1.resize((w_new, h_new)); img2 = img2.resize((w_new, h_new))
            img1_tensor = transform(img1).unsqueeze(0).to(device); img2_tensor = transform(img2).unsqueeze(0).to(device)

            frame_generator = make_inference_generator(img1_tensor, img2_tensor, n_recursions)

            for j, frame_tensor in enumerate(frame_generator):
                frame_image = transforms.ToPILImage()(frame_tensor.squeeze(0).cpu())
                caminho_frame = os.path.join(PASTA_TEMP, f"trans_{i+1}_{j+1:02d}.jpg")
                frame_image.save(caminho_frame)

            print(f"Limpando memória da GPU após transição {i+1}...")
            try:
                del img1_tensor, img2_tensor, frames_tensor
            except:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

        print("\nGeração de todos os quadros de transição concluída.")
    except Exception as e:
        print(f"Erro durante a criação de transições: {e}")
        # Limpeza de emergência
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Continuando sem transições...")


print("\n--- Etapa 8.5: Removendo vocais da música com Spleeter ---")
nome_arquivo_musica = os.path.splitext(os.path.basename(caminho_musica))[0]
caminho_musica_instrumental = os.path.join(PASTA_TEMP, nome_arquivo_musica, "accompaniment.wav")

if REEXECUTAR_TUDO or not os.path.exists(caminho_musica_instrumental):
    try:
        print("Inicializando o separador Spleeter (pode demorar na primeira vez)...")
        # Usando 2 stems (vocal + acompanhamento) para separar em vocal e instrumental
        separator = Separator('spleeter:2stems')

        print(f"Processando a música '{caminho_musica}' para separar os vocais...")
        # A saída será salva em uma subpasta dentro de PASTA_TEMP/
        separator.separate_to_file(caminho_musica, PASTA_TEMP, codec='wav')

        print(f"Faixa instrumental salva em '{caminho_musica_instrumental}'")

    except Exception as e:
        print(f"Erro durante a separação de vocais com Spleeter: {e}")
        exit()
else:
    print(f"Arquivo de música instrumental '{caminho_musica_instrumental}' já existe. Pulando Etapa 8.5.")


print("\n--- Etapa 9: Montando o vídeo final ---")
def generate_frames_recursively(model, img1, img2, n_recursions):
    """Gerador recursivo para interpolação de frames com RIFE."""
    if n_recursions == 0:
        return
    middle_frame = model.inference(img1, img2)
    yield from generate_frames_recursively(model, img1, middle_frame, n_recursions - 1)
    yield middle_frame
    yield from generate_frames_recursively(model, middle_frame, img2, n_recursions - 1)

caminho_video_final = os.path.join(PASTA_SAIDA, "video_final.mp4")

if REEXECUTAR_TUDO or not os.path.exists(caminho_video_final):
    try:
        print("--- Etapa 9: Montando o vídeo final ---")

        print("Carregando áudios e roteiro estruturado para montagem final...")
        audio_narracao = AudioFileClip(os.path.join(PASTA_TEMP, "narracao.mp3"))
        audio_musica = AudioFileClip(caminho_musica_instrumental).volumex(0.1)
        with open(os.path.join(PASTA_TEMP, "roteiro_estruturado.json"), "r", encoding="utf-8") as f:
            cenas_roteiro = json.load(f)

        print("Preparando clipes de vídeo sincronizados com a narração...")
        lista_de_clipes_visuais = []
        duracao_total_cenas = 0

        tamanho_padrao, primeira_imagem_encontrada = None, None
        for i, cena in enumerate(cenas_roteiro):
            caminho_img_cena = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
            if os.path.exists(caminho_img_cena):
                tamanho_padrao = Image.open(caminho_img_cena).size
                primeira_imagem_encontrada = caminho_img_cena
                break
        if not tamanho_padrao:
            print("ERRO CRÍTICO: Nenhuma imagem foi encontrada. Impossível continuar.")
            exit()

        print(f"Tamanho de quadro padrão para o vídeo: {tamanho_padrao}")

        clipes_posicionados = []
        tempo_cursor = 0.0

        # Primeiro, identifica quais cenas têm imagens válidas
        cenas_validas = []
        for i, cena in enumerate(cenas_roteiro):
            caminho_img_principal = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
            if os.path.exists(caminho_img_principal) and os.path.getsize(caminho_img_principal) > 0:
                cenas_validas.append((i, cena, caminho_img_principal))
            else:
                print(f"AVISO: Imagem da cena {i+1} não encontrada ou inválida. Pulando.")

        print(f"Processando {len(cenas_validas)} cenas válidas de {len(cenas_roteiro)} totais.")

        for idx_valida, (i, cena, caminho_img_principal) in enumerate(cenas_validas):
            frames_da_cena = []
            frames_da_cena.append(caminho_img_principal)
            print(f"Processando cena {i+1} (válida {idx_valida+1}/{len(cenas_validas)})")
            # Só adiciona transições se há uma próxima cena válida
            if idx_valida < len(cenas_validas) - 1:
                proxima_cena_valida = cenas_validas[idx_valida + 1]
                proxima_cena_index = proxima_cena_valida[0]

                # Verifica se existe transição entre as cenas atuais
                j = 1
                frames_transicao_adicionados = 0
                while True:
                    caminho_trans = os.path.join(PASTA_TEMP, f"trans_{i+1}_{j:02d}.jpg")
                    if os.path.exists(caminho_trans) and os.path.getsize(caminho_trans) > 0:
                        frames_da_cena.append(caminho_trans)
                        frames_transicao_adicionados += 1
                        j += 1
                    else:
                        break
                print(f"Cena {i+1}: {frames_transicao_adicionados} frames de transição adicionados")

            # Verifica e redimensiona frames se necessário (sem modificar originais)
            frames_verificados = []
            for k, caminho_quadro in enumerate(frames_da_cena):
                try:
                    img = Image.open(caminho_quadro)
                    # Verifica se a imagem não está corrompida
                    img.verify()
                    # Recarrega para uso após verify()
                    img = Image.open(caminho_quadro)
                    if img.size != tamanho_padrao:
                        img = img.resize(tamanho_padrao, Image.Resampling.LANCZOS)
                    frames_verificados.append(img)
                except Exception as e:
                    print(f"AVISO: Frame {caminho_quadro} corrompido ou inválido: {e}")
                    # Pula frames corrompidos em vez de usar
                    continue

            if not frames_verificados:
                print(f"ERRO: Nenhum frame válido encontrado para cena {i+1}")
                continue

            # Salva frames verificados temporariamente
            frames_da_cena_corrigidos = []
            for k, img in enumerate(frames_verificados):
                caminho_temp = os.path.join(PASTA_TEMP, f"temp_cena_{i+1}_frame_{k}.jpg")
                img.save(caminho_temp, "JPEG", quality=95)
                frames_da_cena_corrigidos.append(caminho_temp)

            frames_da_cena = frames_da_cena_corrigidos

            duracao_animacao = cena['end'] - cena['start']
            if duracao_animacao <= 0: duracao_animacao = 0.1

            # Usa FPS fixo para evitar problemas de sincronização
            fps_cena = 30.0  # FPS fixo para consistência

            print(f"Cena {i+1}: {len(frames_da_cena)} frames, duração {duracao_animacao:.2f}s, FPS {fps_cena:.2f}")

            try:
                clipe_animado = ImageSequenceClip(frames_da_cena, fps=fps_cena)
                # Define duração explícita para evitar problemas
                clipe_animado = clipe_animado.set_duration(duracao_animacao)
            except Exception as e:
                print(f"ERRO ao criar clipe para cena {i+1}: {e}")
                # Fallback: usa apenas a primeira imagem como clipe estático
                clipe_animado = ImageClip(frames_da_cena[0], duration=duracao_animacao)

            # Calcula duração considerando apenas cenas válidas
            if idx_valida < len(cenas_validas) - 1:
                proxima_cena_valida = cenas_validas[idx_valida + 1][1]  # Pega o objeto cena
                duracao_total_clipe = (proxima_cena_valida['start'] - cena['start'])
            else:
                duracao_total_clipe = clipe_animado.duration

            # Ajusta duração de forma mais robusta
            if abs(duracao_total_clipe - clipe_animado.duration) > 0.01:
                # Em vez de freeze, simplesmente define a duração desejada
                clipe_com_pausa = clipe_animado.set_duration(duracao_total_clipe)
                print(f"Cena {i+1}: Ajustando duração de {clipe_animado.duration:.2f}s para {duracao_total_clipe:.2f}s")
            else:
                clipe_com_pausa = clipe_animado
                print(f"Cena {i+1}: Duração OK ({clipe_animado.duration:.2f}s)")

            # Implementa sincronização inteligente sem gaps
            if 'timestamp_palavra_chave' in cena:
                # Calcula timing baseado na palavra-chave
                inicio_palavra = cena['timestamp_palavra_chave']
                inicio_sincronizado = inicio_palavra + ATRASO_VISUAL_SEGUNDOS
                palavra_chave = cena.get('palavra_chave_visual', 'N/A')
                print(f"Cena {i+1}: 📍 Palavra '{palavra_chave}' em {inicio_palavra:.3f}s → imagem em {inicio_sincronizado:.3f}s")

                # Armazena info de sincronização mas usa posicionamento sequencial
                cena['inicio_sincronizado'] = inicio_sincronizado
            else:
                print(f"Cena {i+1}: ⚠️  Sem palavra-chave, usando sequencial")

            # Usa posicionamento sequencial para evitar gaps
            inicio_real = tempo_cursor
            clipe_posicionado = clipe_com_pausa.set_start(0)  # Sem posicionamento absoluto
            clipes_posicionados.append(clipe_posicionado)

            print(f"         ✅ Clipe sequencial: duração {clipe_com_pausa.duration:.2f}s")
            tempo_cursor += duracao_total_clipe

        if not clipes_posicionados:
            print("ERRO CRÍTICO: Nenhum clipe visual pôde ser criado.")
            exit()

        print(f"\n=== DEBUG: Informações dos clipes antes da composição ===")
        for idx, clipe in enumerate(clipes_posicionados):
            print(f"Clipe {idx+1}: início={clipe.start:.2f}s, fim={(clipe.start + clipe.duration):.2f}s, duração={clipe.duration:.2f}s")
            print(f"  Tamanho: {clipe.size}, FPS: {clipe.fps}")

        # Calcula duração total baseada nos clipes posicionados
        duracao_total_cenas = max(clipe.start + clipe.duration for clipe in clipes_posicionados) if clipes_posicionados else 0
        print(f"\n=== Criando CompositeVideoClip ===")
        print(f"Número total de clipes: {len(clipes_posicionados)}")
        print(f"Tamanho padrão: {tamanho_padrao}")
        print(f"Duração total das cenas: {duracao_total_cenas:.2f}s")

        try:
            # Concatena clipes sequencialmente - sem gaps, sem imagens pretas
            video_clip_animado = concatenate_videoclips(clipes_posicionados, method="compose")
            print(f"Video concatenado criado com sucesso: {video_clip_animado.duration:.2f}s")

            # Aplica sincronização através de áudio deslocado em vez de vídeo
            # Isso mantém a continuidade visual mas sincroniza o áudio
            ajuste_audio_necessario = False
            for i, cena in enumerate(cenas_roteiro):
                if 'inicio_sincronizado' in cena and i < len(clipes_posicionados):
                    ajuste_audio_necessario = True
                    break

            if ajuste_audio_necessario:
                print("🎵 Aplicando sincronização através de ajuste de áudio...")

        except Exception as e:
            print(f"ERRO na criação do vídeo concatenado: {e}")
            # Diagnóstico individual dos clipes
            for idx, clipe in enumerate(clipes_posicionados):
                try:
                    frame_teste = clipe.get_frame(0)
                    print(f"Clipe {idx+1}: OK (shape: {frame_teste.shape})")
                except Exception as clipe_error:
                    print(f"Clipe {idx+1}: ERRO - {clipe_error}")
            raise

        caminho_ultima_imagem_existente = None
        for i in reversed(range(len(cenas_roteiro))):
            caminho_img = os.path.join(PASTA_TEMP, f"img_{i+1}.jpg")
            if os.path.exists(caminho_img):
                caminho_ultima_imagem_existente = caminho_img
                break

        if not primeira_imagem_encontrada or not caminho_ultima_imagem_existente:
             print("ERRO CRÍTICO: Imagem de início ou fim para os respiros não encontrada.")
             exit()

        clipe_inicio = ImageClip(primeira_imagem_encontrada, duration=TEMPO_RESPIRO_SEGUNDOS).resize(tamanho_padrao)
        clipe_fim = ImageClip(caminho_ultima_imagem_existente, duration=TEMPO_RESPIRO_SEGUNDOS).resize(tamanho_padrao)
        video_clip_visual_total = concatenate_videoclips([clipe_inicio, video_clip_animado, clipe_fim], method="compose")

        duracao_total_video = video_clip_visual_total.duration
        print(f"Duração total do vídeo: {duracao_total_video:.2f}s")
        print(f"🎯 Sincronização: Sistema híbrido - vídeo sequencial + áudio ajustado")

        # Estatísticas de sincronização
        cenas_com_sync = sum(1 for cena in cenas_roteiro if 'inicio_sincronizado' in cena)
        print(f"📊 Estatísticas: {cenas_com_sync}/{len(cenas_roteiro)} cenas com timestamps de palavras-chave")

        # Ajusta posicionamento do áudio da narração para sincronizar
        inicio_narracao = TEMPO_RESPIRO_SEGUNDOS
        if cenas_com_sync > 0:
            # Calcula offset médio necessário para melhor sincronização
            offsets = []
            tempo_video_atual = TEMPO_RESPIRO_SEGUNDOS
            for i, cena in enumerate(cenas_roteiro):
                if 'inicio_sincronizado' in cena and i < len(clipes_posicionados):
                    tempo_ideal = cena['inicio_sincronizado'] + TEMPO_RESPIRO_SEGUNDOS
                    offset = tempo_ideal - tempo_video_atual
                    offsets.append(offset)
                if i < len(clipes_posicionados):
                    tempo_video_atual += clipes_posicionados[i].duration

            if offsets:
                offset_medio = sum(offsets) / len(offsets)
                inicio_narracao += offset_medio
                print(f"🎵 Áudio ajustado com offset de {offset_medio:.3f}s para melhor sincronização")

        audio_narracao_posicionado = audio_narracao.set_start(inicio_narracao)
        audio_musica_fundo = audio_musica.fx(afx.audio_fadein, TEMPO_RESPIRO_SEGUNDOS).fx(afx.audio_fadeout, TEMPO_RESPIRO_SEGUNDOS)
        audio_musica_fundo = audio_musica_fundo.set_duration(duracao_total_video)
        audio_final = CompositeAudioClip([audio_narracao_posicionado, audio_musica_fundo])

        video_clip_final = video_clip_visual_total.set_audio(audio_final)

        print(f"Renderizando o vídeo final em '{caminho_video_final}'...")
        video_clip_final.write_videofile(caminho_video_final, codec='libx264', audio_codec='aac', fps=24, preset='medium', ffmpeg_params=['-pix_fmt', 'yuv420p'])

        print("\nPROJETO CONCLUÍDO!")
        print(f"O vídeo final foi salvo em: {caminho_video_final}")

        # Limpeza de arquivos temporários para evitar problemas em execuções futuras
        print("\nLimpando arquivos temporários...")
        import glob
        temp_files = glob.glob(os.path.join(PASTA_TEMP, "temp_cena_*.jpg"))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        print(f"Removidos {len(temp_files)} arquivos temporários.")

    except Exception as e:
        print(f"Erro durante a montagem do vídeo: {e}")
        exit()
else:
    print(f"\nVídeo final '{caminho_video_final}' já existe. Pulando Etapa 9.")
