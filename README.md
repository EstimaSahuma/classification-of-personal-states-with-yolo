# 👁️ EyeState Detector: Análise de Atenção em Tempo Real

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

**EyeState Detector** é um projeto de visão computacional avançado que analisa o estado de atenção de uma pessoa (focada, distraída, cansada, etc.) em tempo real, utilizando apenas a imagem do olho capturada por uma webcam.

Sua principal aplicação é na **segurança veicular**, onde pode ser usado para monitorar motoristas e gerar alertas caso sinais de distração ou sonolência sejam detectados, ajudando a prevenir acidentes de trânsito.

*(Sugestão: grave um GIF curto mostrando o programa funcionando e coloque aqui)*

---

## 📝 Índice

- [📌 Sobre o Projeto](#-sobre-o-projeto)
- [⚙️ Como Funciona](#️-como-funciona)
- [🚀 Instalação e Execução](#-instalação-e-execução)
- [🛠️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
- [🤝 Como Contribuir](#-como-contribuir)
- [📝 Licença](#-licença)

---

## 📌 Sobre o Projeto

O objetivo do **EyeState Detector** é criar um sistema robusto e não invasivo para inferir o estado cognitivo de uma pessoa. Ao invés de analisar expressões faciais complexas, o projeto se concentra nos micro-movimentos e padrões do olho, que são fortes indicadores de atenção e fadiga.

O sistema classifica o estado do usuário em categorias como:
- **Focado (Focus)**
- **Distraído (Distracted)**
- **Cansado (Tired)**
- E outros estados que podem ser treinados.

A tecnologia central é uma combinação de detecção de pontos-chave (keypoints) e uma abordagem bio-inspirada que simula uma **câmera de eventos (Event-based Camera)**, tornando o sistema mais resiliente a variações de iluminação.

## ⚙️ Como Funciona

O pipeline do projeto é dividido em várias etapas inteligentes que trabalham em conjunto para fornecer uma análise estável e precisa:

1.  **Detecção do Olho com YOLOv8-Pose**:
    - Utilizamos o modelo `yolov8n-pose.pt` para detectar os pontos-chave (keypoints) do corpo humano na imagem da webcam.
    - Focamos especificamente no keypoint do olho esquerdo (`LEFT_EYE_INDEX = 2`) para obter uma localização precisa.

2.  **Estabilização e Rastreamento**:
    - Para evitar que a caixa de detecção (bounding box) ao redor do olho "pule" a cada frame, aplicamos um **filtro de média móvel exponencial**. Isso suaviza a posição do centro do olho, resultando em um rastreamento mais estável.
    - Um mecanismo de persistência mantém a caixa de detecção visível por alguns frames mesmo se o olho for temporariamente perdido (ex: durante uma piscada).

3.  **Conversão para Câmera de Eventos (V2E - Video to Events)**:
    - Este é o coração do projeto. Ao invés de alimentar o classificador com a imagem RGB crua do olho, nós a processamos com a classe `V2E_Simulator_Pro`.
    - Esta classe simula o funcionamento de uma câmera de eventos, que captura apenas as **mudanças de luminosidade** em cada pixel.
    - Ela gera uma imagem em preto, branco e cinza, onde:
        - **Branco**: Pixels que ficaram mais claros (eventos "ON").
        - **Preto**: Pixels que ficaram mais escuros (eventos "OFF").
        - **Cinza**: Nenhuma mudança significativa.
    - Essa abordagem torna a classificação muito mais robusta a variações absolutas de iluminação e foca nos padrões de movimento do olho.

4.  **Classificação do Estado com YOLOv8-Clf**:
    - A imagem de eventos gerada na etapa anterior é então enviada para um segundo modelo YOLOv8, treinado especificamente para **classificação de imagens**.
    - Este modelo (`best.pt`) foi treinado com um dataset de imagens de eventos de olhos para aprender a diferenciar os estados de "focado", "distraído", etc.
    - O desempenho do classificador no conjunto de validação pode ser visualizado na **matriz de confusão normalizada** abaixo. Ela mostra a capacidade do modelo de distinguir corretamente entre as diferentes classes.

    ![Matriz de Confusão Normalizada](https://github.com/Netinhoklz/classification-of-personal-states-with-yolo/blob/main/train412/confusion_matrix_normalized.png?raw=true)
    
    *Como podemos observar, os valores na diagonal principal são altos, indicando uma alta taxa de acerto para cada classe. As confusões entre estados opostos (ex: `Focus` e `Tired`) são mínimas, validando a eficácia da abordagem.*

5.  **Sistema de Votação para Estabilidade**:
    - Classificações frame a frame podem ser voláteis. Para garantir um resultado final confiável, implementamos um **sistema de votação**.
    - As últimas `75` predições são armazenadas em uma fila (`deque`).
    - O estado final exibido ao usuário é a **moda** (a classe mais frequente) dessa janela de tempo, prevenindo "flickering" entre estados e fornecendo uma leitura muito mais estável e confiável.

6.  **Visualização em Tempo Real**:
    - O resultado é exibido na tela com:
        - Uma caixa colorida ao redor do olho (Verde para foco, Vermelho para distração).
        - O status atual e a confiança da votação.
        - Um cronômetro que registra o tempo gasto em cada estado.
        - Uma janela separada que mostra a visão da "câmera de eventos", permitindo visualizar exatamente o que o classificador está "vendo".

## 🚀 Instalação e Execução

Siga os passos abaixo para executar o projeto em sua máquina local.

### Pré-requisitos
- Python 3.9 ou superior
- Git
- Uma webcam conectada

### Passos

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Netinhoklz/classification-of-personal-states-with-yolo.git
    cd classification-of-personal-states-with-yolo
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo:
    ```txt
    opencv-python
    numpy
    ultralytics
    Pillow
    ```
    E então instale-as:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe os modelos YOLO:**
    - O `yolov8n-pose.pt` será baixado automaticamente pela biblioteca `ultralytics` na primeira execução.
    - **IMPORTANTE**: O modelo classificador treinado (`best.pt`) já deve estar no repositório ou você precisa baixá-lo. **Certifique-se de que o caminho no script está correto**:

    ```python
    # Altere esta linha no seu código, se necessário
    model_classifier = YOLO(r'C:\caminho\completo\para\o\seu\best.pt') 
    ```

5.  **Execute o script:**
    ```bash
    python seu_script.py
    ```
    *(Substitua `seu_script.py` pelo nome do seu arquivo Python principal)*

Pressione a tecla `q` com a janela do OpenCV em foco para fechar o programa.

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **OpenCV**: Para captura de vídeo e manipulação de imagens em tempo real.
- **Ultralytics YOLOv8**: Para detecção de pose e classificação de imagens, utilizando redes neurais de última geração.
- **NumPy**: Para computação numérica e manipulação de arrays de imagem.
- **Pillow (PIL)**: Para renderizar texto com fontes customizadas na tela.

---

## 🤝 Como Contribuir

Contribuições são o que tornam a comunidade de código aberto um lugar incrível para aprender, inspirar e criar. Qualquer contribuição que você fizer será **muito apreciada**.

1.  Faça um **Fork** do projeto.
2.  Crie uma **Branch** para sua feature (`git checkout -b feature/AmazingFeature`).
3.  Faça o **Commit** de suas mudanças (`git commit -m 'Add some AmazingFeature'`).
4.  Faça o **Push** para a Branch (`git push origin feature/AmazingFeature`).
5.  Abra um **Pull Request**.

---

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
