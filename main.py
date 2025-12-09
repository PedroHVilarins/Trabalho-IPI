"""
Análise Morfológica de Imagens para Detecção de Bordas Endocárdicas do 
Ventrículo Esquerdo em Ecocardiogramas 2D

Implementação baseada em:
Choy, M. M., & Jin, J. S. (1996). Morphological image analysis of left-ventricular 
endocardial borders in 2D echocardiograms. SPIE Vol. 2710, pp. 852-863.

Pipeline (Figura 1 do paper):
1. Imagem Original
2. Filtragem Morfológica (redução de ruído usando conceito de elevação)
3. Filtragem LoG (detecção de bordas)
4. Segmentação Watershed (contorno inicial)
5. Extração de Contorno (borda final)
6. Resultado

Dataset: CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)

Autor: Vilarins
Data: Dezembro 2024
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import gaussian_laplace, label, binary_dilation, binary_erosion
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import watershed
from skimage.morphology import disk, reconstruction, remove_small_objects
from skimage.morphology import skeletonize, binary_opening
from skimage.filters import sobel
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AnalisadorMorfologicoEco:
    """
    Método semi-automático para detecção de bordas em imagens ecocardiográficas.
    
    Baseado em Choy & Jin (1996):
    - Filtragem morfológica para redução de ruído (baseada em elevação)
    - Filtragem LoG para detecção de bordas
    - Segmentação watershed para contorno inicial
    - Extração de contorno com refinamento por zero-crossings
    """
    
    def __init__(self, limiar_elevacao: int = 25, sigma_log: float = 3.0):
        """
        Argumentos:
            limiar_elevacao: Limiar para redução de ruído baseada em elevação (paper: 15-55)
            sigma_log: Desvio padrão para o filtro LoG
        """
        self.limiar_elevacao = limiar_elevacao
        self.sigma_log = sigma_log
        
    def carregar_imagem_nifti(self, caminho: str) -> np.ndarray:
        """Carrega imagem NIfTI."""
        img = nib.load(caminho)
        dados = img.get_fdata()
        if dados.ndim > 2:
            dados = dados[:, :, 0]
        return dados.astype(np.float64)
    
    def carregar_ground_truth(self, caminho: str) -> np.ndarray:
        """Carrega segmentação ground truth."""
        img = nib.load(caminho)
        dados = img.get_fdata()
        if dados.ndim > 2:
            dados = dados[:, :, 0]
        return dados.astype(np.int32)
    
    # =========================================================================
    # PASSO 1: FILTRAGEM MORFOLÓGICA
    # Seção 2.2 do paper: "Morphological filtering"
    # =========================================================================
    
    def filtragem_morfologica(self, imagem: np.ndarray) -> np.ndarray:
        """
        Aplica filtragem morfológica para redução de ruído.
        
        Do paper Seção 2.2:
        "Considere o gráfico de uma imagem de eco 2D como uma superfície topográfica...
        Podemos aumentar o contraste da imagem resetando a intensidade de pontos
        com elevação menor que um limiar."
        
        A elevação de um ponto é: altura(x) - altura(minimo(bacia(x)))
        Pontos com baixa elevação (ruído na cavidade) são zerados.
        """
        # Normaliza imagem para 0-255
        img = imagem.copy()
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min()) * 255
        
        # Passo 1: Abertura por reconstrução
        # Remove pequenas manchas brilhantes (ruído) preservando bordas
        # Paper: "limpa a maior parte do ruído na cavidade, mantendo bordas suaves"
        elem_estrut = disk(3)
        erodida = ndimage.grey_erosion(img, footprint=elem_estrut)
        aberta = reconstruction(erodida, img, method='dilation')
        
        # Passo 2: Fechamento por reconstrução
        # Preenche pequenos buracos escuros
        dilatada = ndimage.grey_dilation(aberta, footprint=elem_estrut)
        fechada = reconstruction(dilatada, aberta, method='erosion')
        
        # Passo 3: Transformada h-minima (limiarização baseada em elevação)
        # Implementa o conceito de "elevação" do paper
        # elevacao(x) = altura(x) - altura(minimo(bacia(x)))
        h = self.limiar_elevacao
        semente = np.minimum(fechada + h, 255)
        hmin = reconstruction(semente, fechada, method='erosion')
        
        # Calcula elevação aproximada
        elevacao = fechada - ndimage.grey_erosion(fechada, footprint=disk(7))
        
        # Pontos com baixa elevação são provavelmente ruído na cavidade
        filtrada = fechada.copy()
        baixa_elevacao = elevacao < h
        
        # Em regiões de cavidade (escuras), zera os pixels
        # Este é o passo chave do paper
        limiar_cavidade = np.percentile(fechada[fechada > 0], 30) if np.any(fechada > 0) else 50
        eh_regiao_cavidade = fechada < limiar_cavidade
        filtrada[baixa_elevacao & eh_regiao_cavidade] = 0
        
        return filtrada
    
    # =========================================================================
    # PASSO 2: DETECÇÃO DE BORDAS - LAPLACIANO DA GAUSSIANA (LoG)
    # Seção 3.1 do paper: "Edge detection"
    # =========================================================================
    
    def filtragem_log(self, imagem: np.ndarray):
        """
        Aplica filtragem Laplaciano da Gaussiana para detecção de bordas.
        
        Do paper Seção 3.1:
        "O operador Laplaciano da Gaussiana (LoG) demonstrou ter um perfil
        similar aos campos receptivos do sistema visual humano.
        Isso faz do LoG um operador amplamente usado em detecção de bordas."
        
        Retorna:
            convoluida: Imagem convoluída com LoG
            zero_crossings: Imagem binária dos pontos de borda
        """
        # Aplica filtro LoG
        # Nota: gaussian_laplace do scipy retorna -LoG, então negamos
        convoluida = -gaussian_laplace(imagem, sigma=self.sigma_log)
        
        # Encontra zero-crossings (locais das bordas)
        # Paper: "Zero-crossings detectados da imagem filtrada"
        zero_crossings = self._encontrar_zero_crossings(convoluida)
        
        return convoluida, zero_crossings
    
    def _encontrar_zero_crossings(self, imagem_log: np.ndarray) -> np.ndarray:
        """
        Encontra pontos de zero-crossing na imagem LoG.
        
        Zero-crossings indicam bordas com precisão sub-pixel.
        Um zero-crossing ocorre onde o sinal muda entre vizinhos.
        """
        zc = np.zeros_like(imagem_log, dtype=bool)
        
        # Verifica conectividade-8 para mudanças de sinal
        # Horizontal
        zc[:, :-1] |= (imagem_log[:, :-1] * imagem_log[:, 1:]) < 0
        # Vertical
        zc[:-1, :] |= (imagem_log[:-1, :] * imagem_log[1:, :]) < 0
        # Diagonais
        zc[:-1, :-1] |= (imagem_log[:-1, :-1] * imagem_log[1:, 1:]) < 0
        zc[:-1, 1:] |= (imagem_log[:-1, 1:] * imagem_log[1:, :-1]) < 0
        
        # Remove artefatos de borda
        zc[0, :] = zc[-1, :] = zc[:, 0] = zc[:, -1] = False
        
        return zc
    
    # =========================================================================
    # PASSO 3: SEGMENTAÇÃO WATERSHED
    # Seção 3.2 do paper: "Watershed segmentation"
    # =========================================================================
    
    def segmentacao_watershed(self, filtrada: np.ndarray, convoluida: np.ndarray,
                               ponto_cavidade: tuple):
        """
        Aplica segmentação watershed baseada em marcadores.
        
        Do paper Seção 3.2:
        "Os picos positivos da imagem convoluída, localizados a poucos pixels
        dos zero-crossings, separam a cavidade e as bordas em duas regiões."
        
        Propriedades da entrada (do paper):
        1. Apenas valores positivos usados, negativos zerados
        2. Altos valores de convolução obtidos poucos pixels dos zero-crossings
        3. Região da cavidade tem valor de convolução constante (após redução de ruído)
        """
        # Usa magnitude do gradiente para watershed
        gradiente = sobel(filtrada)
        
        # Cria marcadores
        marcadores = np.zeros(filtrada.shape, dtype=np.int32)
        
        # Marcador 1: Cavidade do VE (do ponto fornecido)
        # Paper: "Um ponto dentro da cavidade é necessário para selecionar o contorno inicial correto"
        
        # Crescimento de região a partir do ponto da cavidade
        mascara_cavidade = self._crescer_regiao_cavidade(filtrada, ponto_cavidade)
        
        # Erode para obter marcador interno (conservador)
        marcador_cavidade = binary_erosion(mascara_cavidade, disk(10))
        if not np.any(marcador_cavidade):
            marcador_cavidade = binary_erosion(mascara_cavidade, disk(5))
        if not np.any(marcador_cavidade):
            marcador_cavidade[ponto_cavidade[0], ponto_cavidade[1]] = True
            
        marcadores[marcador_cavidade] = 1
        
        # Marcador 2: Fundo/Exterior (borda da imagem + regiões brilhantes)
        # Borda
        marcadores[0, :] = 2
        marcadores[-1, :] = 2
        marcadores[:, 0] = 2
        marcadores[:, -1] = 2
        
        # Regiões muito brilhantes (definitivamente não são cavidade)
        if np.any(filtrada > 0):
            limiar_brilho = np.percentile(filtrada[filtrada > 0], 90)
            mascara_brilho = filtrada > limiar_brilho
            mascara_brilho = binary_erosion(mascara_brilho, disk(3))
            marcadores[mascara_brilho] = 2
        
        # Aplica watershed
        rotulos_ws = watershed(gradiente, marcadores)
        
        # Extrai contorno inicial (borda da região da cavidade)
        regiao_cavidade = rotulos_ws == 1
        dilatada = binary_dilation(regiao_cavidade, disk(1))
        contorno_inicial = dilatada & ~regiao_cavidade
        
        return contorno_inicial, rotulos_ws, 1  # rotulo_cavidade = 1
    
    def _crescer_regiao_cavidade(self, imagem: np.ndarray, semente: tuple,
                                  fator_tolerancia: float = 1.5) -> np.ndarray:
        """
        Crescimento de região a partir do ponto semente para segmentar a cavidade.
        
        A cavidade é uma região escura, então crescemos para incluir pixels de intensidade similar.
        """
        h, w = imagem.shape
        valor_semente = imagem[semente[0], semente[1]]
        
        # Calcula estatísticas locais ao redor da semente
        y0, y1 = max(0, semente[0]-30), min(h, semente[0]+30)
        x0, x1 = max(0, semente[1]-30), min(w, semente[1]+30)
        regiao_local = imagem[y0:y1, x0:x1]
        
        # Limiar: pixels similares à semente (região escura)
        media_local = np.mean(regiao_local)
        std_local = np.std(regiao_local)
        
        # Aceita pixels mais escuros que o limiar
        limiar = valor_semente + fator_tolerancia * std_local + 20
        limiar = min(limiar, media_local)
        
        # Máscara inicial: pixels abaixo do limiar
        mascara = imagem < limiar
        
        # Mantém apenas componente conectado contendo a semente
        rotulada, num = label(mascara)
        rotulo_semente = rotulada[semente[0], semente[1]]
        
        if rotulo_semente > 0:
            mascara_cavidade = rotulada == rotulo_semente
        else:
            # Fallback: expande a partir da semente
            mascara_cavidade = np.zeros_like(imagem, dtype=bool)
            mascara_cavidade[semente[0], semente[1]] = True
            for _ in range(50):
                dilatada = binary_dilation(mascara_cavidade, disk(1))
                mascara_cavidade = dilatada & (imagem < limiar)
        
        # Limpeza
        mascara_cavidade = binary_fill_holes(mascara_cavidade)
        mascara_cavidade = binary_opening(mascara_cavidade, disk(3))
        mascara_cavidade = remove_small_objects(mascara_cavidade, min_size=500)
        
        return mascara_cavidade
    
    # =========================================================================
    # PASSO 4: EXTRAÇÃO DE CONTORNO
    # Seção 3.3 do paper: "Contour extraction"
    # =========================================================================
    
    def extrair_contorno(self, mascara_cavidade: np.ndarray, zero_crossings: np.ndarray,
                         raio_busca: int = 5) -> np.ndarray:
        """
        Extrai contorno final usando o processo de três passos do paper.
        
        Do paper Seção 3.3:
        "Para extrair o contorno final, percorremos o contorno inicial três vezes:
        Passo (a): Primeira busca de pontos do contorno na direção normal
        Passo (b): Busca na vizinhança para maximizar número de pontos do contorno
        Passo (c): Interpolação nas bordas faltantes para completar o contorno"
        """
        # Obtém borda da máscara da cavidade
        dilatada = binary_dilation(mascara_cavidade, disk(1))
        borda = dilatada & ~mascara_cavidade
        
        coords_borda = np.argwhere(borda)
        
        if len(coords_borda) == 0:
            return borda
        
        # Passos (a) & (b): Busca zero-crossings perto da borda
        pontos_finais = []
        
        for pt in coords_borda:
            y, x = pt
            melhor_ponto = (y, x)
            encontrado = False
            
            # Busca em raio expandindo por zero-crossing
            for r in range(1, raio_busca + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dy) == r or abs(dx) == r:  # Apenas perímetro
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < zero_crossings.shape[0] and 0 <= nx < zero_crossings.shape[1]:
                                if zero_crossings[ny, nx]:
                                    melhor_ponto = (ny, nx)
                                    encontrado = True
                                    break
                    if encontrado:
                        break
                if encontrado:
                    break
            
            pontos_finais.append(melhor_ponto)
        
        # Cria imagem do contorno
        contorno_final = np.zeros_like(mascara_cavidade, dtype=bool)
        for y, x in pontos_finais:
            contorno_final[y, x] = True
        
        # Passo (c): Conecta pontos (interpolação)
        # Usa operações morfológicas para conectar pontos próximos
        contorno_final = binary_dilation(contorno_final, disk(1))
        contorno_final = skeletonize(contorno_final)
        
        return contorno_final
    
    # =========================================================================
    # PIPELINE PRINCIPAL DE PROCESSAMENTO
    # =========================================================================
    
    def processar_imagem(self, imagem: np.ndarray, ponto_cavidade: tuple) -> dict:
        """
        Processa uma imagem de ecocardiograma através do pipeline completo.
        
        Argumentos:
            imagem: Ecocardiograma de entrada
            ponto_cavidade: Ponto dentro da cavidade do VE (y, x) - entrada semi-automática
        
        Retorna:
            Dicionário com todos os resultados intermediários e finais
        """
        resultados = {'original': imagem, 'ponto_cavidade': ponto_cavidade}
        
        # Passo 1: Filtragem Morfológica (Seção 2.2)
        print("  Passo 1: Filtragem morfológica (redução de ruído)...")
        filtrada = self.filtragem_morfologica(imagem)
        resultados['filtrada'] = filtrada
        
        # Passo 2: Filtragem LoG (Seção 3.1)
        print("  Passo 2: Filtragem LoG (detecção de bordas)...")
        convoluida, zero_crossings = self.filtragem_log(filtrada)
        resultados['convoluida'] = convoluida
        resultados['zero_crossings'] = zero_crossings
        
        # Passo 3: Segmentação Watershed (Seção 3.2)
        print("  Passo 3: Segmentação watershed (contorno inicial)...")
        contorno_inicial, rotulos_ws, rotulo_cavidade = self.segmentacao_watershed(
            filtrada, convoluida, ponto_cavidade
        )
        resultados['contorno_inicial'] = contorno_inicial
        resultados['rotulos_watershed'] = rotulos_ws
        resultados['rotulo_cavidade'] = rotulo_cavidade
        
        # Obtém máscara da cavidade
        mascara_cavidade = rotulos_ws == rotulo_cavidade
        resultados['mascara_cavidade'] = mascara_cavidade
        
        # Passo 4: Extração de Contorno (Seção 3.3)
        print("  Passo 4: Extração de contorno (borda final)...")
        contorno_final = self.extrair_contorno(mascara_cavidade, zero_crossings)
        resultados['contorno_final'] = contorno_final
        
        return resultados
    
    # =========================================================================
    # ANÁLISE DE ACURÁCIA (Seção 4)
    # =========================================================================
    
    def calcular_erro_contorno(self, computado: np.ndarray, gt: np.ndarray) -> dict:
        """
        Calcula erro RMS entre contornos computado e ground truth.
        
        Do paper Seção 4 (Equação 2):
        rms = sqrt(1/N * sum(distancia(Lc_i, Lh_i)))
        """
        coords_comp = np.argwhere(computado)
        coords_gt = np.argwhere(gt)
        
        if len(coords_comp) == 0 or len(coords_gt) == 0:
            return {'erro_rms': np.inf, 'erro_medio': np.inf}
        
        # Distâncias bidirecionais
        dist_c2g = [np.min(np.sqrt(np.sum((coords_gt - p)**2, axis=1))) for p in coords_comp]
        dist_g2c = [np.min(np.sqrt(np.sum((coords_comp - p)**2, axis=1))) for p in coords_gt]
        
        todas_dist = np.array(dist_c2g + dist_g2c)
        
        return {
            'erro_rms': np.sqrt(np.mean(todas_dist**2)),
            'erro_medio': np.mean(todas_dist),
            'erro_std': np.std(todas_dist),
            'erro_max': np.max(todas_dist),
            'hausdorff': max(np.max(dist_c2g), np.max(dist_g2c)),
            'distancias': todas_dist
        }
    
    def calcular_metricas_area(self, mascara_comp: np.ndarray, mascara_gt: np.ndarray) -> dict:
        """Calcula métricas baseadas em área (Seção 5)."""
        area_comp = np.sum(mascara_comp)
        area_gt = np.sum(mascara_gt)
        
        intersecao = np.sum(mascara_comp & mascara_gt)
        uniao = np.sum(mascara_comp | mascara_gt)
        
        dice = 2 * intersecao / (area_comp + area_gt) if (area_comp + area_gt) > 0 else 0
        iou = intersecao / uniao if uniao > 0 else 0
        
        return {
            'area_computada': area_comp,
            'area_gt': area_gt,
            'razao_area': area_comp / area_gt if area_gt > 0 else 0,
            'dice': dice,
            'iou': iou
        }


class ProcessadorCAMUS:
    """Processa imagens do dataset CAMUS."""
    
    def __init__(self, dir_dados: str):
        self.dir_dados = Path(dir_dados)
        self.analisador = AnalisadorMorfologicoEco()
    
    def obter_arquivos(self) -> list:
        """Obtém arquivos de imagem disponíveis."""
        return sorted([f.name for f in self.dir_dados.glob("*.nii.gz") if "_gt" not in f.name])
    
    def processar(self, nome_imagem: str, limiar_elevacao: int = 25, 
                  sigma_log: float = 3.0) -> dict:
        """Processa uma única imagem."""
        # Carrega imagem e ground truth
        caminho_img = self.dir_dados / nome_imagem
        caminho_gt = self.dir_dados / nome_imagem.replace(".nii.gz", "_gt.nii.gz")
        
        imagem = self.analisador.carregar_imagem_nifti(str(caminho_img))
        gt = self.analisador.carregar_ground_truth(str(caminho_gt))
        
        # Configura analisador
        self.analisador.limiar_elevacao = limiar_elevacao
        self.analisador.sigma_log = sigma_log
        
        # Obtém ponto da cavidade do GT (semi-automático - paper requer entrada de ponto)
        cavidade_ve = gt == 1
        coords_cavidade = np.argwhere(cavidade_ve)
        ponto_cavidade = tuple(np.mean(coords_cavidade, axis=0).astype(int))
        
        # Processa
        resultados = self.analisador.processar_imagem(imagem, ponto_cavidade)
        
        # Adiciona dados do GT
        resultados['gt'] = gt
        resultados['gt_ve'] = cavidade_ve
        resultados['gt_borda'] = binary_dilation(cavidade_ve, disk(1)) & ~cavidade_ve
        
        # Calcula métricas
        resultados['metricas_contorno'] = self.analisador.calcular_erro_contorno(
            resultados['contorno_final'], resultados['gt_borda']
        )
        resultados['metricas_area'] = self.analisador.calcular_metricas_area(
            resultados['mascara_cavidade'], resultados['gt_ve']
        )
        
        return resultados


def visualizar_pipeline(resultados: dict, titulo: str = "", caminho_salvar: str = None):
    """Visualiza pipeline completo de processamento."""
    fig, eixos = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    
    # Linha 1: Passos de processamento
    eixos[0, 0].imshow(resultados['original'], cmap='gray')
    eixos[0, 0].plot(resultados['ponto_cavidade'][1], resultados['ponto_cavidade'][0], 'r*', markersize=15)
    eixos[0, 0].set_title('(a) Original + Ponto Cavidade')
    eixos[0, 0].axis('off')
    
    eixos[0, 1].imshow(resultados['filtrada'], cmap='gray')
    eixos[0, 1].set_title('(b) Filtragem Morfológica\n(Redução de Ruído)')
    eixos[0, 1].axis('off')
    
    vmax = np.percentile(np.abs(resultados['convoluida']), 99)
    eixos[0, 2].imshow(resultados['convoluida'], cmap='RdBu', vmin=-vmax, vmax=vmax)
    eixos[0, 2].set_title('(c) Imagem Convoluída LoG')
    eixos[0, 2].axis('off')
    
    eixos[0, 3].imshow(resultados['original'], cmap='gray')
    zc = np.zeros((*resultados['original'].shape, 4))
    zc[resultados['zero_crossings'], :] = [1, 0, 0, 0.8]
    eixos[0, 3].imshow(zc)
    eixos[0, 3].set_title('(d) Zero-crossings (Bordas)')
    eixos[0, 3].axis('off')
    
    # Linha 2: Segmentação e resultados
    eixos[1, 0].imshow(resultados['rotulos_watershed'], cmap='nipy_spectral')
    eixos[1, 0].set_title('(e) Segmentação Watershed')
    eixos[1, 0].axis('off')
    
    eixos[1, 1].imshow(resultados['original'], cmap='gray')
    init = np.zeros((*resultados['original'].shape, 4))
    init[resultados['contorno_inicial'], :] = [0, 1, 0, 0.9]
    eixos[1, 1].imshow(init)
    eixos[1, 1].set_title('(f) Contorno Inicial (Verde)')
    eixos[1, 1].axis('off')
    
    eixos[1, 2].imshow(resultados['original'], cmap='gray')
    final = np.zeros((*resultados['original'].shape, 4))
    final[resultados['contorno_final'], :] = [1, 1, 0, 0.9]
    eixos[1, 2].imshow(final)
    eixos[1, 2].set_title('(g) Contorno Final (Amarelo)')
    eixos[1, 2].axis('off')
    
    # Comparação com GT
    eixos[1, 3].imshow(resultados['original'], cmap='gray')
    gt_overlay = np.zeros((*resultados['original'].shape, 4))
    gt_overlay[resultados['gt_borda'], :] = [0, 1, 0, 0.7]
    eixos[1, 3].imshow(gt_overlay)
    comp_overlay = np.zeros((*resultados['original'].shape, 4))
    comp_overlay[resultados['contorno_final'], :] = [1, 0, 0, 0.7]
    eixos[1, 3].imshow(comp_overlay)
    eixos[1, 3].set_title('(h) GT (Verde) vs Computado (Vermelho)')
    eixos[1, 3].axis('off')
    
    plt.tight_layout()
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
        print(f"  Salvo: {caminho_salvar}")
    return fig


def visualizar_comparacao(resultados: dict, titulo: str = "", caminho_salvar: str = None):
    """Visualização detalhada de comparação."""
    fig, eixos = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    
    # Sobreposição
    eixos[0].imshow(resultados['original'], cmap='gray')
    pts_gt = np.argwhere(resultados['gt_borda'])
    pts_comp = np.argwhere(resultados['contorno_final'])
    eixos[0].scatter(pts_gt[:, 1], pts_gt[:, 0], c='lime', s=1, alpha=0.7, label='GT')
    eixos[0].scatter(pts_comp[:, 1], pts_comp[:, 0], c='red', s=1, alpha=0.9, label='Computado')
    eixos[0].legend(loc='upper right')
    eixos[0].set_title('Sobreposição de Contornos')
    eixos[0].axis('off')
    
    # Comparação de máscaras
    eixos[1].imshow(resultados['original'], cmap='gray', alpha=0.5)
    eixos[1].contour(resultados['gt_ve'], colors='green', linewidths=2)
    eixos[1].contour(resultados['mascara_cavidade'], colors='red', linewidths=2)
    eixos[1].set_title('Cavidade: GT (Verde) vs Computado (Vermelho)')
    eixos[1].axis('off')
    
    # Sobreposição de regiões
    sobreposicao = np.zeros((*resultados['original'].shape, 3))
    sobreposicao[resultados['gt_ve'], 1] = 1
    sobreposicao[resultados['mascara_cavidade'], 0] = 1
    eixos[2].imshow(sobreposicao)
    eixos[2].set_title('Sobreposição (Amarelo=Concordância)')
    eixos[2].axis('off')
    
    # Histograma de erro
    if 'distancias' in resultados['metricas_contorno']:
        dist = resultados['metricas_contorno']['distancias']
        eixos[3].hist(dist, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        rms = resultados['metricas_contorno']['erro_rms']
        eixos[3].axvline(rms, color='red', linestyle='--', linewidth=2, 
                        label=f'RMS={rms:.2f}px')
        eixos[3].set_xlabel('Distância (pixels)')
        eixos[3].set_ylabel('Frequência')
        eixos[3].set_title('Distribuição de Erro')
        eixos[3].legend()
    
    plt.tight_layout()
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
        print(f"  Salvo: {caminho_salvar}")
    return fig


def imprimir_metricas(resultados: dict, nome: str):
    """Imprime métricas."""
    print(f"\n  {'='*50}")
    print(f"  MÉTRICAS: {nome}")
    print(f"  {'='*50}")
    
    m = resultados['metricas_contorno']
    print(f"\n  Erro de Contorno (Eq. 2 do Paper):")
    print(f"    Erro RMS:       {m['erro_rms']:.2f} px")
    print(f"    Erro Médio:     {m['erro_medio']:.2f} px")
    print(f"    Erro Std:       {m['erro_std']:.2f} px")
    print(f"    Hausdorff:      {m['hausdorff']:.2f} px")
    print(f"    Paper reportou: RMS = 2.56 px (σ = 1.21 px)")
    
    m = resultados['metricas_area']
    print(f"\n  Métricas de Área:")
    print(f"    Dice:           {m['dice']:.3f}")
    print(f"    IoU:            {m['iou']:.3f}")
    print(f"    Razão de Área:  {m['razao_area']:.3f}")
    print(f"    Paper reportou: r = 0.99")


def main(dir_dados: str = None):
    """
    Função principal.
    
    Argumentos:
        dir_dados: Caminho para a pasta com os arquivos .nii.gz do CAMUS
                   Se None, usa a pasta atual
    """
    print("="*70)
    print("ANÁLISE MORFOLÓGICA DE BORDAS ENDOCÁRDICAS DO VE")
    print("Implementação de Choy & Jin (1996), SPIE Vol. 2710")
    print("="*70)
    
    # Configuração - usa a pasta especificada ou a pasta atual
    if dir_dados is None:
        dir_dados = Path(".")
    else:
        dir_dados = Path(dir_dados)
    
    # Cria pasta de resultados
    pasta_resultados = Path("resultados")
    pasta_resultados.mkdir(exist_ok=True)
    
    processador = ProcessadorCAMUS(str(dir_dados))
    arquivos = processador.obter_arquivos()
    
    print(f"\nEncontrados {len(arquivos)} imagens: {arquivos}")
    
    # Parâmetros de processamento
    params = {'limiar_elevacao': 25, 'sigma_log': 3.0}
    
    todos_resultados = []
    
    for nome_arq in arquivos:
        print(f"\n{'='*60}")
        print(f"Processando: {nome_arq}")
        print(f"{'='*60}")
        
        try:
            resultados = processador.processar(nome_arq, **params)
            resultados['nome'] = nome_arq
            todos_resultados.append(resultados)
            
            imprimir_metricas(resultados, nome_arq)
            
            base = nome_arq.replace('.nii.gz', '')
            
            fig1 = visualizar_pipeline(resultados, f"Pipeline: {base}",
                                        str(pasta_resultados / f"{base}_pipeline.png"))
            plt.close(fig1)
            
            fig2 = visualizar_comparacao(resultados, f"Comparação: {base}",
                                          str(pasta_resultados / f"{base}_comparacao.png"))
            plt.close(fig2)
            
        except Exception as e:
            print(f"  ERRO: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumo
    print("\n" + "="*70)
    print("ESTATÍSTICAS RESUMIDAS")
    print("="*70)
    
    if todos_resultados:
        rms = [r['metricas_contorno']['erro_rms'] for r in todos_resultados]
        dice = [r['metricas_area']['dice'] for r in todos_resultados]
        
        print(f"\nErro RMS: {np.mean(rms):.2f} ± {np.std(rms):.2f} px")
        print(f"Intervalo: [{np.min(rms):.2f}, {np.max(rms):.2f}] px")
        print(f"\nDice: {np.mean(dice):.3f} ± {np.std(dice):.3f}")
        
        # Histograma cumulativo (como Figura 10 do paper)
        todas_dist = np.concatenate([r['metricas_contorno']['distancias'] for r in todos_resultados])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.arange(0, 15, 1)
        cumulativo = []
        for b in bins:
            cumulativo.append(100 * np.sum(todas_dist < b) / len(todas_dist))
        
        ax.bar(bins[:-1] + 0.5, cumulativo[:-1], width=0.8, color='steelblue', 
               edgecolor='black', alpha=0.7)
        ax.set_xlabel('Faixa de Erro (pixels)', fontsize=12)
        ax.set_ylabel('Percentual Cumulativo (%)', fontsize=12)
        ax.set_title('Histograma de Frequência Cumulativa dos Erros Lineares\n(cf. Figura 10 do paper)', 
                    fontsize=14)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Linha de referência
        ax.axhline(80, color='red', linestyle='--', alpha=0.5)
        ax.text(12, 82, 'Paper: 80% < 3px', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(str(pasta_resultados / "histograma_cumulativo.png"), dpi=150)
        plt.close()
        print(f"\nHistograma cumulativo salvo")
    
    print("\n" + "="*70)
    print("Citação do Dataset CAMUS (Obrigatória):")
    print("-"*70)
    print('S. Leclerc et al., "Deep Learning for Segmentation using an Open')
    print('Large-Scale Dataset in 2D Echocardiography," IEEE TMI, 2019.')
    print("="*70)
    
    print(f"\nResultados salvos na pasta 'resultados/'")
    return todos_resultados


if __name__ == "__main__":
    import sys
    
    # Permite passar o diretório de dados como argumento
    # Uso: python main.py [caminho_para_pasta_com_nii]
    if len(sys.argv) > 1:
        caminho_dados = sys.argv[1]
    else:
        caminho_dados = None  # Usa pasta atual
    
    resultados = main(caminho_dados)
