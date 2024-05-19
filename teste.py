import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def generate_random_colors(num_colors):
    np.random.seed(0)
    colors = np.random.rand(num_colors, 3)
    return colors

def show_connected_components(binary_image):
    # Rotula os componentes conectados
    labeled_image, num_features = label(binary_image)
    
    # Gera uma cor única para cada componente
    colors = generate_random_colors(num_features + 1)  # +1 para incluir o fundo
    
    # Cria uma imagem colorida baseada nos componentes conectados
    colored_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=float)
    for label_num in range(1, num_features + 1):
        colored_image[labeled_image == label_num] = colors[label_num]
    
    # Plota a imagem colorida
    plt.imshow(colored_image, interpolation='none')
    plt.title(f'Connected Components (Total: {num_features})')
    plt.axis('off')
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de imagem binarizada
    binary_image = np.array([[0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1],
                             [1, 1, 0, 0, 1, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1]])
    
    show_connected_components(binary_image)
