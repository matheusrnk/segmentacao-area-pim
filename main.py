import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
from skimage.filters import threshold_multiotsu


def animate_matrices_with_titles(matrices, titles, interval=1500):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        matrix = matrices[frame]
        ax.imshow(matrix, cmap='gray', interpolation='none')
        ax.set_title(titles[frame])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(
        matrices), interval=interval, blit=True)

    plt.show()


def save_processed_images(imagepath: str, titles: list, matrices: list):
    imagepath, imagepath_ext = os.path.splitext(imagepath)

    folder_path = 'imagens_processadas/'

    if not os.path.isdir(folder_path):
        try:
            os.makedirs(folder_path)
        except Exception as e:
            print(
                f'Houve um erro ao criar o diretório "{folder_path}". - {e} \n Se foi por motivo de falta de permissão, considere criá-lo manualmente.')
            sys.exit(4)

    for title, img_matrix in zip(titles, matrices):
        img = Image.fromarray(img_matrix)
        img.save(f'imagens_processadas/{imagepath}_{title}{imagepath_ext}')


def find_center_of_mass_object_in_matrix(component_only_matrix: np.ndarray) -> tuple:
    y_indices, x_indices = np.nonzero(component_only_matrix)

    m00 = np.sum(component_only_matrix)

    m10 = np.sum(x_indices)
    m01 = np.sum(y_indices)

    cmx = int(m10 / m00)
    cmy = int(m01 / m00)

    return (cmy, cmx)


def get_connected_component(labeled_matrix: np.ndarray, selected_component: int) -> np.ndarray:
    apply_diff_sel = labeled_matrix != selected_component
    apply_eq_sel = labeled_matrix == selected_component
    new_labeled_matrix = labeled_matrix.copy()
    new_labeled_matrix[apply_diff_sel] = 0
    new_labeled_matrix[apply_eq_sel] = 1
    return new_labeled_matrix


def count_and_find_biggest_in_labeled_matrix(labeled_matrix: np.ndarray) -> int:
    rows, cols = labeled_matrix.shape
    labels_found = {}

    for row in range(rows):
        for col in range(cols):
            if labeled_matrix[row, col] in labels_found:
                labels_found[labeled_matrix[row, col]] += 1
            else:
                labels_found[labeled_matrix[row, col]] = 1

    background_val = 0
    labels_found.pop(background_val)

    labels_found = list(labels_found.items())
    biggest_component_label = max(labels_found, key=lambda x: x[1])[0]

    return biggest_component_label


def deletes_connected_componentes_on_edges(labeled_matrix: np.ndarray) -> np.ndarray:
    top_row = labeled_matrix[0, :]
    bottom_row = labeled_matrix[-1, :]
    left_column = labeled_matrix[1:-1, 0]
    right_column = labeled_matrix[1:-1, -1]

    border_values = np.concatenate(
        [top_row, right_column, bottom_row[::-1], left_column[::-1]], dtype=labeled_matrix.dtype)
    unique_values = np.unique(border_values)

    labeled_matrix_wo_edges = labeled_matrix.copy()

    rows, cols = labeled_matrix_wo_edges.shape
    for row in range(rows):
        for col in range(cols):
            if labeled_matrix_wo_edges[row, col] in unique_values:
                labeled_matrix_wo_edges[row, col] = 0

    return labeled_matrix_wo_edges


def find_connected_components(th_img_matrix: np.ndarray, conn_type: str) -> np.ndarray:
    rows, cols = th_img_matrix.shape

    # adiciona borda de 0's para evitar verificação
    th_img_matrix_bord = np.pad(
        th_img_matrix, pad_width=1, mode='constant', constant_values=0)
    labels_bord = np.zeros((rows + 2, cols + 2), dtype=th_img_matrix.dtype)

    current_label = 1

    for row in range(1, rows):
        for col in range(1, cols):
            if th_img_matrix_bord[row, col] == 1 and labels_bord[row, col] == 0:
                labels_bord[row, col] = current_label
                queue = [(row, col)]
                while queue:
                    current_row, current_col = queue.pop(0)
                    if conn_type == 'c4':
                        neighbors = [(current_row-1, current_col), (current_row, current_col-1),
                                     (current_row, current_col+1), (current_row+1, current_col)]
                    else:
                        neighbors = [(current_row-1, current_col-1), (current_row-1, current_col),
                                     (current_row-1, current_col +
                                      1), (current_row, current_col-1),
                                     (current_row, current_col +
                                      1), (current_row+1, current_col-1),
                                     (current_row+1, current_col), (current_row+1, current_col+1)]
                    for nb_row, nb_col in neighbors:
                        if th_img_matrix_bord[nb_row, nb_col] == 1 and labels_bord[nb_row, nb_col] == 0:
                            labels_bord[nb_row, nb_col] = current_label
                            queue.append((nb_row, nb_col))
                current_label += 1

    # remove a borda de -1 para retornar só os labels
    return labels_bord[1:-1, 1:-1].copy()


def apply_threshold(img_matrix: np.ndarray) -> np.ndarray:
    # se utilizar 3 classes, ele limpa mais. Porém, se perde algumas informações, oq pode ser
    # ruim para imagens diferentes.
    threshold = threshold_multiotsu(img_matrix, classes=2)[0]
    apply_lower_th = img_matrix < threshold
    apply_upper_th = img_matrix > threshold
    new_th_image_matrix = img_matrix.copy()
    new_th_image_matrix[apply_lower_th] = 0
    new_th_image_matrix[apply_upper_th] = 1
    return new_th_image_matrix


def gets_image_matrix(image: Image) -> np.ndarray:
    return np.asarray(image)


def loads_image(imagepath: str) -> Image:
    try:
        image = Image.open(imagepath)
        return image
    except Exception as e:
        print(f'Houve um erro ao carregar a imagem {imagepath} - Erro: {e}')
        sys.exit(3)


def verify_args(args: list) -> str:
    if len(args) <= 0:
        print("Nenhum caminho foi fornecido ao programa.")
        print("Exemplo: python3 main.py <nome_do_arquivo.png>")
        print("Saindo...")
        sys.exit(1)

    if len(args) > 1:
        print("Coloque apenas o caminho do arquivo desejado.")
        print("Exemplo: python3 main.py <nome_do_arquivo.png>")
        print("Saindo...")
        sys.exit(2)

    return args[0]


def main(args):

    images_matrices_archive = []

    imagepath = verify_args(args)

    img = loads_image(args[0])

    img_matrix = gets_image_matrix(img)

    # Guarda a matriz da imagem original
    images_matrices_archive.append(img_matrix)

    # Aplicação do limiar
    th_img_matrix = apply_threshold(img_matrix)

    # Guarda a matriz da imagem limiarizada (ou binarizada)
    th_img_matrix_cp = th_img_matrix.copy()
    th_img_matrix_cp[th_img_matrix == 1] = 255
    images_matrices_archive.append(th_img_matrix_cp)

    # Busca pelos componentes conexos
    labeled_matrix = find_connected_components(th_img_matrix, 'c8')

    # Elimina os componentes conexos na borda
    labeled_matrix_wo_edges = deletes_connected_componentes_on_edges(
        labeled_matrix)

    # Conta os rótulos e encontra o maior componente (eliminando o fundo)
    biggest_component_label = count_and_find_biggest_in_labeled_matrix(
        labeled_matrix_wo_edges)

    # Retorna matriz da imagem (somente com o componente) gerada
    component_only_matrix = get_connected_component(
        labeled_matrix_wo_edges, biggest_component_label)

    # Guarda matriz da imagem somente com o componente
    component_only_matrix_cp = component_only_matrix.copy()
    component_only_matrix_cp[component_only_matrix == 1] = 255
    images_matrices_archive.append(component_only_matrix_cp)

    # Acha o centro de massa do objeto
    crow, ccol = find_center_of_mass_object_in_matrix(component_only_matrix)
    print(f'Center Row = {crow}; Center Column = {ccol}')

    # Identifica o centro do objeto colocando uma celula 0
    component_with_center_matrix = component_only_matrix_cp.copy()
    component_with_center_matrix[crow, ccol] = 0
    images_matrices_archive.append(component_with_center_matrix)

    return imagepath, images_matrices_archive


if __name__ == '__main__':
    imagepath, matrices = main(sys.argv[1:])

    titles = ['Entrada'] + \
        [f'Transição {i}' for i in range(1, len(matrices)-1)] + ['Saída']

    save_processed_images(imagepath, titles, matrices)

    animate_matrices_with_titles(matrices, titles)
