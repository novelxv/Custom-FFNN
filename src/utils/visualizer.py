import matplotlib.pyplot as plt
import networkx as nx

def plot_weight_distribution(model, layer_indices=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))
    
    for idx in layer_indices:
        weights = model.layers[idx].weights.flatten()
        plt.hist(weights, bins=50, alpha=0.7)
        plt.title(f"Distribusi Bobot Layer {idx}")
        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi")
        plt.grid(True)
        plt.show()

def plot_gradient_distribution(model, layer_indices=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))

    for idx in layer_indices:
        grads = model.layers[idx].grad_weights.flatten()
        plt.hist(grads, bins=50, alpha=0.7, color='orange')
        plt.title(f"Distribusi Gradien Bobot Layer {idx}")
        plt.xlabel("Nilai Gradien")
        plt.ylabel("Frekuensi")
        plt.grid(True)
        plt.show()

def visualize_network_structure(model, max_neurons_per_layer=6):
    G = nx.DiGraph()

    def neuron_id(layer_idx, neuron_idx):
        return f"L{layer_idx}_N{neuron_idx}"

    for l_idx, layer in enumerate(model.layers):
        input_size = layer.weights.shape[1]
        output_size = layer.weights.shape[0]

        input_neurons = range(min(input_size, max_neurons_per_layer))
        output_neurons = range(min(output_size, max_neurons_per_layer))

        for i in input_neurons:
            G.add_node(neuron_id(l_idx, i), layer=l_idx)

        for j in output_neurons:
            G.add_node(neuron_id(l_idx + 1, j), layer=l_idx + 1)

            for i in input_neurons:
                try:
                    weight = layer.weights[j, i]
                    grad = layer.grad_weights[j, i]
                    label = f"w={weight:.2f}\ng={grad:.2f}"
                except:
                    label = "?"

                G.add_edge(
                    neuron_id(l_idx, i),
                    neuron_id(l_idx + 1, j),
                    label=label
                )

    pos = nx.multipartite_layout(G, subset_key="layer")
    edge_labels = nx.get_edge_attributes(G, "label")

    plt.figure(figsize=(12, 7))
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=8, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Struktur Jaringan FFNN: Bobot & Gradien Bobot")
    plt.tight_layout()
    plt.show()