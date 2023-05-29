import networkx as nx
import gzip
import csv


class GraphHandler(object):
    def __init__(self, triplets: list) -> None:
        self.graph = nx.DiGraph()

        for relation, head, tail in triplets:
            self.graph.add_edge(head, tail)


    def add_edge(self, head: str, tail: str):
        self.graph.add(head, tail)


    def get_num_nodes(self) -> int:
        return self.graph.number_of_nodes()


    def get_num_edges(self) -> int:
        return self.graph.number_of_edges()


    def get_max_degree(self) -> int:
        return max(self.graph.degree(), key=lambda x: x[1])[1]


    def get_avg_degree(self) -> int:
        return sum(dict(self.graph.degree()).values()) / len(self.graph)


    def get_longest_path_length(self) -> int:
        return nx.dag_longest_path(self.graph)


if __name__ == "__main__":
    lang = "ja"
    dataset_dir = f"datasets/conceptnet-assertions-5.7.0/{lang}"
    input_file = f"conceptnet-assertions-5.7.0_{lang}.csv.gz"
    conceptnet_path = f"{dataset_dir}/{input_file}"

    conceptnet = []
    with gzip.open(conceptnet_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            conceptnet.append(row[1:-1])  # 0番目と4番目をメモリ節約のため排除

    graph_handler = GraphHandler(conceptnet)

    print(f"Num of vertices: {graph_handler.get_num_nodes()}")
    print(f"Num of edges: {graph_handler.get_num_edges()}")
    print(f"Max degree: {graph_handler.get_max_degree()}")
    print(f"Average degree: {graph_handler.get_avg_degree()}")
    # print(f"Longest path length: {graph_handler.get_longest_path_length()}")

