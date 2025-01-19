import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

def egonet(G):
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        Ni = len(neighbors)
        
        egonet = G.subgraph(neighbors + [node])
        Ei = egonet.number_of_edges()

        Wi = sum(data['weight'] for _, _, data in egonet.edges(data=True))


        adjacency_matrix = nx.adjacency_matrix(egonet, weight='weight').todense()
        eigenvalues = np.linalg.eigvals(adjacency_matrix)
        lambda_w_i = np.max(eigenvalues)
        features = {
            'num_neighbours': Ni,
            'num_edges': Ei,
            'total_weight': Wi,
            'lambda_w': lambda_w_i
        }
        
        nx.set_node_attributes(G, {node: features})
    return G

def compute_anomaly_scores(G):
    log_N = []
    log_E = []
    nodes = []

    for node in G.nodes():
        Ni = G.nodes[node]['num_neighbours']
        Ei = G.nodes[node]['num_edges']
        
        log_N.append(np.log(Ni))
        log_E.append(np.log(Ei))
        nodes.append(node)
    
    X = np.array(log_N).reshape(-1, 1)  
    y = np.array(log_E)  
    
    reg = LinearRegression().fit(X, y)
    theta = reg.coef_[0]
    log_C = reg.intercept_
    C = np.exp(log_C)
    
    anomaly_scores = {}
    for node in G.nodes():
        Ni = G.nodes[node]['num_neighbours']
        Ei = G.nodes[node]['num_edges']
        predicted_Ei = C * (Ni ** theta)  
        score = max(Ei, predicted_Ei) / min(Ei, predicted_Ei) * np.log(np.abs(Ei - predicted_Ei) + 1)
        anomaly_scores[node] = score
    
    nx.set_node_attributes(G, anomaly_scores, name='anomaly_score')
    
    return G

def compute_modified_anomaly_scores(G):
    log_N = []
    log_E = []
    nodes = []

    for node in G.nodes():
        Ni = G.nodes[node]['num_neighbours']
        Ei = G.nodes[node]['num_edges']
        
        log_N.append(np.log(Ni))
        log_E.append(np.log(Ei))
        nodes.append(node)
    
    X = np.array(log_N).reshape(-1, 1)  
    y = np.array(log_E)  
    
    lof_model = LocalOutlierFactor(n_neighbors=20)
    lof_scores = -lof_model.fit_predict(X, y)  
    
    scaler = MinMaxScaler()
    existing_scores = np.array([G.nodes[node]['anomaly_score'] for node in nodes]).reshape(-1, 1)
    lof_scores = lof_scores.reshape(-1, 1)
    
    normalized_existing_scores = scaler.fit_transform(existing_scores)
    normalized_lof_scores = scaler.fit_transform(lof_scores)

    combined_scores = normalized_existing_scores + normalized_lof_scores

    for i, node in enumerate(nodes):
        G.nodes[node]['modified_anomaly_score'] = combined_scores[i][0]
    return G

def draw_top_anomalies(G, attr, filename, top_k = 10):

    scores = {node: G.nodes[node][attr] for node in G.nodes()}
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)

    color_map = ['red' if node in sorted_nodes[:top_k] else 'blue' for node in G.nodes()]

    plt.figure(figsize=(10, 10))
    nx.draw(G, node_color=color_map, node_size=10)
    plt.savefig(filename)
    plt.clf()

def ex1():
    
    # 1
    def load_graph(n_rows = 1500):
        file_path = "ca-AstroPh.txt"
        data = np.loadtxt(file_path).astype(int)
        G = nx.Graph()
        G.add_edges_from(data[: n_rows])
        
        for edge in G.edges():
            if 'weight' not in G[edge[0]][edge[1]]:
                G[edge[0]][edge[1]]['weight'] = 1
            else:
                G[edge[0]][edge[1]]['weight'] += 1
        
        return G
    
    G = load_graph()
    # 2
    G = egonet(G)
    
    # 3
    G = compute_anomaly_scores(G)
    # 4    
    draw_top_anomalies(G, 'anomaly_score', "ex1.4.pdf")
    
    # 5
    G = compute_modified_anomaly_scores(G)
    draw_top_anomalies(G, 'modified_anomaly_score', "ex1.5.pdf")
    
def ex2():
    # 1
    regular_graph = nx.random_regular_graph(3, 100)
    caveman_graph = nx.connected_caveman_graph(10, 20)
    G = nx.union(regular_graph, caveman_graph, rename=("R", "C"))

    np.random.seed(42)
    nodes_R = list(regular_graph.nodes())
    nodes_C = list(caveman_graph.nodes())
    for _ in range(20): 
        node1 = np.random.choice(nodes_R)
        node2 = "C" + str(np.random.choice(nodes_C))
        G.add_edge(node1, node2)
    for edge in G.edges:
        G.add_edge(edge[0], edge[1], weight=1)
    G = egonet(G)
    G = compute_anomaly_scores(G)
    G = compute_modified_anomaly_scores(G)
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=10)
    plt.savefig("ex2.1.graph.pdf")
    plt.clf()
    draw_top_anomalies(G, 'modified_anomaly_score', "ex2.1.anomalies.pdf")
    
    # 2
    graph_3 = nx.random_regular_graph(3, 100) 
    graph_5 = nx.random_regular_graph(5, 100)  
    G = nx.union(graph_3, graph_5, rename=("A", "B"))

    for edge in G.edges():
        G.add_edge(edge[0], edge[1], weight=1)

    np.random.seed(42)
    selected_nodes = np.random.choice(list(G.nodes()), 2)

    for node in selected_nodes:
        for neighbor in G.neighbors(node):
            G[node][neighbor]["weight"] += 10

    G = egonet(G)
    G = compute_anomaly_scores(G)
    G = compute_modified_anomaly_scores(G)
    draw_top_anomalies(G, 'modified_anomaly_score', "ex2.2.anomalies.pdf", 4)

def ex3():
    mat_data = scipy.io.loadmat("ACM.mat")

    attributes = torch.tensor(mat_data["Attributes"].todense(), dtype=torch.float32) 
    adj_matrix = mat_data["Network"] 
    labels = torch.tensor(mat_data["Label"].flatten(), dtype=torch.long)  

    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)

    data = Data(x=attributes, edge_index=edge_index, edge_weight=edge_weight)

    class GCNEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
            super(GCNEncoder, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, latent_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)  
            return x

    class AttributeDecoder(nn.Module):
        def __init__(self, latent_dim=64, hidden_dim=128, output_dim=None):
            super(AttributeDecoder, self).__init__()
            self.conv1 = GCNConv(latent_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, z, edge_index):
            z = self.conv1(z, edge_index)
            z = F.relu(z)
            z = self.conv2(z, edge_index)
            return z 

    class StructureDecoder(nn.Module):
        def __init__(self, latent_dim=64):
            super(StructureDecoder, self).__init__()
            self.conv = GCNConv(latent_dim, latent_dim)  

        def forward(self, z, edge_index):
            z = self.conv(z, edge_index)
            z = F.relu(z)
            adj_reconstructed = torch.sigmoid(torch.matmul(z, z.T))  
            return adj_reconstructed

    class GraphAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
            super(GraphAutoencoder, self).__init__()
            self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
            self.attr_decoder = AttributeDecoder(latent_dim, hidden_dim, input_dim)
            self.struct_decoder = StructureDecoder(latent_dim)

        def forward(self, x, edge_index):
            z = self.encoder(x, edge_index)  
            x_reconstructed = self.attr_decoder(z, edge_index)  
            adj_reconstructed = self.struct_decoder(z, edge_index)  
            return x_reconstructed, adj_reconstructed
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAutoencoder(input_dim=data.x.shape[1]).to(device)

    data = data.to(device)

    def custom_loss(x, x_hat, adj, adj_hat, alpha=0.8):
        attribute_loss = torch.norm(x - x_hat, p='fro') ** 2 
        structure_loss = torch.norm(adj - adj_hat, p='fro') ** 2
        return alpha * attribute_loss + (1 - alpha) * structure_loss

    from sklearn.metrics import roc_auc_score

    optimizer = optim.Adam(model.parameters(), lr=0.004)
    num_epochs = 1

    loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(loader):  
            optimizer.zero_grad()
            
            batch_x = batch.x.to(device)
            batch_edge_index = batch.edge_index.to(device)
            
            x_hat, a_hat = model(batch_x, batch_edge_index)
            
            a = to_scipy_sparse_matrix(batch_edge_index).todense()
            a = torch.from_numpy(a).float().to(device)
            
            loss = custom_loss(batch_x, x_hat, a, a_hat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                x_hat, a_hat = model(data.x.to(device), data.edge_index.to(device))
                attr_errors = torch.mean((x_hat - data.x) ** 2, dim=1)
                struct_errors = torch.mean((a_hat - a) ** 2, dim=1)
                anomaly_scores = attr_errors + struct_errors  
            
            scores = anomaly_scores.cpu().numpy()
            true_labels = labels.cpu().numpy()  
            auc_score = roc_auc_score(true_labels, scores)
            
            print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss:.4f} - ROC AUC: {auc_score:.4f}")




ex1()
ex2()
ex3()
