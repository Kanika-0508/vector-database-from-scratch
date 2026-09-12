import time
import numpy as np
import matplotlib.pyplot as plt

class ExactVectorDB:
    """Exact Brute-Force Index using Cosine Similarity."""
    def __init__(self, dimension: int):
        self.dim = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.empty(0, dtype=np.int64)
        self.deleted_mask = np.empty(0, dtype=bool)

    def insert(self, ids: np.ndarray, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vecs = vectors / norms

        self.vectors = np.vstack([self.vectors, norm_vecs])
        self.ids = np.concatenate([self.ids, ids])
        self.deleted_mask = np.concatenate([self.deleted_mask, np.zeros(len(ids), dtype=bool)])

    def delete(self, doc_id: int) -> bool:
        matches = np.where(self.ids == doc_id)[0]
        if len(matches) == 0 or self.deleted_mask[matches[0]]:
            return False
        self.deleted_mask[matches[0]] = True
        return True

    def query(self, query_vec: np.ndarray, k: int = 5):
        norm = np.linalg.norm(query_vec)
        q_norm = query_vec / (norm if norm > 0 else 1.0)
        
        sims = np.dot(self.vectors, q_norm)
        sims[self.deleted_mask] = -np.inf

        top_k_idx = np.argsort(-sims)[:k]
        return self.ids[top_k_idx], sims[top_k_idx]


class IVFVectorDB:
    """Approximate Inverted File Index (IVF-Flat) built in pure NumPy."""
    def __init__(self, dimension: int, n_clusters: int = 128):
        self.dim = dimension
        self.n_clusters = n_clusters
        self.centroids = None
        self.inverted_index = {i: [] for i in range(n_clusters)}
        self.id_to_vec = {}
        self.deleted_ids = set()

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms

    def train_and_insert(self, ids: np.ndarray, vectors: np.ndarray, max_iter: int = 15):
        vectors = self._normalize(vectors)
        N = len(vectors)
        
        # 1. Initialize centroids randomly
        rng = np.random.default_rng(42)
        initial_indices = rng.choice(N, size=self.n_clusters, replace=False)
        self.centroids = vectors[initial_indices].copy()

        # 2. Simple K-Means clustering
        for _ in range(max_iter):
            sims = np.dot(vectors, self.centroids.T)
            labels = np.argmax(sims, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for c in range(self.n_clusters):
                cluster_members = vectors[labels == c]
                if len(cluster_members) > 0:
                    new_centroids[c] = cluster_members.mean(axis=0)
                else:
                    new_centroids[c] = vectors[rng.integers(0, N)]
            self.centroids = self._normalize(new_centroids)

        # 3. Populate inverted lists
        sims = np.dot(vectors, self.centroids.T)
        labels = np.argmax(sims, axis=1)

        for doc_id, vec, cluster_id in zip(ids, vectors, labels):
            self.inverted_index[cluster_id].append(doc_id)
            self.id_to_vec[doc_id] = vec

        for c in range(self.n_clusters):
            self.inverted_index[c] = np.array(self.inverted_index[c], dtype=np.int64)

    def delete(self, doc_id: int) -> bool:
        if doc_id in self.id_to_vec and doc_id not in self.deleted_ids:
            self.deleted_ids.add(doc_id)
            return True
        return False

    def query(self, query_vec: np.ndarray, k: int = 5, nprobe: int = 4):
        """The 'knob' is nprobe: number of nearest centroids to probe."""
        q_norm = self._normalize(query_vec)

        # Find closest centroids
        centroid_sims = np.dot(self.centroids, q_norm)
        probed_clusters = np.argsort(-centroid_sims)[:nprobe]

        candidate_ids = []
        for c in probed_clusters:
            candidate_ids.extend(self.inverted_index[c])

        if not candidate_ids:
            return np.array([]), np.array([])

        candidate_ids = np.array(candidate_ids, dtype=np.int64)
        
        # Filter deleted vectors via tombstone
        valid_mask = ~np.isin(candidate_ids, list(self.deleted_ids))
        candidate_ids = candidate_ids[valid_mask]

        if len(candidate_ids) == 0:
            return np.array([]), np.array([])

        cand_vectors = np.array([self.id_to_vec[i] for i in candidate_ids])
        sims = np.dot(cand_vectors, q_norm)

        top_indices = np.argsort(-sims)[:k]
        return candidate_ids[top_indices], sims[top_indices]


def run_benchmark():
    DIM = 64
    N_VECTORS = 50_000
    N_QUERIES = 500
    K = 10
    
    print(f"[*] Generating {N_VECTORS} vectors (dim={DIM})...")
    rng = np.random.default_rng(1337)
    raw_data = rng.normal(size=(N_VECTORS, DIM)).astype(np.float32)
    doc_ids = np.arange(N_VECTORS, dtype=np.int64)

    print("[*] Building Exact Index...")
    exact_db = ExactVectorDB(DIM)
    exact_db.insert(doc_ids, raw_data)

    print("[*] Building IVF Index (128 clusters)...")
    ivf_db = IVFVectorDB(dimension=DIM, n_clusters=128)
    ivf_db.train_and_insert(doc_ids, raw_data)

    # Test deletion
    print("[*] Testing tombstone deletion on doc_id=42...")
    exact_db.delete(42)
    ivf_db.delete(42)

    query_vectors = rng.normal(size=(N_QUERIES, DIM)).astype(np.float32)

    # Compute Ground Truth
    print(f"[*] Computing Ground Truth with Exact DB across {N_QUERIES} queries...")
    t0 = time.perf_counter()
    ground_truth = []
    for q in query_vectors:
        res_ids, _ = exact_db.query(q, k=K)
        ground_truth.append(set(res_ids))
    exact_total_time = time.perf_counter() - t0
    exact_qps = N_QUERIES / exact_total_time
    print(f"    Exact Search QPS: {exact_qps:.2f} queries/sec")

    # Evaluate IVF across different values of nprobe (The Knob)
    nprobe_values = [1, 2, 4, 8, 16, 32, 64]
    recalls = []
    qps_list = []

    print("\n--- Benchmarking Approximate Index ---")
    for np_val in nprobe_values:
        t0 = time.perf_counter()
        hits = 0
        total_elements = N_QUERIES * K

        for i, q in enumerate(query_vectors):
            res_ids, _ = ivf_db.query(q, k=K, nprobe=np_val)
            hits += len(set(res_ids).intersection(ground_truth[i]))

        duration = time.perf_counter() - t0
        qps = N_QUERIES / duration
        recall = hits / total_elements

        recalls.append(recall)
        qps_list.append(qps)
        print(f"nprobe: {np_val:2d} | Recall@{K}: {recall * 100:.2f}% | QPS: {qps:7.2f}")

    # Generate Plot
    plt.figure(figsize=(8, 5))
    plt.plot(qps_list, [r * 100 for r in recalls], marker='o', color='crimson', linewidth=2)
    for i, txt in enumerate(nprobe_values):
        plt.annotate(f"nprobe={txt}", (qps_list[i], recalls[i] * 100), textcoords="offset points", xytext=(0, 8), ha='center')

    plt.title("Speed vs Accuracy Trade-off (IVF-Flat with NumPy)")
    plt.xlabel("Queries Per Second (QPS)")
    plt.ylabel("Recall@10 (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("accuracy_vs_speed.png", dpi=300)
    print("\n[+] Benchmark graph saved as 'accuracy_vs_speed.png'")

if __name__ == "__main__":
    run_benchmark()