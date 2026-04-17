"""
临界认知AI 最终版（孵化自动进入学习 + 多地址千问探测）
适配 16GB RAM + GTX 1660 (6GB)
IDFT理论完整保留
"""

import os, re, time, json, pickle, random, requests, threading, gc
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ================== 配置 ==================
@dataclass
class Config:
    eta: float = 0.0893105
    d_s: float = 0.8995
    D_f_target: float = 2.45
    
    num_nodes: int = 30000
    branching_factor: int = 2
    depth: int = 10
    
    default_delta: float = 0.12
    gamma: float = 1.0
    dt: float = 0.05
    noise_scale: float = eta
    
    n_levels_by_depth: List[int] = field(default_factory=lambda: [1,1,2,2,3,3,4,4,5,5,6,6])
    
    learning_rate: float = 0.01
    target_avalanche_ratio: float = 1.5
    
    image_resize: int = 32
    perturbation_ratio: float = 0.2
    evolve_steps_per_input: int = 10
    
    qwen_model: str = "qwen2.5:1.8b"
    
    download_dir: str = './internet_images'
    knowledge_dir: str = './knowledge'
    checkpoint_dir: str = './checkpoints'
    
    def __post_init__(self):
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

CONFIG = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}, η = {CONFIG.eta:.6f}")

# ================== 分形树拓扑（CSR优化）==================
class FractalTreeTopology:
    def __init__(self, num_nodes, branching_factor, depth, spatial_scale=1.0):
        self.num_nodes = num_nodes
        self.branching_factor = branching_factor
        self.depth = depth
        self.spatial_scale = spatial_scale
        self.coords = self._generate_tree_coords()
        self.num_nodes = len(self.coords)
        self.adjacency = self._build_adjacency()
        self.D_f = self._estimate_fractal_dimension()
        print(f"分形树: {self.num_nodes}节点, D_f ≈ {self.D_f:.2f}")
        
    def _generate_tree_coords(self):
        coords = [np.array([0.0, 0.0, 0.0])]
        parent_ids = [0]
        for d in range(self.depth):
            new_coords, new_parents = [], []
            for pid in parent_ids:
                num_children = np.random.poisson(self.branching_factor)
                for _ in range(num_children):
                    if len(coords) + len(new_coords) >= self.num_nodes:
                        break
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    scale = self.spatial_scale * (0.8 ** d)
                    child = coords[pid] + direction * scale * np.random.uniform(0.5, 1.5)
                    new_coords.append(child)
                    new_parents.append(len(coords) + len(new_coords) - 1)
                if len(coords) + len(new_coords) >= self.num_nodes:
                    break
            coords.extend(new_coords)
            parent_ids = new_parents
            if len(coords) >= self.num_nodes:
                break
        return np.array(coords[:self.num_nodes])
    
    def _build_adjacency(self):
        coords_t = torch.tensor(self.coords, dtype=torch.float32, device=device)
        num_nodes = self.num_nodes
        avg_radius = self.spatial_scale * (0.6 ** (self.depth / 2))
        chunk_size = 2000
        all_crow = [0]
        all_col = []
        for i in tqdm(range(0, num_nodes, chunk_size), desc="构建邻接"):
            i_end = min(i + chunk_size, num_nodes)
            coords_i = coords_t[i:i_end]
            for j in range(0, num_nodes, chunk_size):
                j_end = min(j + chunk_size, num_nodes)
                coords_j = coords_t[j:j_end]
                diff = coords_i.unsqueeze(1) - coords_j.unsqueeze(0)
                dist = torch.norm(diff, dim=2)
                mask = (dist > 0) & (dist < avg_radius)
                for local_i in range(i_end - i):
                    global_i = i + local_i
                    nb = j + torch.where(mask[local_i])[0]
                    nb = nb[nb != global_i]
                    all_col.append(nb.cpu())
                    all_crow.append(all_crow[-1] + len(nb))
            torch.cuda.empty_cache()
        col = torch.cat(all_col).to(device)
        crow = torch.tensor(all_crow, dtype=torch.int32, device=device)
        vals = torch.ones(len(col), dtype=torch.float32, device=device)
        return torch.sparse_csr_tensor(crow, col, vals, size=(num_nodes, num_nodes))

    def _estimate_fractal_dimension(self):
        coords = self.coords
        scales = np.logspace(-2, 0, 10)
        counts = [len(np.unique(np.round(coords / s).astype(int), axis=0)) for s in scales]
        slope, _ = np.polyfit(np.log(scales), np.log(counts), 1)
        return -slope

# ================== 临界变量（动态n值）==================
class CriticalVariablesGPU:
    def __init__(self, num_nodes, n_levels, delta=0.12):
        self.num_nodes = num_nodes
        self.n_levels = torch.tensor(n_levels, dtype=torch.int32, device=device)
        self.num_states = 2 ** self.n_levels
        self.delta = torch.full((num_nodes,), delta, dtype=torch.float32, device=device)
        self.state = torch.randint(0, 2, (num_nodes,), dtype=torch.int32, device=device)
        self.phase = torch.rand(num_nodes, device=device) * 2 * np.pi
        self._clamp_state()
        self.noise_pool = torch.randn(20_000_000, device=device)
        self.noise_ptr = 0

    def _clamp_state(self):
        self.state = torch.clamp(self.state, torch.zeros_like(self.num_states), self.num_states - 1)

    def get_value(self):
        denom = torch.where(self.num_states > 1, self.num_states - 1, torch.ones_like(self.num_states))
        return 2.0 * self.state.float() / denom - 1.0

    def _get_noise(self, size):
        if self.noise_ptr + size > len(self.noise_pool):
            self.noise_ptr = 0
        noise = self.noise_pool[self.noise_ptr:self.noise_ptr+size]
        self.noise_ptr += size
        return noise

    def update(self, coupling, noise_scale, dt, gamma=1.0):
        x = self.get_value()
        restore = 2.0 * self.delta * x * (1.0 - x**2)
        d_phase = coupling * dt + noise_scale * self._get_noise(self.num_nodes) * np.sqrt(dt)
        d_phase += gamma * restore * dt
        self.phase = (self.phase + d_phase) % (2 * np.pi)
        new_state = (self.phase / (2 * np.pi) * self.num_states).long()
        self.state = torch.clamp(new_state, torch.zeros_like(self.num_states), self.num_states - 1)
        return self.state

    def get_x(self):
        return self.get_value()

# ================== GPU临界网络 ==================
class CriticalNetworkGPU:
    def __init__(self):
        print("生成分形拓扑...")
        self.topo = FractalTreeTopology(CONFIG.num_nodes, CONFIG.branching_factor, CONFIG.depth)
        self.num_nodes = self.topo.num_nodes
        n_levels = self._assign_n_levels()
        print("初始化临界变量...")
        self.vars = CriticalVariablesGPU(self.num_nodes, n_levels, CONFIG.default_delta)
        self.activity = torch.zeros(self.num_nodes, device=device)
        self.activity_decay = 0.99
        self.equivalent_ops = 0
        print("初始化耦合矩阵...")
        self.J = self._init_coupling()
        self.x = self.vars.get_x()
        self.avalanche_sizes = deque(maxlen=500)
        self.total_avalanches = 0
        self.phase = "incubation"

    def _assign_n_levels(self):
        coords = self.topo.coords
        dist = np.linalg.norm(coords, axis=1)
        max_dist = dist.max()
        norm_dist = dist / (max_dist + 1e-6)
        n_bins = len(CONFIG.n_levels_by_depth)
        bin_idx = (norm_dist * (n_bins - 1)).astype(int)
        return np.array(CONFIG.n_levels_by_depth)[bin_idx]

    def _init_coupling(self):
        adj = self.topo.adjacency
        crow = adj.crow_indices()
        col = adj.col_indices()
        new_crow = [0]
        new_col, new_vals = [], []
        for i in tqdm(range(self.num_nodes), desc="初始化耦合", leave=False):
            start, end = crow[i].item(), crow[i+1].item()
            for idx in range(start, end):
                j = col[idx].item()
                if j > i:
                    n_i = self.vars.n_levels[i].float()
                    n_j = self.vars.n_levels[j].float()
                    delta_n = torch.abs(n_i - n_j)
                    decay = 2.0 ** (-delta_n / CONFIG.d_s)
                    sign = 1.0 if torch.rand(1).item() < 0.8 else -1.0
                    base = CONFIG.eta * 0.5
                    w = sign * base * decay.item() * (0.5 + 0.5 * torch.rand(1).item())
                    new_col.extend([j, i])
                    new_vals.extend([w, w])
            new_crow.append(len(new_col))
        crow_t = torch.tensor(new_crow, dtype=torch.int32, device=device)
        col_t = torch.tensor(new_col, dtype=torch.int32, device=device)
        vals_t = torch.tensor(new_vals, dtype=torch.float32, device=device)
        return torch.sparse_csr_tensor(crow_t, col_t, vals_t, size=(self.num_nodes, self.num_nodes))

    def energy(self):
        x = self.vars.get_x()
        E_delta = torch.sum(self.vars.delta * (1 - x**2))
        E_couple = -0.5 * x @ (torch.sparse.mm(self.J, x.unsqueeze(1)).squeeze())
        return (E_delta + E_couple).item()

    def _update_activity(self, flips_mask):
        self.activity = self.activity * self.activity_decay
        self.activity[flips_mask] += 1.0

    def _adjust_n_levels(self):
        high = self.activity > 2.0
        low = self.activity < 0.3
        new_n = self.vars.n_levels.clone()
        new_n[high] = torch.clamp(new_n[high] + 1, 1, 6)
        new_n[low] = torch.clamp(new_n[low] - 1, 1, 6)
        if (new_n != self.vars.n_levels).any():
            self.vars.n_levels = new_n
            self.vars.num_states = 2 ** new_n
            self.vars._clamp_state()

    def apply_perturbation(self, pattern, target_indices):
        pattern_t = torch.tensor(pattern, dtype=torch.float32, device=device)
        for i, idx in enumerate(target_indices):
            if i >= len(pattern_t): break
            val = pattern_t[i]
            max_s = self.vars.num_states[idx].item()
            if max_s > 1:
                state_idx = int((val + 1) / 2 * (max_s - 1))
                state_idx = max(0, min(max_s-1, state_idx))
                self.vars.state[idx] = state_idx
                self.vars.phase[idx] = state_idx / max_s * 2 * np.pi
        self.x = self.vars.get_x()

    def evolve(self, steps=1):
        avalanche = 0
        for _ in range(steps):
            old_state = self.vars.state.clone()
            coupling = torch.sparse.mm(self.J, self.vars.get_x().unsqueeze(1)).squeeze()
            new_state = self.vars.update(coupling, CONFIG.noise_scale, CONFIG.dt, CONFIG.gamma)
            self.x = self.vars.get_x()
            flips = (new_state != old_state)
            avalanche += flips.sum().item()
            self._update_activity(flips)
        self.avalanche_sizes.append(avalanche)
        self.total_avalanches += avalanche
        if len(self.avalanche_sizes) % 10 == 0:
            self._adjust_n_levels()
        self.equivalent_ops += (2 ** self.vars.n_levels.float().mean().item()) * steps
        return avalanche

    def hebbian_learn(self):
        lr = CONFIG.learning_rate
        crow = self.J.crow_indices()
        col = self.J.col_indices()
        vals = self.J.values()
        new_vals = vals.clone()
        for i in range(self.num_nodes):
            start, end = crow[i].item(), crow[i+1].item()
            for idx in range(start, end):
                j = col[idx].item()
                xi, xj = self.x[i], self.x[j]
                hebb = xi * xj - 0.05 * (xi**2 + xj**2)
                new_vals[idx] += lr * hebb
        new_vals = torch.clamp(new_vals, -1.0, 1.0)
        self.J = torch.sparse_csr_tensor(crow, col, new_vals, self.J.shape)

    def adjust_criticality(self):
        if len(self.avalanche_sizes) < 50: return
        sizes = np.array(self.avalanche_sizes)
        ratio = np.var(sizes) / (np.mean(sizes) + 1e-6)
        if ratio > CONFIG.target_avalanche_ratio:
            self.vars.delta *= 1.01
        elif ratio < 0.8:
            self.vars.delta *= 0.99
        self.vars.delta = torch.clamp(self.vars.delta, 0.02, 0.5)

    def check_criticality(self):
        if len(self.avalanche_sizes) < 100: return False
        sizes = np.array(self.avalanche_sizes)
        ratio = np.var(sizes) / (np.mean(sizes) + 1e-6)
        return abs(ratio - CONFIG.target_avalanche_ratio) < 0.3

# ================== 原始输入处理 ==================
class RawInputProcessor:
    @staticmethod
    def image_to_pattern(image_path, target_length):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((CONFIG.image_resize, CONFIG.image_resize))
            pixels = np.array(img).flatten() / 127.5 - 1.0
            if len(pixels) < target_length:
                pixels = np.tile(pixels, target_length // len(pixels) + 1)
            return pixels[:target_length]
        except:
            return np.random.uniform(-1, 1, target_length)

    @staticmethod
    def text_to_pattern(text, target_length):
        if not text:
            return np.random.uniform(-1, 1, target_length)
        data = text.encode('utf-8')
        pattern = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        if len(pattern) < target_length:
            pattern = np.tile(pattern, target_length // len(pattern) + 1)
        return pattern[:target_length]

# ================== 内生知识库 ==================
class EndogenousKnowledgeBase:
    def __init__(self):
        self.items = []

    def add(self, attractor_state, metadata):
        self.items.append({
            'attractor_state': attractor_state.cpu().numpy().astype(np.int8),
            'metadata': metadata
        })

    def query(self, query_state, top_k=3):
        if not self.items: return []
        query_np = query_state.cpu().numpy().astype(np.int8)
        sims = []
        for it in self.items:
            match = (it['attractor_state'] == query_np).sum()
            sims.append(match / len(query_np))
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [(self.items[i], sims[i]) for i in top_idx]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.items, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.items = pickle.load(f)

# ================== 上网代理（多地址自动探测）==================
class QwenWebAgent:
    def __init__(self):
        # ✅ 多地址列表，按优先级排序
        self.potential_hosts = [
            "http://127.0.0.1:5000",
            "http://192.168.10.8:5000",
            "http://localhost:11434",
        ]
        self.active_host = None
        self.model = CONFIG.qwen_model

    def _find_active_host(self):
        for host in self.potential_hosts:
            try:
                resp = requests.get(f"{host}/api/tags", timeout=2)
                if resp.status_code == 200:
                    print(f"✅ 成功连接到 Ollama 服务: {host}")
                    return host
            except:
                continue
        print("⚠️ 警告：无法连接到任何 Ollama 服务，将使用离线模式")
        return None

    def is_available(self):
        if self.active_host is None:
            self.active_host = self._find_active_host()
        return self.active_host is not None

    def generate_topic(self):
        return self._call("你是一个好奇的AI。生成一个想了解的话题。只给出名称。")

    def extract_keywords(self, text):
        return self._call(f"提取2-3个搜索关键词，空格分隔。\n{text[:300]}")

    def _call(self, prompt):
        if not self.is_available():
            return ""
        try:
            resp = requests.post(
                f"{self.active_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 100}
                },
                timeout=30
            )
            return resp.json().get("response", "").strip()
        except:
            return ""

# ================== 爬虫 ==================
class SimpleCrawler:
    @staticmethod
    def download_images(keyword, max_num=2):
        try:
            from icrawler.builtin import GoogleImageCrawler
            safe_keyword = re.sub(r'[^\w\-_]', '_', keyword)
            temp_dir = os.path.join(CONFIG.download_dir, safe_keyword)
            os.makedirs(temp_dir, exist_ok=True)
            crawler = GoogleImageCrawler(storage={'root_dir': temp_dir})
            crawler.crawl(keyword=keyword, max_num=max_num)
            return [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]
        except:
            return []

    @staticmethod
    def fetch_text_snippet(topic):
        try:
            import wikipedia
            return wikipedia.summary(topic, sentences=2)
        except:
            return ""

# ================== 主认知系统 ==================
class EndogenousCognitiveAI:
    def __init__(self):
        self.net = CriticalNetworkGPU()
        self.kb = EndogenousKnowledgeBase()
        self.qwen = QwenWebAgent()
        self.processor = RawInputProcessor()
        self.crawler = SimpleCrawler()
        self.step_count = 0
        self.phase = "incubation"

    def incubate(self):
        print("🔥 临界孕育...")
        while not self.net.check_criticality():
            self.net.evolve(steps=5)
            self.net.hebbian_learn()
            self.step_count += 1
            if self.step_count % 200 == 0:
                self.net.adjust_criticality()
                print(f"Step {self.step_count}, 能量: {self.net.energy():.2f}, 等效ops: {self.net.equivalent_ops:.2e}")
                gc.collect(); torch.cuda.empty_cache()
        print(f"✅ 临界态达成！")
        self.phase = "online"
        self.online_learn()

    def learn_from_input(self, input_pattern, metadata):
        num_perturb = int(self.net.num_nodes * CONFIG.perturbation_ratio)
        target_indices = np.random.choice(self.net.num_nodes, num_perturb, replace=False)
        self.net.apply_perturbation(input_pattern, target_indices)
        self.net.evolve(steps=CONFIG.evolve_steps_per_input)
        self.net.hebbian_learn()
        attractor = self.net.vars.state.clone()
        self.kb.add(attractor, metadata)
        return attractor

    def query(self, input_pattern):
        num_perturb = int(self.net.num_nodes * CONFIG.perturbation_ratio)
        target_indices = np.random.choice(self.net.num_nodes, num_perturb, replace=False)
        self.net.apply_perturbation(input_pattern, target_indices)
        self.net.evolve(steps=CONFIG.evolve_steps_per_input)
        return self.kb.query(self.net.vars.state, top_k=3)

    def online_learn(self):
        print("🌍 自主上网学习...")
        use_web = self.qwen.is_available()
        if not use_web:
            print("⚠️ 千问离线，使用随机话题")
        while True:
            topic = self.qwen.generate_topic() if use_web else random.choice(["猫","狗","汽车","飞机","海洋"])
            print(f"\n📚 主题: {topic}")
            text = self.crawler.fetch_text_snippet(topic)
            if text:
                pat = self.processor.text_to_pattern(text, 1024)
                self.learn_from_input(pat, {'type':'text','topic':topic,'content':text[:80]})
                print("  📖 文本学习完成")
            imgs = self.crawler.download_images(topic, max_num=2)
            for img in imgs:
                pat = self.processor.image_to_pattern(img, 1024)
                self.learn_from_input(pat, {'type':'image','topic':topic,'path':img})
                os.remove(img)
                print(f"  🖼️ 图像学习完成")
            if self.step_count % 5 == 0:
                self._save_checkpoint(f"step_{self.step_count}")
                gc.collect(); torch.cuda.empty_cache()
            self.step_count += 1
            time.sleep(1)

    def _save_checkpoint(self, tag):
        base = os.path.join(CONFIG.checkpoint_dir, tag)
        torch.save(self.net.vars.state.cpu(), base + "_state.pt")
        torch.save(self.net.J.cpu(), base + "_J.pt")
        self.kb.save(base + "_kb.pkl")
        with open(base + "_meta.pkl", 'wb') as f:
            pickle.dump({'step': self.step_count, 'phase': self.phase, 'ops': self.net.equivalent_ops}, f)
        print(f"💾 已保存检查点: {tag}")

    def run(self):
        print("="*60)
        print("🧠 临界认知AI 最终版 | 30k节点 | 多地址千问探测")
        print("="*60)
        if self.phase == "incubation":
            self.incubate()

if __name__ == "__main__":
    ai = EndogenousCognitiveAI()
    ai.run()
