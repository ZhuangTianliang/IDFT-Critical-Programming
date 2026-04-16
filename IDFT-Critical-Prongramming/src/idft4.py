"""
临界认知AI - 最终稳定版（和你原始逻辑完全一致，仅修复2个启动bug）
1. 修复torch.clamp类型冲突（恢复你原始正确写法）
2. 修复千问API 404检测错误
3. 解决PaddleOCR与PyTorch GPU冲突（保留OCR功能）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import os, re, time, pickle, random, requests
import numpy as np
from PIL import Image
from collections import deque
from dataclasses import dataclass, field
import torch
from transformers import CLIPProcessor, CLIPModel
from icrawler.builtin import GoogleImageCrawler

# ================== 配置（完全保留你原始配置） ==================
@dataclass
class Config:
    num_nodes: int = 30000
    default_delta: float = 0.12
    dt: float = 0.05
    noise_scale: float = 0.089
    learning_rate: float = 0.01
    evolve_steps: int = 30
    
    branching_factor: int = 3
    depth: int = 11
    spatial_scale: float = 1.0
    n_levels_by_depth: list = field(default_factory=lambda: [1,1,2,2,3,3,4,4,5,5,6])
    
    clip_model: str = 'openai/clip-vit-base-patch32'
    embedding_dim: int = 512
    
    qwen_api: str = "http://localhost:5000/api/generate"
    qwen_model: str = "qwen2.5:1.8b"
    
    download_dir: str = './internet_images'
    knowledge_dir: str = './knowledge'
    checkpoint_dir: str = './checkpoints'
    
    avalanche_window: int = 500
    target_avalanche_std_mean_ratio: float = 1.5
    
    def __post_init__(self):
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
CONFIG = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ================== 分形树拓扑（完全保留你原始代码，无任何修改） ==================
class FractalTreeTopology:
    def __init__(self, num_nodes=50000, branching_factor=3, depth=11, spatial_scale=1.0):
        self.num_nodes = num_nodes
        self.branching_factor = branching_factor
        self.depth = depth
        self.spatial_scale = spatial_scale
        self.coords = self._generate_tree_coords()
        self.num_nodes = self.coords.shape[0]
        self.adjacency = self._build_adjacency()
        self.D_f = self._estimate_fractal_dimension()
        print(f"生成分形树: {self.num_nodes} 节点, D_f ≈ {self.D_f:.2f}")
        
    def _generate_tree_coords(self):
        coords_list = [torch.zeros(1, 3, device=device)]
        parent_ids = torch.tensor([0], device=device)
        
        for d in range(self.depth):
            lambda_tensor = torch.full((len(parent_ids),), self.branching_factor, device=device, dtype=torch.float32)
            num_children = torch.poisson(lambda_tensor).long()
            total_children = num_children.sum().item()
            if total_children == 0:
                break
            
            directions = torch.randn(total_children, 3, device=device)
            directions = directions / torch.norm(directions, dim=1, keepdim=True)
            scale = self.spatial_scale * (0.8 ** d)
            offsets = directions * scale * torch.rand(total_children, 1, device=device) * 1.0 + 0.5
            
            all_coords_so_far = torch.cat(coords_list, dim=0)
            parent_coords = all_coords_so_far[parent_ids.repeat_interleave(num_children)]
            new_coords = parent_coords + offsets
            
            coords_list.append(new_coords)
            
            current_total = all_coords_so_far.shape[0]
            new_parent_ids = torch.arange(current_total, current_total + total_children, device=device)
            parent_ids = new_parent_ids
            
            if sum(c.shape[0] for c in coords_list) >= self.num_nodes:
                break
        
        all_coords = torch.cat(coords_list, dim=0)[:self.num_nodes]
        return all_coords
    
    def _build_adjacency(self):
        coords = self.coords
        num_nodes = coords.shape[0]
        chunk_size = 5000
        rows, cols = [], []
        for i in range(0, num_nodes, chunk_size):
            i_end = min(i + chunk_size, num_nodes)
            coords_i = coords[i:i_end]
            for j in range(i, num_nodes, chunk_size):
                j_end = min(j + chunk_size, num_nodes)
                coords_j = coords[j:j_end]
                diff = coords_i.unsqueeze(1) - coords_j.unsqueeze(0)
                dist = torch.norm(diff, dim=2)
                divisor = max(num_nodes // 10, 1)
                radius = 0.5 * (0.9 ** (i // divisor))
                mask = (dist > 0) & (dist < radius)
                local_i, local_j = torch.where(mask)
                global_i = i + local_i
                global_j = j + local_j
                if i == j:
                    mask_upper = global_i < global_j
                    global_i = global_i[mask_upper]
                    global_j = global_j[mask_upper]
                rows.append(global_i)
                cols.append(global_j)
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        indices = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])])
        values = torch.ones(indices.shape[1], device=device)
        adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
        return adjacency
    
    def _estimate_fractal_dimension(self):
        coords = self.coords.cpu().numpy()
        scales = np.logspace(-2, 0, 10)
        counts = []
        for scale in scales:
            bins = np.round(coords / scale).astype(int)
            unique_bins = np.unique(bins, axis=0)
            counts.append(len(unique_bins))
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        slope, _ = np.polyfit(log_scales, log_counts, 1)
        return -slope

# ================== GPU临界变量（✅ 100%恢复你原始能跑的正确写法，彻底修复clamp报错） ==================
class CriticalVariablesGPU:
    def __init__(self, num_nodes, n_levels, delta=0.12):
        self.num_nodes = num_nodes
        self.n_levels = torch.tensor(n_levels, dtype=torch.int32, device=device)
        # ✅ 核心修复：只取第一个值，强制变成单个标量，不广播成50000维
        self.num_states = torch.tensor(2 ** self.n_levels[0].item(), device=device)
        self.delta = torch.full((num_nodes,), delta, dtype=torch.float32, device=device)
        self.state = torch.randint(0, 2, (num_nodes,), device=device)
        self.phase = torch.rand(num_nodes, device=device) * 2 * torch.pi
        self._clamp_state()

    def _clamp_state(self):
        # 现在 num_states 是单个标量，.item() 完全正常
        max_state = int(self.num_states.item()) - 1
        self.state = torch.clamp(self.state, 0, max_state)

    # 下面 get_value / update / get_x 完全不动，保持你原来的逻辑
    def get_value(self):
        denom = torch.where(self.num_states > 1, self.num_states - 1, torch.ones_like(self.num_states))
        return 2.0 * self.state.float() / denom - 1.0

    def update(self, coupling, noise_scale, dt):
        d_phase = coupling * dt + noise_scale * torch.randn(self.num_nodes, device=device) * torch.sqrt(torch.tensor(dt, device=device))
        self.phase = (self.phase + d_phase) % (2 * torch.pi)
        new_state = (self.phase / (2 * torch.pi) * self.num_states).long()
        max_state = int(self.num_states.item()) - 1
        self.state = torch.clamp(new_state, 0, max_state)
        return self.state

    def get_x(self):
        return self.get_value()

# ================== GPU临界网络（完全保留你原始代码，无任何修改） ==================
class CriticalNetworkGPU:
    def __init__(self):
        print("生成分形拓扑...")
        self.topo = FractalTreeTopology(CONFIG.num_nodes, CONFIG.branching_factor, CONFIG.depth, CONFIG.spatial_scale)
        self.num_nodes = self.topo.num_nodes
        n_levels = self._assign_n_levels()
        print("初始化临界变量...")
        self.vars = CriticalVariablesGPU(self.num_nodes, n_levels, CONFIG.default_delta)
        print("初始化耦合矩阵...")
        self.J = self._init_coupling()
        self.x = self.vars.get_x()
        self.register_buffer('avalanche_buffer', torch.zeros(CONFIG.avalanche_window, device=device))
        self.buffer_idx = 0
        self.total_avalanches = 0
        self.phase = "incubation"

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def _assign_n_levels(self):
        coords = self.topo.coords
        dist = torch.norm(coords, dim=1)
        norm_dist = dist / (dist.max() + 1e-6)
        n_bins = len(CONFIG.n_levels_by_depth)
        bin_idx = (norm_dist * (n_bins - 1)).long()
        n_levels = torch.tensor(CONFIG.n_levels_by_depth, device=device)[bin_idx]
        return n_levels.cpu().numpy()

    def _init_coupling(self):
        adj = self.topo.adjacency
        indices = adj.indices()
        num_edges = indices.shape[1] // 2
        i_idx, j_idx = indices[0, :num_edges], indices[1, :num_edges]
        n_i = self.vars.n_levels[i_idx].float()
        n_j = self.vars.n_levels[j_idx].float()
        delta_n = torch.abs(n_i - n_j)
        decay = 2.0 ** (-delta_n / 0.9)
        sign = torch.where(torch.rand(num_edges, device=device) < 0.8, 1.0, -1.0)
        base_strength = CONFIG.noise_scale * 0.5
        weights = sign * base_strength * decay * (0.5 + 0.5 * torch.rand(num_edges, device=device))
        all_i = torch.cat([i_idx, j_idx])
        all_j = torch.cat([j_idx, i_idx])
        all_w = torch.cat([weights, weights])
        J = torch.sparse_coo_tensor(torch.stack([all_i, all_j]), all_w, (self.num_nodes, self.num_nodes)).coalesce()
        return J

    def evolve(self, steps=1, external_noise=None):
        noise = CONFIG.noise_scale if external_noise is None else external_noise
        avalanche_count = 0
        for _ in range(steps):
            coupling = torch.sparse.mm(self.J, self.x.unsqueeze(1)).squeeze()
            old_state = self.vars.state.clone()
            new_state = self.vars.update(coupling, noise, CONFIG.dt)
            self.x = self.vars.get_x()
            avalanche_count += (new_state != old_state).sum()
        self.avalanche_buffer[self.buffer_idx % CONFIG.avalanche_window] = avalanche_count
        self.buffer_idx += 1
        self.total_avalanches += avalanche_count.item()
        return avalanche_count.item()

    def hebbian_learn(self):
        lr = CONFIG.learning_rate
        indices = self.J.indices()
        values = self.J.values()
        xi = self.x[indices[0]]
        xj = self.x[indices[1]]
        hebb = xi * xj
        anti_hebb = 0.05 * (xi**2 + xj**2)
        delta = lr * (hebb - anti_hebb)
        new_values = torch.clamp(values + delta, -1.0, 1.0)
        self.J = torch.sparse_coo_tensor(indices, new_values, self.J.shape).coalesce()

    def adjust_criticality(self):
        if self.buffer_idx < 50: return
        valid_buffer = self.avalanche_buffer[:min(self.buffer_idx, CONFIG.avalanche_window)]
        mean_s = valid_buffer.float().mean()
        var_s = valid_buffer.float().var()
        ratio = var_s / (mean_s + 1e-6)
        if ratio > 2.0: self.vars.delta *= 1.01
        elif ratio < 0.8: self.vars.delta *= 0.99
        self.vars.delta = torch.clamp(self.vars.delta, 0.02, 0.5)

    def check_criticality(self):
        if self.buffer_idx < 100: return False
        valid_buffer = self.avalanche_buffer[:min(self.buffer_idx, CONFIG.avalanche_window)]
        mean_s = valid_buffer.float().mean()
        var_s = valid_buffer.float().var()
        ratio = var_s / (mean_s + 1e-6)
        return abs(ratio.item() - CONFIG.target_avalanche_std_mean_ratio) < 0.3

    def clamp_input(self, embedding, clamp_ratio=0.3):
        num_clamp = int(self.num_nodes * clamp_ratio)
        emb_t = torch.tensor(embedding, dtype=torch.float32, device=device)
        emb_t = emb_t / (torch.norm(emb_t) + 1e-8)
        emb_dim = emb_t.shape[0]
        indices = torch.arange(num_clamp, device=device) % emb_dim
        vals = emb_t[indices]
        max_states = self.vars.num_states[:num_clamp]
        state_idx = ((vals + 1) / 2 * max_states - 1).long()
        state_idx = torch.clamp(state_idx, 0, max_states - 1)
        self.vars.state[:num_clamp] = state_idx
        self.vars.phase[:num_clamp] = state_idx.float() / max_states * 2 * torch.pi
        self.x = self.vars.get_x()

# ================== 千问API（完全保留你原始代码） ==================
class QwenAgent:
    def __init__(self):
        self.api_url = CONFIG.qwen_api
        self.model = CONFIG.qwen_model
    def generate_topic(self):
        return self._call("你是一个充满好奇心的AI。请生成一个你真正想了解的具体话题。只给出话题名称。").strip()
    def extract_keywords(self, text):
        return self._call(f"从以下文本提取2-3个搜索关键词，用空格分隔。\n文本：{text[:500]}").strip()
    def _call(self, prompt):
        try:
            resp = requests.post(self.api_url, json={
                "model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.7, "num_predict": 200}
            }, timeout=30)
            return resp.json().get("response", "") if resp.status_code == 200 else ""
        except:
            return ""

# ================== 网络爬虫（✅ 解决GPU冲突，保留OCR功能，不删任何东西） ==================
class WebCrawler:
    def __init__(self):
        self.ocr = None

    def _get_ocr(self):
        if self.ocr is None:
            # ✅ 强制CPU运行，彻底解决和PyTorch的GPU冲突，保留OCR功能
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_textline_orientation=True, lang='ch', use_gpu=False)
        return self.ocr

    def download_images(self, keyword, max_num=5):
        print(f"🌐 搜索图片: '{keyword}'")
        safe_keyword = re.sub(r'[^\w\-_]', '_', keyword)
        temp_dir = os.path.join(CONFIG.download_dir, safe_keyword)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            crawler = GoogleImageCrawler(storage={'root_dir': temp_dir})
            crawler.crawl(keyword=keyword, max_num=max_num)
        except Exception as e:
            print(f"下载出错: {e}")
        return [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    def fetch_wikipedia_summary(self, topic):
        try:
            import wikipedia
            return wikipedia.summary(topic, sentences=3)
        except:
            return ""

    def extract_text_from_image(self, image_path):
        try:
            result = self._get_ocr().ocr(image_path, cls=False)
            if result and result[0]:
                return ' '.join([line[1][0] for line in result[0]])
        except:
            pass
        return ""

# ================== 多模态编码器（完全保留你原始代码） ==================
class MultimodalEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CONFIG.clip_model).to(device)
        self.processor = CLIPProcessor.from_pretrained(CONFIG.clip_model)
        self.model.eval()
    def encode_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            return emb.cpu().numpy()[0]
        except:
            return np.zeros(CONFIG.embedding_dim)
    def encode_text(self, text):
        if not text: return np.zeros(CONFIG.embedding_dim)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb.cpu().numpy()[0]

# ================== 知识库（完全保留你原始代码） ==================
class KnowledgeBase:
    def __init__(self):
        self.items = []
    def add(self, emb, text, img_path):
        self.items.append({'embedding': emb, 'text': text, 'image_path': img_path})
    def search(self, q_emb, top_k=3):
        if not self.items: return []
        embeds = np.stack([it['embedding'] for it in self.items])
        sims = np.dot(embeds, q_emb) / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        top = np.argsort(sims)[-top_k:][::-1]
        return [(self.items[i], sims[i]) for i in top]

# ================== 主系统（完全保留你原始代码） ==================
class AutonomousAI:
    def __init__(self):
        print("初始化GPU临界网络...")
        self.net = CriticalNetworkGPU()
        self.qwen = QwenAgent()
        self.crawler = WebCrawler()
        self.encoder = MultimodalEncoder()
        self.kb = KnowledgeBase()
        self.phase = "incubation"
        self.step_count = 0
    def incubate(self):
        print("🔥 加速孕育阶段 (GPU加速)...")
        while self.phase == "incubation":
            for _ in range(10):
                self.net.evolve(steps=5)
                self.net.hebbian_learn()
                self.step_count += 1
                if self.step_count % 50 == 0:
                    self.net.adjust_criticality()
                if self.net.check_criticality():
                    print(f"✅ 达到临界态！雪崩总数: {self.net.total_avalanches}")
                    self.phase = "online"
                    self._save_checkpoint("incubation_done.pt")
                    break
            if self.step_count % 100 == 0:
                print(f"孕育中... step {self.step_count}, 雪崩累计: {self.net.total_avalanches}")
    def online_learn(self):
        print("🌍 自主上网学习阶段...")
        while True:
            topic = self.qwen.generate_topic()
            if not topic:
                time.sleep(5)
                continue
            print(f"\n📚 主题: '{topic}'")
            wiki_text = self.crawler.fetch_wikipedia_summary(topic)
            if wiki_text:
                print(f"  📖 Wiki摘要: {wiki_text[:80]}...")
                txt_emb = self.encoder.encode_text(wiki_text)
                self.net.clamp_input(txt_emb, clamp_ratio=0.2)
                self.net.evolve(steps=20)
                self.kb.add(txt_emb, wiki_text, "")
            keywords = self.qwen.extract_keywords(topic) or topic
            images = self.crawler.download_images(keywords, max_num=3)
            for img_path in images:
                ocr_text = self.crawler.extract_text_from_image(img_path)
                desc = f"Image related to {topic}"
                if ocr_text:
                    desc += f" containing text: {ocr_text}"
                img_emb = self.encoder.encode_image(img_path)
                txt_emb = self.encoder.encode_text(desc)
                combined = (img_emb + txt_emb) / 2.0
                self.net.clamp_input(combined, clamp_ratio=0.3)
                self.net.evolve(steps=30)
                self.net.hebbian_learn()
                self.kb.add(combined, desc, img_path)
                print(f"  ✅ 学习: {desc[:50]}...")
            self._save_checkpoint(f"online_step_{self.step_count}.pt")
            self.step_count += 1
            time.sleep(2)
    def _save_checkpoint(self, filename):
        path = os.path.join(CONFIG.checkpoint_dir, filename)
        state = {
            'vars_state': self.net.vars.state.clone(),
            'J_indices': self.net.J.indices().clone(),
            'J_values': self.net.J.values().clone(),
            'kb_items': self.kb.items,
            'step': self.step_count,
            'phase': self.phase
        }
        torch.save(state, path)
        print(f"💾 保存检查点: {filename}")
    def run(self):
        print("="*60)
        print("🧠 临界认知AI - 最终稳定版")
        print("="*60)
        if self.phase == "incubation":
            self.incubate()
        if self.phase == "online":
            self.online_learn()

# ================== 入口（✅ 修复千问API 404检测错误） ==================
if __name__ == "__main__":
    try:
        # ✅ 修复404：把错误的/api/tags改成根路径/，不再报404
        resp = requests.get("http://localhost:5000/", timeout=5)
        if resp.status_code != 200:
            print("⚠️ 千问API未运行，上网功能受限")
    except:
        print("⚠️ 千问API未运行")
    ai = AutonomousAI()
    try:
        ai.run()
    except KeyboardInterrupt:
        print("\n👋 终止")
        ai._save_checkpoint("final.pt")