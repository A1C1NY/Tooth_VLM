import os
import shutil
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, 'images')
output_dir = os.path.join(current_dir, 'output')
similarity_threshold = 0.947

# 检查输入目录是否存在
if not os.path.exists(input_dir):
    print(f"Error:Input directory '{input_dir}' not found.")
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的ResNet模型
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后的全连接层
model = model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

paths = []
# 遍历目录及其子目录（如果是为了只扫描当前目录，os.listdir已经足够）
if not os.path.exists(input_dir):
    print(f"Directory {input_dir} does not exist.")
    exit()

for f in os.listdir(input_dir):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        paths.append(os.path.join(input_dir, f))

if len(paths) == 0:
    print(f"No images found in {input_dir}")
    exit()

# features 提取（对每张图生成若干旋转/镜像变体以提高对旋转/镜像的鲁棒性）
def get_variants(img):
    variants = []
    for angle in (0, 90, 180, 270):
        variants.append(img.rotate(angle, expand=True))
    variants.append(ImageOps.mirror(img))
    variants.append(ImageOps.flip(img))
    return variants

# 计算每张图片的特征（包括变体），并进行归一化处理
features_variants = []
for path in tqdm(paths, desc='Extracting features (with variants)'):
    img = Image.open(path).convert('RGB')
    embs = []
    for v in get_variants(img):
        img_tensor = transform(v).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).squeeze().cpu().numpy()
            norm = np.linalg.norm(feat)
            if norm == 0:
                emb = feat
            else:
                emb = feat / norm
        embs.append(emb)
    features_variants.append(np.stack(embs))  # (K, D)

# 计算两张图片之间的最大变体相似度作为它们的相似度
N = len(features_variants)
similarity_matrix = np.eye(N, dtype=float)
for i in tqdm(range(N), desc='Computing similarity'):
    for j in range(i + 1, N):
        sims = cosine_similarity(features_variants[i], features_variants[j])  # (K, K)
        max_sim = float(np.max(sims))
        similarity_matrix[i, j] = similarity_matrix[j, i] = max_sim

# 并查集实现
parents = list(range(len(paths)))

def find(x):
    if parents[x] != x:
        parents[x] = find(parents[x]) 
    return parents[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parents[root_y] = root_x

# 遍历相似度矩阵并将满足条件的节点合并
for i in range(len(paths)):
    for j in range(i + 1, len(paths)):
        if similarity_matrix[i][j] > similarity_threshold:
            union(i, j)

# 重置 parents 状态以确保并查集正确
for i in range(len(paths)):
    find(i)

clusters = {}
for i in range(len(paths)):
    root = find(i)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(paths[i])


if os.path.exists(output_dir):
    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        print(f"Warning: Could not remove directory {output_dir}: {e}")
        # 如果无法删除，可能是有文件正在被使用，但通常不应该影响脚本核心逻辑
        pass # 继续尝试重建

os.makedirs(output_dir, exist_ok=True) # exist_ok=True 避免文件夹已删除失败再次创建时报错

# 将只有 1 张图的簇集中到一个文件夹中
result = os.path.join(output_dir, 'single_images')
os.makedirs(result, exist_ok=True)

duplicates_log = []
for root, group in clusters.items():
    if len(group) == 1:
        src = group[0]
        base = os.path.basename(src)
        dest_path = os.path.join(result, base)
        # 如果文件名冲突，添加簇 id 避免覆盖
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(base)
            dest_path = os.path.join(result, f"{name}_{root}{ext}")
        shutil.copy2(src, dest_path)
        print(f"Cluster {root}: 1 image -> {dest_path}")
    else:
        # 多图簇：选择簇中的第一张图并复制到 result，同时记录该簇的所有图片和所选图片
        selected = group[0]  # 选择簇中的第一张图
        base = os.path.basename(selected)
        dest_path = os.path.join(result, base)
        # 如果文件名冲突，添加簇 id 避免覆盖
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(base)
            dest_path = os.path.join(result, f"{name}_{root}{ext}")

        shutil.copy2(selected, dest_path)
        # 记录日志项（仅保存文件名以便可读）
        group_basenames = [os.path.basename(p) for p in group]
        duplicates_log.append({
            'cluster': root,
            'images': group_basenames,
            'selected': os.path.basename(dest_path)
        })
        print(f"Cluster {root}: {len(group)} images. Selected 1st -> {dest_path}")

# 将重复簇信息写入文本文件
if duplicates_log:
    log_path = os.path.join(output_dir, 'duplicates.txt')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            for item in duplicates_log:
                f.write(f"Cluster {item['cluster']}\n")
                f.write("Images:\n")
                for im in item['images']:
                    f.write(f"  {im}\n")
                f.write(f"Selected: {item['selected']}\n")
                f.write("\n")
        print(f"Duplicates log written to: {log_path}")
    except Exception as e:
        print(f"Warning: could not write duplicates log: {e}")

print("Processing complete!")


