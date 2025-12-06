## GUI 隐私识别 VLM 工具

本项目提供一个轻量脚本，调用视觉语言模型（VLM）对桌面 / 应用截图中的隐私相关信息进行识别，输出带标注框的图片和对应的 JSON 标注文件。

> English version: see **[README.md](README.md)**.

### 1. 环境与安装

- **Python 版本**：建议使用 Python 3.8 及以上。
- **必须安装的 pip 包**：

```bash
pip install openai pillow
```

### 2. 配置 `base_url` 和 `api_key`

`privacy.py` 顶部创建了一个 OpenAI 兼容的客户端：

```python
from openai import OpenAI

client = OpenAI(
    base_url="",
    api_key=os.getenv("OPENAI_API_KEY", ""),
)
```

你可以按以下两种方式配置：

- **方式 A – 使用环境变量（推荐）**

```bash
export OPENAI_API_KEY="你的_API_Key"
```

- **方式 B – 直接在代码中写死（仅适合本地快速测试）**

```python
client = OpenAI(
    base_url="",
    api_key="你的_API_Key",
)
```
### 3. 数据目录结构

脚本期望每个任务的数据目录形如：

```text
<task_folder>/
  task_result.json      # 包含 "goal"
  traj.jsonl            # 每行包含一个带 "response" 的 "manager" 对象
  images/
    *.png               # 每一步的截图
```

本仓库中的示例：

```text
data/Gmail_View_third_email_in_Sen/
  task_result.json
  traj.jsonl
  images/
    screenshot_....png
```

### 4. 运行命令说明

基本命令：

```bash
python privacy.py <task_folder> --model <model_name>
```

示例：
```bash
python privacy.py data/Gmail_View_third_email_in_Sen --model "google/gemini-3-pro-preview"
```

脚本主要流程：

1. 从 `task_result.json` 读取任务 **goal**；
2. 从 `traj.jsonl` 逐行读取 **manager.response**；
3. 将每个 response 与 `images/` 下对应的截图一一配对；
4. 使用 `get_prompt_template()` 构造详细的隐私分类提示词；
5. 为每张图调用 VLM（文本 + 图像）；
6. 解析模型输出，并：
   - 在原图上画出标注框，保存到  
     `annotations/<model_name>/..._annotated.png`
   - 生成汇总 JSON 文件：  
     `annotations/<model_name>/ai_results.json`

### 5. 关于 Claude 与归一化参数

不同模型 / 服务商在坐标系上的约定可能不同。  
当前 `privacy.py` 中的 `convert_normalized_coords_to_pixels()` 默认假设模型返回的是 **0–1000 归一化坐标**：

```python
x1 = coords["x1"] * width / 1000
y1 = coords["y1"] * height / 1000
x2 = coords["x2"] * width / 1000
y2 = coords["y2"] * height / 1000
```

对于某些 Claude 的配置，实际更像是基于固定分辨率做归一化。  
例如在 1080x2400 的截图上，你可能通过实验发现更合适的缩放为：

- \( \text{scale\_x} = \text{img\_width} / 705 \)
- \( \text{scale\_y} = \text{img\_height} / 1567 \)

此时可以按类似方式修改转换逻辑，例如：

```python
scale_x = img_width / 705
scale_y = img_height / 1567
x1 = coords["x1"] * scale_x
y1 = coords["y1"] * scale_y
...
```

由于这完全依赖于**具体服务商和模型的返回格式**，推荐做法是：

1. 先在少量 **测试截图** 上跑一遍；
2. 打开 `annotations/<model_name>/` 下的标注图片；
3. 手动调节 `convert_normalized_coords_to_pixels()` 中的归一化公式，直到标注框与模型描述的区域尽量对齐。

### 6. 其它说明

- 解析函数 `parse_ai_output()` 要求模型严格按照提示词中的格式输出，不符合格式的行会被跳过。
- 当前脚本只做了最基本的错误处理，未包含重试、限流、并发等工程化功能，更适合作为研究 / Demo 使用。
- 大模型的输出结果可能不理想，主要有两方面原因：  
  (1) 大模型对“隐私”这一概念的理解能力有限，可能无法识别出图中的部分隐私信息；  
  (2) 大模型的 grounding 能力有限，即使识别出了隐私内容，也可能无法给出足够精确的位置（坐标）。

