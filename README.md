# AI 学习项目

从基础计算图到模型结构的项目式学习仓库。支持 Ubuntu 和 Windows；网站不需要构建工具。

## 内容

- [学习路线图](ROADMAP.md)：循序渐进的项目与实验。
- [模型博物馆](site/README.md)：离线三维教学示意，包含早期规则推理到 Transformer 等展品。
- [微型自动微分项目](projects/p01-micrograd/README.md)：动手实现算子、对照数值梯度和 PyTorch。

## 在两台电脑上使用

在 GitHub 创建你自己的空仓库后，将其地址替换下面的 `<仓库地址>`：

```bash
git clone <仓库地址>
cd <克隆后的目录>
```

每次开始前先运行 `git pull`；完成修改后运行 `git add .`、`git commit -m "描述更改"` 和 `git push`。首次上传已有本地项目的完整步骤见下文。发生冲突时先解决冲突再提交，不要强制覆盖另一台电脑的修改。

## 打开网站（Windows / Ubuntu）

在仓库根目录的终端运行：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

若 Ubuntu 上 `python` 不存在，可用 `python3`；Windows 上也可用 `py -3`。打开浏览器访问：

- 模型博物馆：`http://127.0.0.1:8000/site/`
按 `Ctrl+C` 停止服务。三维页面需要支持 WebGL 的浏览器。数学学习站 `mathbank/` 是旁边独立的项目，不包含在此仓库中；它的浏览器本地进度也不会通过 Git 同步。

## 设置 Python 学习环境

先在各自系统安装 Miniconda 或 Anaconda，然后在仓库根目录运行：

```bash
conda env create -f environment.yml
conda activate ai-learn
```

环境仅覆盖第一个项目需要的 Python、NumPy、pytest 和 PyTorch。PyTorch 默认安装包的设备支持因平台而异；初期使用 CPU 即可，无需连接外置显卡。需要 GPU 时，再根据当时的操作系统和显卡使用 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)；不要复制另一台电脑的 Conda 环境目录。

Windows PowerShell 与 Ubuntu 终端均可运行：

```bash
cd projects/p01-micrograd
python -m pytest -q tests/test_api.py
python eval/acceptance.py
```

这是学习骨架：尚未实现的算子会报 `NotImplementedError`，验收失败是初始状态，不是安装失败。逐步实现后再运行完整验收；PyTorch 对照测试需要成功安装 `torch`。

## 首次上传自己的 GitHub 仓库

本项目不会自动创建远程仓库或上传。先在 GitHub 手动创建一个**空仓库**（不要自动添加 README、许可证或 `.gitignore`），可选择私有仓库；在本地仓库根目录检查内容，然后：

```bash
git init
git add .
git status
git commit -m "Initial learning project"
git branch -M main
git remote add origin <仓库地址>
git push -u origin main
```

`git status` 中确认没有数据集、权重、密钥或不想公开的笔记再提交。GitHub 忽略规则不会自动移除已跟踪的文件，上传前请仔细检查。`site/model-museum-3d.desktop` 和 `site/launch.sh` 是原电脑专用的菜单配置，默认不纳入仓库；它们仍保留在本机，不影响已有快捷方式。其他电脑可使用上述跨平台启动命令。

## 同步边界

不提交 Conda 环境、训练数据、模型权重、运行记录或密钥；`.gitignore` 提供基本保护，仍需在每次提交前检查。浏览器 `localStorage` 中的进度属于本机浏览器，不随 Git 同步。GitHub 不适合直接保存大型数据集和模型权重。
