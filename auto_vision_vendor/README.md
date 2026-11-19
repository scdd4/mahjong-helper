# majsoul-vision

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-4.x%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Release](https://img.shields.io/badge/release-v1.0.0-4B8BBE?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-3DA639?style=for-the-badge)
[![App](https://img.shields.io/badge/client-majsoul--helper-black?style=for-the-badge&logo=apple)](https://github.com/sdtafxy/majsoul-helper)

从雀魂（Mahjong Soul）的对局截图中读取牌局状态，输出结构化 JSON。纯本地算法，不联网、不注入游戏，一张截图约 0.3 秒。

## 能读到什么

- **自家手牌**：暗牌、摸牌、副露（吃 / 碰 / 杠）
- **四家牌河**：按各家自己的视角排序（第一行第一张到最后一行最后一张）
- **各家副露**：整组张数、被鸣的横放牌、牌名
- **宝牌**：指示牌与实际宝牌（含三麻字牌循环）
- **置信度**：每张牌附带匹配分，低分牌单独标记，方便人工复核

输出坐标是原图像素，可以直接拿去画图或做上层逻辑。

## 使用

需要 Python 3.9 及以上，装两个依赖即可：

```bash
pip install numpy opencv-python
```

克隆或下载本项目后，在项目目录里：

```bash
python recognize.py screenshot.png
```

常用参数：

```bash
python recognize.py screenshot.png --pretty          # 缩进输出
python recognize.py screenshot.png --seats own,left  # 只读指定座位
python recognize.py screenshot.png -o result.json    # 写入文件
python recognize.py screenshot.png --export out      # 导出调试图（原图 + JSON + 标注）
```

也可以当库用：

```python
from recognize import read

r = read("screenshot.png")
print([t["name"] for t in r["hand"]["tiles"]])
print({k: len(v["tiles"]) for k, v in r["seats"].items()})
```

## 输出格式

```jsonc
{
  "ok": true,
  "elapsed_ms": 287,
  "image": [2400, 1720],              // [宽, 高]
  "canvas": [4, 267, 2396, 1611],     // 游戏画面在原图中的矩形 [x0, y0, x1, y1]
  "hand": {
    "tiles": [
      {"name": "4m", "score": 0.91, "quad": [[x, y], ...], "is_drawn": false}
    ],
    "meld_tiles": 3,
    "meld_rect": [x0, y0, x1, y1]     // 没有副露时为 null
  },
  "seats": {
    "own": {
      "name": "自家",
      "tiles": [
        {"name": "2m", "score": 0.83, "quad": [[x, y], ...],
         "is_riichi": false, "is_low": false, "is_blank": false}
      ]
    },
    "across": {"...": "同上"},
    "left": {"...": "同上"},
    "right": {"...": "同上"}
  },
  "melds": {
    "own": {
      "groups": [[
        {"name": "5m", "score": 0.77, "rotated": false, "quad": [[x, y], ...]}
      ]],
      "note": "认出 3/3 张"
    }
  },
  "dora": {
    "indicators": ["8p"],   // 指示牌
    "dora": ["9p"]          // 换算后的宝牌
  },
  "summary": {"total": 69, "recognized": 69, "blank": 0, "none": 0, "low_score": 12}
}
```

牌名写法：

- 万：`1m` 至 `9m`；筒：`1p` 至 `9p`；索：`1s` 至 `9s`
- 字牌：`1z` 东、`2z` 南、`3z` 西、`4z` 北、`5z` 白、`6z` 发、`7z` 中
- 赤宝牌：`0m`、`0p`、`0s`

`quad` 是四个顶点（左上、右上、右下、左下），侧家的牌是平行四边形。`score` 低于阈值（默认 0.70）时 `is_low` 为 true；`is_blank` 表示白板；`name` 为 null 表示这一张没认出来。

## 截图要求

- 雀魂网页版或客户端，游戏画面完整可见
- 画面比例 16:9，分辨率不低于 1280×720
- 窗口被其他窗口部分遮挡不影响（只要画面区域完整）
- 不支持：对局外的界面、回放时间轴遮挡画面

## 目录

```
majsoul-vision/
├── recognize.py     # 入口：命令行 + read()
├── reader.py        # 牌面识别（模板匹配、评分、摆向判定）
├── split.py         # 切分：牌河 / 手牌 / 副露区的单张位置
├── hand.py          # 自家手牌
├── meld.py          # 副露分组
├── orientation.py   # 摆向规则（横放、斜放、阈值）
├── paths.py         # 模板路径
├── templates/       # 牌面模板
└── tests/           # 回归测试
```

## 回归测试

`tests/regress.py` 需要一张标注好的截图，放到 `tests/river_gt.png` 和 `tests/river_gt.json`（格式见脚本里的 `load_gt`），然后：

```bash
python tests/regress.py
```

仓库不附带截图样本。

## 许可

MIT
