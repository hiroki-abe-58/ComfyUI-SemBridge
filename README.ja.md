# ComfyUI 用 SemBridge

音声合成・歌声合成・「話し声と歌声が混ざった」生成を、公式モデル **SemBridge**
（[ASLP-lab/SemBridge](https://github.com/ASLP-lab/SemBridge)、論文 [arXiv:2608.07462](https://arxiv.org/abs/2608.07462)）で
ComfyUI 上から行うノード集です。モデルのコードは ComfyUI のプロセス内で動き（サブプロセスなし、実行時ダウンロードなし）、
出力は標準の `AUDIO` 型（モノラル・44.1 kHz）です。

> **非公式**の統合です。SemBridge の著者（ASLP-lab）や Comfy Org とは無関係で、承認も受けていません。
> 詳しい英語版は [README.md](README.md)、検証の生データは [docs/VERIFICATION.md](docs/VERIFICATION.md) にあります。

## ノード

| ノード | 役割 |
| --- | --- |
| SemBridge Loader | `models/sembridge` のモデルフォルダ、dtype、デバイス、常駐の有無を選ぶ |
| SemBridge Text to Speech | 中国語・英語のテキストから音声。参照音声を付ければその声質で生成 |
| SemBridge Singing | 音符に揃えた歌詞（歌詞・MIDI 音高・秒数）から歌声 |
| SemBridge Speech + Singing | 話す区間と歌う区間を順に並べて、ひと続きの声で生成 |
| SemBridge Score Preview | モデルを読み込まずに楽譜を検証し、モデルに渡るトークン列を表示 |
| SemBridge Unload | 常駐モデルを解放（GPU・RAM） |

## 何をどこまで確認したか（Windows 11 / RTX 5090 32 GB）

- 公式リポジトリ付属の 10 例（音声 4・歌 4・混合 2）で、このノードの出力が公式の `SemBridge.generate` と
  **16 ビット PCM でサンプル単位まで一致**しました。同一 Python 環境どうしの比較で、torch 2.8.0（公式対応範囲）で 10/10、
  torch 2.14.1 + transformers 5.20（現行 ComfyUI の環境）でも 10/10 です。
- ComfyUI v0.38.0 の実キュー（`POST /prompt`）で 10 例とも成功。中断、エラー表示、解放、3 種の dtype、CPU デバイスも
  実モデルで確認しました。
- デモの歌声は、書いた音符との音高誤差の中央値が 0.15 半音（7 音すべて 1 半音以内）でした。話し声のデモは Whisper で
  文字起こしして確認しています（詳細は検証ページ）。人による試聴はまだ行っていません。
- GitHub Actions（Ubuntu と Windows、torch 2.8 / 2.14）では、学習済み重みを使わない小さなランダム初期化モデルで CPU テストと
  実 ComfyUI キューを実行します。

**未確認:** GPU 付きの Linux / macOS、AMD・Apple GPU、量子化、複数 GPU、中国語・英語以外、40 秒を超える入力。

## インストール

1. ComfyUI（V3 ノード API、v0.38.0 で確認）と Python 3.10 以上。速度のため NVIDIA GPU 推奨。
2. このフォルダを `ComfyUI/custom_nodes/` に置き、追加の依存（`soundfile` のみ）を ComfyUI の Python に入れます。
   `pip install sembridge` はしないでください（torch 2.9 未満に固定されるため。モデルコードは同梱済みです）。

   ```
   python -m pip install -r custom_nodes/ComfyUI-SemBridge/requirements.txt
   ```
3. 公式モデルフォルダ（約 4.3 GB、モデルカード上は Apache-2.0）を `models/sembridge/` にダウンロードします。

   ```
   python custom_nodes/ComfyUI-SemBridge/tools/download_model.py --models-dir ComfyUI/models/sembridge
   ```
4. ComfyUI を再起動し、`workflows/gui/` のワークフローを読み込みます。

ノード自体は何もダウンロードしません。

## 使い方の要点

- **歌声の楽譜**は 1 行 1 音で `歌詞 音高 秒数 [note_type]`（音高は MIDI 番号、休符は `<SP>` と 0、`note_type` 3 は
  「前の歌詞を別の音高で続ける」）。公式の JSON も使えます。**MIDI ファイルの取り込みではありません**
  （歌詞が音符に揃っている必要があります）。
- 参照音声を使うときは、その**一字一句の書き起こし**を `prompt_text` に入れてください。権利のある録音だけを使ってください。
- `max_new_patches` は長さの上限（1 パッチ = 80 ms、500 = 40 秒）。通常はそれより早く終わります。

## 実測値と注意点

RTX 5090、bfloat16：生成時間は音声 1 秒あたり 約 1.0〜1.6 秒（音声）／約 2.2〜2.7 秒（歌・混合）、モデルが確保する
VRAM は 2.7〜3.9 GiB（29 秒の長い発話で 4.3 GiB）。つまりこの GPU でもリアルタイムより遅いです。

- 同じモデル・入力・シード・設定・**環境**なら同じ音声になります。torch のバージョンや GPU が違うと数値が少し変わり、
  停止位置が数パッチずれることがあります。
- **ロード直後の最初の実行**：公式のコーデックは Snake 活性化を `@torch.jit.script` 関数で計算しており、TorchScript は最初の数回の
  呼び出しの後にこの関数を再最適化するため、公式コードでは「ロード直後の最初の生成」だけ丸めが異なります（同じリクエストとシードで
  最初は 69 パッチ、以降は 67 パッチ、という例がありました）。Loader の `codec_numerics = reproducible`（既定）は同じ式を通常の
  PyTorch 演算で計算するので、毎回同じ音声になります。`official-torchscript` は公式の関数をそのまま使います。
- とても長い文章では途中で止まることがあります（29 秒の試験で最後の 6 語が欠けました）。文ごとに分けてください。
- 出力の音量はまちまちです（公式例でも -12〜-31 dBFS）。必要なら後段で正規化してください。
- テキストの先頭の空白は取り除きます（transformers 4.x と 5.x で先頭の空白の扱いが異なるため）。

## ライセンス

- このノード集のコード：**MIT**（[LICENSE](LICENSE)）。
- `sembridge_comfy/vendor/sembridge`：公式パッケージの無改変コピー、**Apache-2.0**（第三者表示は [NOTICE](NOTICE) と
  `sembridge_comfy/vendor/LICENSE`）。
- モデルの重みは**同梱していません**（モデルカードは Apache-2.0）。公式サンプルの参照録音も同梱しておらず、それらは
  上記ライセンスの対象外です。`docs/audio/` のデモは、参照録音なしでモデルだけから生成した合成音声です。
- 参照録音は許可を得たものだけを使い、合成音声であることを適切に明示してください（モデルカードの依頼でもあります）。

引用は英語版 README の BibTeX をご利用ください。
