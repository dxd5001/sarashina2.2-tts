# デスクトップアプリとして使用する

このドキュメントは、Sarashina2.2-TTSをmacOSデスクトップアプリとして使用する方法を説明します。PyInstallerとPystrayを使用して、メニューバーから簡単に起動できる.appを作成できます。

## 特徴

- **FastAPIサーバー**: OpenAI互換のTTS APIエンドポイントを提供
- **非同期タスク処理**: 長時間の音声生成でもタイムアウトしない
- **メニューバーアプリ**: Pystrayによるトレイアイコンからの起動
- **ポータブル**: .appをどこに移動しても使用可能
- **MPS対応**: Apple Silicon (M1/M2/M3) で高速推論

## 前提条件

- macOS on Apple Silicon
- Native arm64 Python 3.12
- Xcode Command Line Tools
- PyInstaller
- pystray
- Pillow

## ビルド手順

### 1. モデルの準備

モデルファイルを固定の場所に配置します（.appには含まれません）。

```bash
mkdir -p ~/.sarashina_tts/pretrained_models
cp -r pretrained_models/* ~/.sarashina_tts/pretrained_models/
```

### 2. vLLM-metal環境を作成

macOSでvLLMを使用する場合は、vLLM-metalの公式インストールスクリプトを使用します。専用のPython 3.12環境が`~/.venv-vllm-metal`に作成されます。

```bash
uname -m
xcode-select --install
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
```

### 3. Sarashina TTSをvLLM-metal環境に追加

vLLM-metalが要求する依存関係を壊さないように、Sarashina TTSは依存関係を再解決せずにインストールします。

```bash
cd sarashina2.2-tts
source ~/.venv-vllm-metal/bin/activate
pip install -e . --no-deps
pip install pyinstaller pystray pillow
```

> **注意**: `pip install -e .`をそのまま実行すると、Sarashina TTS側の依存制約によりvLLM-metal環境の`torch`/`torchaudio`が変更される場合があります。vLLM-metalを使用する場合は`--no-deps`を付けてください。

### 4. .appのビルド

```bash
source ~/.venv-vllm-metal/bin/activate
pyinstaller "Sarashina TTS.spec"
```

ビルドが完了すると、`dist/Sarashina TTS.app`が作成されます。

### 5. .appの使用

.appを任意の場所（デスクトップなど）に移動し、ダブルクリックで起動します。メニューバーにアイコンが表示されます。

## 使用方法

### vLLM-metalを有効化して起動

vLLM-metalを使用する場合は、起動前に`SARASHINA_USE_VLLM=1`を設定します。

```bash
source ~/.venv-vllm-metal/bin/activate
export SARASHINA_USE_VLLM=1
python tts_launcher.py
```

.appビルド後に環境変数を指定して起動する場合は、ターミナルから以下のように実行します。

```bash
SARASHINA_USE_VLLM=1 open "dist/Sarashina TTS.app"
```

### メニューバーから起動

1. .appをダブルクリックして起動
2. メニューバーのアイコンをクリック
3. 以下のオプションが表示されます:
   - **Open Sarashina TTS**: Gradio UIをブラウザで開く
   - **Start Gradio**: Gradioサーバーを起動
   - **Stop Gradio**: Gradioサーバーを停止
   - **Start FastAPI**: FastAPIサーバーを起動
   - **Stop FastAPI**: FastAPIサーバーを停止
   - **Show Logs**: ログファイルを表示
   - **Quit**: アプリを終了

### サーバーのURL

- **Gradio UI**: http://localhost:7860
- **FastAPI**: http://localhost:8000
- **APIドキュメント**: http://localhost:8000/docs

## FastAPIエンドポイント

### 非同期TTS生成（推奨）

長時間の音声生成でもタイムアウトしない非同期エンドポイントです。

#### 1. タスクを作成

```bash
curl -X POST "http://localhost:8000/tts" \
  -F "prompt_file=@reference.wav" \
  -F "prompt_text=これは参照音声の転写です" \
  -F "text=生成したいテキスト"
```

レスポンス:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 2. タスクステータスを確認

```bash
curl "http://localhost:8000/tts/550e8400-e29b-41d4-a716-446655440000"
```

レスポンス:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": "Generating segment 2/5"
}
```

ステータス: `pending`, `processing`, `completed`, `failed`

#### 3. 完了したら音声をダウンロード

```bash
curl "http://localhost:8000/tts/550e8400-e29b-41d4-a716-446655440000/download" \
  --output output.wav
```

### 同期TTS生成（短いテキスト向け）

```bash
curl -X POST "http://localhost:8000/tts/sync" \
  -F "prompt_file=@reference.wav" \
  -F "prompt_text=これは参照音声の転写です" \
  -F "text=短いテキスト" \
  --output output.wav
```

### その他のエンドポイント

- `GET /health`: サーバーの健全性チェック
- `GET /models`: 利用可能なモデル一覧

## モデルディレクトリの設定

デフォルトでは`~/.sarashina_tts/pretrained_models`を使用します。以下の方法で変更できます。

### 環境変数で指定

```bash
export SARASHINA_MODEL_DIR=/path/to/models
open "Sarashina TTS.app"
```

### コマンドラインから指定（開発時）

```bash
python tts_launcher.py --model-dir /path/to/models
```

## ログ

ログファイルは`~/.sarashina_tts/logs/tts_launcher.log`に保存されます。メニューバーの「Show Logs」から直接開くこともできます。

## トラブルシューティング

### モデルが見つからないエラー

`~/.sarashina_tts/pretrained_models`にモデルファイルが正しく配置されているか確認してください。

```bash
ls ~/.sarashina_tts/pretrained_models/
```

### ポートが既に使用されている

ポート7860または8000が他のアプリで使用されている場合、起動に失敗します。使用中のプロセスを停止してください。

```bash
lsof -i :7860
lsof -i :8000
```

### .appが起動しない

セキュリティ設定により、不明な開発者からの.appがブロックされる場合があります。システム設定で許可してください。

## 開発者向け

### ソースから実行

```bash
python tts_launcher.py
```

### コードの変更後の再ビルド

コードを変更した場合は、再度PyInstallerでビルドしてください。

```bash
pyinstaller "Sarashina TTS.spec"
```

## ライセンス

このデスクトップアプリの実装は、オリジナルのSarashina2.2-TTSのライセンスに従います。
