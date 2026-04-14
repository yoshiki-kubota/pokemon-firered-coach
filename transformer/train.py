"""
Transformer（GPT）の学習スクリプト

文字レベルの言語モデルとして学習する。
キャラクター1文字 = 1トークン。シンプルだが原理は同じ。

実行方法:
  python transformer/train.py

学習データ:
  サンプルテキストをそのまま埋め込んでいるので、外部ファイル不要。
  もちろん自分のテキストファイルを読み込むことも可能。
"""

import time
import torch
from model import GPT, count_parameters


# ============================================================
# 1. 文字レベルトークナイザー
# ============================================================
#
# 本格的なLLMではBPE（Byte Pair Encoding）を使うが、
# 仕組みを理解するには文字レベルで十分。
#
# "hello" → [104, 101, 108, 108, 111]
# [104, 101, 108, 108, 111] → "hello"
# ============================================================


class CharTokenizer:
    """文字レベルのトークナイザー"""

    def __init__(self, text: str):
        # テキスト中の全ユニーク文字を集めてソート
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        # 文字 → ID、ID → 文字 の辞書を作る
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """テキスト → トークンIDリスト"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """トークンIDリスト → テキスト"""
        return "".join(self.id_to_char[i] for i in ids)


# ============================================================
# 2. データセット準備
# ============================================================
#
# 言語モデルの学習データの作り方:
#   テキスト: "hello world"
#   入力:     "hello worl"  (最後の1文字を除く)
#   正解:     "ello world"  (最初の1文字を除く)
#
# つまり「各位置で次の文字を予測する」タスクとして学習する。
# ============================================================


def create_batches(
    data: torch.Tensor, seq_len: int, batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """学習データからバッチを作成する"""
    batches = []
    # データからランダムな位置を取ってバッチを作る
    n_batches = len(data) // (seq_len * batch_size)
    for i in range(n_batches):
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            start = i * seq_len * batch_size + j * seq_len
            end = start + seq_len
            if end + 1 > len(data):
                break
            batch_x.append(data[start:end])
            batch_y.append(data[start + 1 : end + 1])
        if len(batch_x) == batch_size:
            batches.append((torch.stack(batch_x), torch.stack(batch_y)))
    return batches


# ============================================================
# 3. 学習テキスト
# ============================================================

TRAINING_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscoverd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slippered pantaloon,
With spectacles on nose and pouch on side;
His youthful hose, well saved, a world too wide
For his shrunk shank, and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest,
For Brutus is an honourable man;
So are they all, all honourable men,
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.
""".strip()


# ============================================================
# 4. 学習ループ
# ============================================================


def train():
    """モデルの学習を実行する"""

    print("=" * 60)
    print("  Transformer (GPT) を1から学習する")
    print("=" * 60)

    # --- ハイパーパラメータ ---
    # 小さい値にして、学習の流れを観察しやすくしている
    d_model = 128       # モデルの次元数
    n_heads = 4         # Attentionヘッド数（128 / 4 = 32次元/ヘッド）
    n_layers = 4        # Transformerブロック数
    max_seq_len = 128   # 最大系列長
    batch_size = 8      # バッチサイズ
    n_epochs = 50       # エポック数
    learning_rate = 3e-4  # 学習率（AdamWの推奨値）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")

    # --- トークナイザー ---
    tokenizer = CharTokenizer(TRAINING_TEXT)
    print(f"語彙サイズ: {tokenizer.vocab_size} 文字")
    print(f"語彙: {''.join(tokenizer.chars)}")

    # --- データ準備 ---
    data = torch.tensor(tokenizer.encode(TRAINING_TEXT), dtype=torch.long)
    print(f"学習データ長: {len(data):,} 文字")

    # 90%を学習データ、10%を検証データに
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_batches = create_batches(train_data, max_seq_len, batch_size)
    val_batches = create_batches(val_data, max_seq_len, batch_size)
    print(f"学習バッチ数: {len(train_batches)}")
    print(f"検証バッチ数: {len(val_batches)}")

    # --- モデル作成 ---
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    param_count = count_parameters(model)
    print(f"\nモデルパラメータ数: {param_count:,}")
    print(f"（参考: GPT-2 small = 117M, GPT-3 = 175B）")

    # --- オプティマイザ ---
    # AdamW: Adamに重み減衰（Weight Decay）を正しく適用したバージョン
    # GPT系モデルの標準的なオプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 学習開始 ---
    print(f"\n{'='*60}")
    print("  学習開始！")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # --- 学習 ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            logits, loss = model(batch_x, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # 勾配クリッピング（勾配爆発を防ぐ）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # --- 検証 ---
        model.eval()
        val_loss = 0.0
        val_batches_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_batches:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                _, loss = model(batch_x, batch_y)
                val_loss += loss.item()
                val_batches_count += 1

        avg_val_loss = val_loss / max(val_batches_count, 1)

        # --- ログ出力 ---
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"経過: {elapsed:.1f}秒"
            )

    elapsed = time.time() - start_time
    print(f"\n学習完了！ 合計時間: {elapsed:.1f}秒")

    # --- テキスト生成 ---
    print(f"\n{'='*60}")
    print("  テキスト生成")
    print(f"{'='*60}\n")

    model.eval()

    prompts = ["To be", "The ", "And ", "Friends"]
    for prompt in prompts:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        # temperature=0.8: やや創造的、top_k=40: 上位40文字から選択
        output_ids = model.generate(
            input_ids, max_new_tokens=200, temperature=0.8, top_k=40
        )
        generated = tokenizer.decode(output_ids[0].tolist())

        print(f"プロンプト: \"{prompt}\"")
        print(f"生成結果:   \"{generated}\"")
        print("-" * 60)

    # --- モデルの保存 ---
    save_path = "transformer/trained_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_chars": tokenizer.chars,
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "max_seq_len": max_seq_len,
            },
        },
        save_path,
    )
    print(f"\nモデルを保存しました: {save_path}")

    print(f"\n{'='*60}")
    print("  学習の解説")
    print(f"{'='*60}")
    print("""
【損失（Loss）の見方】
  - 初期値 ≈ ln(語彙サイズ) ≈ {:.2f} → ランダムに予測している状態
  - 学習が進むと損失が下がる → 次の文字をうまく予測できるようになった
  - Train Loss < Val Loss → 学習データは覚えたが汎化はまだ（過学習）

【なぜ文章が生成できるのか】
  1. モデルは「この文字の次に来やすい文字」の確率分布を学んだ
  2. 生成時は確率に従って1文字ずつサンプリングする
  3. これを繰り返すと文章になる
  4. GPT-4もChatGPTも、原理的にはこれと全く同じ仕組み！
    （規模が違うだけ: 数十文字 vs 数万トークン、数百パラメータ vs 数千億パラメータ）

【さらに学ぶには】
  - d_model, n_layers, n_heads を変えて実験してみる
  - もっと大きなテキストデータで学習する
  - BPEトークナイザーを実装する
  - 学習率スケジューラーを追加する
""".format(
        __import__("math").log(tokenizer.vocab_size)
    ))


if __name__ == "__main__":
    train()
