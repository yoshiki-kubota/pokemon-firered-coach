"""
Transformer（GPTアーキテクチャ）をゼロから実装する

このファイルでは、以下のコンポーネントを1つずつ積み上げてGPT風の
言語モデルを構築する:

  1. Multi-Head Self-Attention  — Transformerの心臓部
  2. Feed-Forward Network (FFN) — 各位置ごとの非線形変換
  3. Transformer Block           — Attention + FFN + 残差接続 + LayerNorm
  4. GPT (デコーダーonlyモデル)  — ブロックを積み重ねた言語モデル

「Attention Is All You Need」(Vaswani et al., 2017) の論文がベース。
ただしLLM（GPT系）はデコーダー部分だけを使う。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Multi-Head Self-Attention
# ============================================================
#
# Attentionの直感的な説明:
#   「文中の各単語が、他のどの単語に注目すべきか」を学習する仕組み。
#
#   例: "猫が魚を食べた" → "食べた"は"猫"と"魚"に強く注目する
#
# 数式:
#   Attention(Q, K, V) = softmax(QK^T / √d_k) V
#
#   Q (Query): 「何を探しているか」
#   K (Key):   「何を持っているか」
#   V (Value): 「実際の情報」
#
# Multi-Head にする理由:
#   1つのAttentionだと1種類の関係しか捉えられない。
#   複数のHeadで「文法的関係」「意味的関係」など異なる観点を同時に学習できる。
# ============================================================


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention の実装

    Parameters:
        d_model:  モデルの次元数（埋め込みベクトルのサイズ）
        n_heads:  Attentionヘッドの数
        dropout:  ドロップアウト率
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model は n_heads で割り切れる必要がある"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 各ヘッドの次元数

        # Q, K, V を作るための線形変換（重みは学習される）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 全ヘッドの出力を結合した後の線形変換
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    入力テンソル [batch_size, seq_len, d_model]
            mask: マスク [batch_size, 1, seq_len, seq_len] (Noneなら使わない)
        Returns:
            出力テンソル [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # --- Step 1: Q, K, V を計算 ---
        # 各 [batch_size, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # --- Step 2: Multi-Head に分割 ---
        # [batch_size, seq_len, d_model] → [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # --- Step 3: Scaled Dot-Product Attention ---
        # QK^T / √d_k  → [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果マスク（causal mask）: 未来のトークンを見えなくする
        # GPTでは「次の単語を予測する」ので、未来の情報は使えない
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmaxで確率分布に変換
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 重み付き和を計算 → [batch_size, n_heads, seq_len, d_k]
        context = torch.matmul(attention_weights, V)

        # --- Step 4: ヘッドを結合 ---
        # [batch_size, n_heads, seq_len, d_k] → [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # --- Step 5: 出力の線形変換 ---
        return self.W_o(context)


# ============================================================
# 2. Feed-Forward Network (FFN)
# ============================================================
#
# Attentionは「どの情報を集めるか」を決める。
# FFNは「集めた情報をどう変換するか」を決める。
#
# 構造: Linear → GELU → Linear
# 中間層は通常 4倍 に拡大する（d_model → 4*d_model → d_model）
#
# GELUはReLUの滑らかな版。GPT-2以降の標準。
# ============================================================


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 慣例: 4倍に拡大

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),               # 活性化関数（GPT-2以降の標準）
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 3. Transformer Block
# ============================================================
#
# 1つのTransformer Blockの流れ:
#
#   入力 → LayerNorm → Multi-Head Attention → + 残差接続
#                                                ↓
#                                    LayerNorm → FFN → + 残差接続 → 出力
#
# 残差接続（Residual Connection）:
#   入力をそのまま出力に足す。これにより:
#   - 勾配が消えにくくなる（深いネットワークでも学習できる）
#   - 「何も学ばない」= 恒等写像 をデフォルトにできる
#
# Pre-Norm（LayerNormを先にやる）を採用。GPT-2以降の標準。
# 元論文のPost-Normより学習が安定する。
# ============================================================


class TransformerBlock(nn.Module):
    """1つのTransformerブロック（Pre-Norm構成）"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Attention + 残差接続
        x = x + self.attn(self.ln1(x), mask)
        # FFN + 残差接続
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# 4. Positional Encoding（位置エンコーディング）
# ============================================================
#
# Transformerは再帰構造を持たないので、単語の「順番」の情報がない。
# 位置エンコーディングで「この単語は何番目か」を教える。
#
# 方法は2つ:
#   (a) 学習可能な位置埋め込み（GPT-2方式）← 今回はこっち
#   (b) sin/cos固定位置エンコーディング（元論文方式）
#
# 学習可能な方が簡単で、小規模モデルでは十分に機能する。
# ============================================================


# ============================================================
# 5. GPT（デコーダーonly Transformer）
# ============================================================
#
# GPTの全体構造:
#
#   トークン列 → トークン埋め込み + 位置埋め込み
#              → Transformer Block × N層
#              → LayerNorm
#              → 線形変換（語彙サイズへ射影）
#              → 次のトークンの確率分布
#
# 学習:
#   入力: [t1, t2, t3, t4]
#   正解: [t2, t3, t4, t5]  ← 1つずらしたもの
#   損失: Cross Entropy Loss
# ============================================================


class GPT(nn.Module):
    """
    GPT（Generative Pre-trained Transformer）

    Parameters:
        vocab_size:  語彙サイズ（何種類の文字/トークンがあるか）
        d_model:     モデルの次元数
        n_heads:     Attentionヘッド数
        n_layers:    Transformerブロックの積み重ね数
        max_seq_len: 最大系列長
        dropout:     ドロップアウト率
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # トークン埋め込み: 整数ID → d_model次元ベクトル
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置埋め込み: 位置（0, 1, 2, ...）→ d_model次元ベクトル
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer Block を n_layers 層積み重ねる
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # 最後の LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # 出力ヘッド: d_model → vocab_size（各トークンの確率を出す）
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: 入力埋め込みと出力ヘッドの重みを共有
        # パラメータ数を減らしつつ性能が上がるテクニック（GPT-2で使用）
        self.head.weight = self.token_embedding.weight

        # パラメータの初期化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """重みの初期化（GPT-2に準拠）"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        因果マスク（Causal Mask）を作る

        下三角行列:
          [[1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 0],
           [1, 1, 1, 1]]

        → 0の位置は -inf にして softmax で確率0にする
        → これで「未来のトークンを見れない」ようにする
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     入力トークンID [batch_size, seq_len]
            targets: 正解トークンID [batch_size, seq_len] (学習時のみ)
        Returns:
            logits: 各トークンの予測スコア [batch_size, seq_len, vocab_size]
            loss:   Cross Entropy Loss (targetsがある場合)
        """
        batch_size, seq_len = idx.shape
        device = idx.device

        assert seq_len <= self.max_seq_len, f"系列長 {seq_len} が最大 {self.max_seq_len} を超えている"

        # --- 位置のインデックスを作る [0, 1, 2, ..., seq_len-1] ---
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

        # --- 埋め込み ---
        tok_emb = self.token_embedding(idx)       # [batch, seq_len, d_model]
        pos_emb = self.position_embedding(positions)  # [1, seq_len, d_model]
        x = self.dropout(tok_emb + pos_emb)

        # --- 因果マスク ---
        mask = self._make_causal_mask(seq_len, device)

        # --- Transformerブロックを順番に通す ---
        for block in self.blocks:
            x = block(x, mask)

        # --- 最終LayerNorm ---
        x = self.ln_f(x)

        # --- 出力: 語彙サイズへ射影 ---
        logits = self.head(x)  # [batch, seq_len, vocab_size]

        # --- 損失計算 ---
        loss = None
        if targets is not None:
            # logits: [batch*seq_len, vocab_size], targets: [batch*seq_len]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        テキスト生成（自己回帰）

        仕組み:
          1. 現在の系列を入力
          2. 最後の位置の確率分布を取得
          3. そこからサンプリングして次のトークンを選ぶ
          4. 選んだトークンを系列に追加
          5. 繰り返し

        Args:
            idx:            初期トークン列 [batch_size, seq_len]
            max_new_tokens: 生成するトークン数
            temperature:    温度パラメータ（高い=多様、低い=確定的）
            top_k:          上位k個からのみサンプリング
        """
        for _ in range(max_new_tokens):
            # 最大系列長を超えないようにクロップ
            idx_cond = idx[:, -self.max_seq_len:]

            # forwardで予測
            logits, _ = self(idx_cond)

            # 最後の位置の logits だけ使う
            logits = logits[:, -1, :] / temperature

            # top-k フィルタリング
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # 確率分布に変換してサンプリング
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 系列に追加
            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ============================================================
# モデルのサイズを確認するユーティリティ
# ============================================================


def count_parameters(model: nn.Module) -> int:
    """学習可能なパラメータ数を返す"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 簡単なテスト
    print("=== Transformer (GPT) のテスト ===\n")

    vocab_size = 256  # 文字レベル
    model = GPT(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
    )

    print(f"パラメータ数: {count_parameters(model):,}")
    print(f"\nモデル構造:\n{model}")

    # ダミー入力でforward
    dummy_input = torch.randint(0, vocab_size, (2, 32))  # batch=2, seq_len=32
    dummy_target = torch.randint(0, vocab_size, (2, 32))

    logits, loss = model(dummy_input, dummy_target)
    print(f"\n入力形状:  {dummy_input.shape}")
    print(f"出力形状:  {logits.shape}")
    print(f"損失:      {loss.item():.4f}")
    print(f"\n期待される初期損失: {math.log(vocab_size):.4f} (= ln({vocab_size}))")
    print("（ランダム初期化なので、初期損失 ≈ ln(語彙サイズ) なら正常）")
