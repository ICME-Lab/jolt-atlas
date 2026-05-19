# QuantRotation

## Motivation

MLP の `down_proj` 出力には、最大で `-800` を超える massive outlier が存在する。
この outlier は出力テンソル `Y` の量子化 range を大きく広げ、量子化精度を著しく落とす。

そのため、単純に `Y` を clip するのではなく、量子化前に QuaRot や KurTail のような rotation を適用し、kurtosis を抑えた座標系で量子化する必要がある。

## Rotated Residual Stream

出力を `y` とすると、量子化前に同じ rotation `R` をかけて:

```text
yR = y'
```

とする。

residual stream 側もすでに:

```text
xR = x'
```

として持てているなら、同じ rotation 行列 `R` を使える限り、residual add は回転後の座標系でそのまま実行できる。

```text
z  = x + y
zR = xR + yR
z' = x' + y'
```

つまり:

```text
z' = x' + y'
```

として、回転された residual stream を保ったまま次の layer に渡せる。

## RMSNorm + Linear In Rotated Space

問題は、`z'` が次 layer の RMSNorm に渡されること。

回転された状態は kurtosis が低いという前提で考えているため、できれば回転したまま RMSNorm を計算したい。

直交行列 `R` ならノルムは保存されるので:

```text
RMS(xR) = RMS(x)
```

が成り立つ。

したがって、`x' = xR` とすると:

```text
RMS(x') = RMS(x)
```

である。

RMSNorm は通常:

```text
RMSNorm(x) = (x / RMS(x)) * Gamma
```

である。

ここで重要なのは、RMSNorm の出力そのものを回転空間で保持する必要はないという点。
RMSNorm の出力は通常すぐ次の Linear/MatMul に渡される。

元の計算が:

```text
RMSNorm(x) W
```

であるなら:

```text
x' = xR
x = x' R^-1
RMS(x) = RMS(x')
```

を使って:

```text
RMSNorm(x) W
  = (x / RMS(x)) Gamma W
  = (x' R^-1 / RMS(x')) Gamma W
  = (x' / RMS(x')) (R^-1 Gamma W)
```

と書ける。

したがって:

```text
W' = R^-1 Gamma W
```

を事前に作っておけば、実行時は:

```text
(x' / RMS(x')) W'
```

として計算できる。

つまり、residual stream を回転空間で持ったまま、RMSNorm + Linear を一つの回転済み weight に吸収できる。
`Gamma` は standalone の dense 行列として扱う必要はなく、次の dense weight `W` に吸収する。

## When RMSNorm Output Must Stay Rotated

もし RMSNorm の出力自体を再び rotation 後の座標系で持ちたいなら、話は変わる。

`x = x' R^-1` なので:

```text
RMSNorm(x)
  = (x' R^-1 / RMS(x')) * Gamma
```

この出力を再び rotation 後の座標系で持ちたいなら:

```text
RMSNorm(x) R
  = ((x' R^-1) / RMS(x')) * Gamma * R
```

となる。

この出力をさらに rotation 後の座標系で持ちたいなら:

```text
RMSNorm(x) R
  = ((x' R^-1) / RMS(x')) * Gamma * R
  = (x' / RMS(x')) * (R^-1 Gamma R)
```

となる。

ここで:

```text
Gamma' = R^-1 Gamma R
```

とすれば:

```text
RMSNorm(x) R = (x' / RMS(x')) * Gamma'
```

となる。

ただし `Gamma'` は一般に diagonal ではなく dense になる可能性がある。
これは RMSNorm 出力そのものを回転空間で保持したい場合だけの caveat。

通常の Transformer block では、RMSNorm 出力は次の Linear に入るため、上の `W' = R^-1 Gamma W` として吸収する方が自然。

## Important Caveat

単純に:

```text
RMSNorm(x) = (x' / RMS(x')) * R^-1 Gamma
```

または:

```text
RMSNorm(x) = RMSNorm(x')
```

とみなすには注意が必要。

`Gamma` は channel-wise の対角重みであり、一般には rotation `R` と可換ではない。

```text
Gamma R != R Gamma
```

そのため、RMSNorm を完全に回転座標系のまま処理するには、`Gamma` を含めた変換を正しく扱う必要がある。
ただし、RMSNorm + Linear として扱えるなら `Gamma` は `W' = R^-1 Gamma W` に吸収できる。

## Open Questions

- `down_proj` 出力 `y` と residual `x` に同じ rotation `R` を使えるか。
- layer 間で同じ `R` を保ち続けられるか。
- RMSNorm + Linear の weight として `W' = R^-1 Gamma W` を事前計算できるか。
- RMSNorm 出力自体を回転空間で保持する必要があるケースがあるか。
- QuaRot の固定 Hadamard rotation で十分か、KurTail のように kurtosis を直接下げる rotation を探索すべきか。
- `down_proj` の massive outlier を抑える rotation と、`gate/up` の activation を良くする rotation が同じでよいか。

## Current Hypothesis

現時点の仮説は:

```text
1. down_proj の Y outlier は clip してはいけない。
2. Y は rotation された座標系で量子化するべき。
3. residual stream も同じ rotation 座標系で持てるなら、residual add は自然に処理できる。
4. RMSNorm は RMS 自体は rotation invariant であり、RMSNorm + Linear は `W' = R^-1 Gamma W` に吸収できる。
5. KurTail 的に kurtosis を下げる rotation を選べば、W8A8Y8 の PPL をさらに改善できる可能性がある。
```
