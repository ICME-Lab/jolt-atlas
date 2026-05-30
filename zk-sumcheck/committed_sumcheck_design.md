# Pedersen commitment による ZK sumcheck メモ

このメモは、sumcheck の round polynomial を Pedersen commitment で隠し、途中 round の claim を scalar として漏らさずに検証する設計を整理したもの。

## 基本形

degree `d` の round polynomial を次のように置く。

```text
g_i(x) = c_{i,0} + c_{i,1}x + ... + c_{i,d}x^d
```

通常の sumcheck では verifier は次を確認する。

```text
g_i(0) + g_i(1) = previous_claim
g_i(r_i)        = next_claim
```

係数で書くと、

```text
g_i(0) + g_i(1)
= 2c_{i,0} + c_{i,1} + ... + c_{i,d}

g_i(r_i)
= c_{i,0} + r_i c_{i,1} + ... + r_i^d c_{i,d}
```

各係数に Pedersen commitment を作る。

```text
C_{i,j} = c_{i,j}G + rho_{i,j}H
```

すると、round 間の関係は commitment 上で次のように確認できる。

```text
sum_j r_i^j C_{i,j}
==
2C_{i+1,0} + C_{i+1,1} + ... + C_{i+1,d}
```

ただし、このままだと blinding randomness 側も一致している必要がある。係数 commitment の randomness に線形制約を入れたくないので、後述する `beta H` slack を使う。

## 端だけを開く

この方式で scalar として開く必要があるのは基本的に端だけ。

```text
g_0(0) + g_0(1)
g_k(r_k)
```

途中 round の値 `g_i(r_i)` は scalar として開かず、commitment の線形関係だけでつなぐ。

端の opening は、今回の設計では Schnorr opening proof で足りる。つまり、値は開くが、blinding randomness は開かない。

理由は、各係数 commitment が同じ message generator `G` を使っているため、verifier が次の scalar commitment を自分で作れるから。

```text
C_eval = sum_i r^i C_i
       = (sum_i r^i c_i)G + (sum_i r^i rho_i)H
```

公開したい値を `v = sum_i r^i c_i` とすると、verifier は次を計算できる。

```text
D = C_eval - vG
```

prover は次を Schnorr proof で示せばよい。

```text
I know rho_eval such that D = rho_eval H
```

ここで `rho_eval = sum_i r^i rho_i` は明かさない。

Bulletproofs / IPA が必要になるのは、係数ベクトル全体を1つの vector commitment にして、その内積値だけを開きたい場合である。この crate の現在の設計では、係数ごとに scalar Pedersen commitment を作るので、端の opening は Schnorr proof でよい。

## beta H slack

係数 commitment の randomness を独立に保つため、transition ごとに zero-value commitment を追加する。

```text
B_i = beta_i H
```

`B_i` は値 `0` への Pedersen commitment であり、次を Schnorr proof で示す。

```text
I know beta_i such that B_i = beta_i H
```

round transition は次のように検証する。

```text
sum_j r_i^j C_{i,j}
==
2C_{i+1,0} + C_{i+1,1} + ... + C_{i+1,d} + B_i
```

ここで、

```text
beta_i
= (sum_j r_i^j rho_{i,j})
  - (2rho_{i+1,0} + rho_{i+1,1} + ... + rho_{i+1,d})
```

こうすると `rho_{i+1,0}, ..., rho_{i+1,d}` は全て独立ランダムに選べる。線形制約は `beta_i` が吸収する。

## zero commitment proof

`B = beta H` が値 `0` への commitment であることは Schnorr proof で示せる。

公開値:

```text
B, H
```

秘密値:

```text
beta
```

証明したい関係:

```text
B = beta H
```

prover:

```text
alpha <- random
T = alpha H
e = Hash(transcript, B, H, T)
z = alpha + e beta
```

verifier:

```text
e = Hash(transcript, B, H, T)
check: zH == T + eB
```

この proof では `beta` は明かさない。

## fan-out

親 claim が複数の子 claim に分岐する場合を考える。

```text
G = A + B
```

commitment は本来、

```text
C_G == C_A + C_B
```

にしたいが、blinding が一致するとは限らない。そこで slack を入れる。

```text
C_G == C_A + C_B + S
S = beta H
```

ここで、

```text
beta = rho_G - rho_A - rho_B
```

`S` に zero commitment proof を付ければ、値側では `G = A + B` を保ちつつ、`rho_A`, `rho_B` は独立ランダムにできる。

一般の `m` 分岐でも同じ。

```text
C_G == C_A1 + ... + C_Am + S
S = beta H
beta = rho_G - sum_i rho_Ai
```

## fan-in

複数の前段 claim が一つの claim に合流する場合も、基本は commitment の加算で扱える。

```text
G = A + B
C_G = C_A + C_B + S
```

ここでも slack を入れることで、各 commitment の blinding に不要な制約を入れずに済む。

注意すべきなのは、同じ commitment が複数の独立な線形制約に参加する場合。制約が重なると、blinding randomness の自由度が減る可能性がある。

## 自由度と leakage

slack を使わずに、例えば degree 3 の次 round randomness を

```text
2rho_0 + rho_1 + rho_2 + rho_3 = target
```

で縛ると、自由度は 4 から 3 に落ちる。1本だけなら通常は直ちに係数が漏れるわけではないが、同じ commitment に複数の制約が重なると危険になる。

特に verifier が次のような blinding-free な線形結合を作れると危ない。

```text
sum_i alpha_i C_i = (sum_i alpha_i m_i)G
```

値が小さい範囲にある場合、離散対数を総当たりできる可能性がある。

`beta H` slack を使うと、係数 commitment 側の randomness は独立に保てる。制約は slack 側に集約される。

## beta H は beta を漏らすか

古典的な離散対数仮定の下では、`B = beta H` から `beta` は求まらない。Schnorr proof でも `beta` は明かさない。

ただし、以下は避ける必要がある。

```text
- beta を plain open しない
- Schnorr nonce を再利用しない
- 同じ slack を複数 relation に使い回さない
- transcript に relation id を入れる
```

量子計算機で離散対数が解ける設定では、`beta` は漏れる。この場合 Pedersen commitment の binding は壊れるし、commitment と slack から線形方程式系を作れる。単体で全 witness が即漏れるとは限らないが、小さい値域の witness では追加漏洩のリスクが高くなる。

## KZG commitment との関係

現在この repository で使っている HyperKZG commitment には hiding は入っていない。

実装上は、ランダムな blinding polynomial を足しておらず、単に polynomial を commit している。

```text
binding: ある
hiding: ない
```

したがって、今の HyperKZG は opening の binding 用であり、witness privacy 用ではない。ZK 性を持たせるには、KZG に hiding を追加するか、Pedersen/IPA 系の committed sumcheck と組み合わせる必要がある。

## GKR 的な claim reduction の toy example

次のベクトル関係を考える。

```text
A * B = E
C * D = F
E * F = G
```

`A, B, C, D, G` は verifier が直接評価できるとする。本来は PCS opening で評価するが、ここでは単純化する。

まず `G(R)` を claim として、`E * F = G` の sumcheck を始める。

最初の round polynomial `g_0` の係数を Pedersen commit し、

```text
2C_{0,0} + C_{0,1} + ... + C_{0,d}
```

を `G(R)` に Schnorr opening proof で open する。

途中 round は、

```text
sum_j r_i^j C_{i,j}
==
2C_{i+1,0} + C_{i+1,1} + ... + C_{i+1,d} + B_i
```

でつなぐ。

最終的に、

```text
C_T = sum_j r_k^j C_{k,j}
```

が得られる。これは概念的には、

```text
T = eq(R,z) * E(z) * F(z)
```

への commitment。

## 乗法分岐を Sigma protocol で隠す方式

中間値 `E(z)` と `F(z)` を開きたくない場合は、新しく child claim commitment を作る。

```text
C_E = Com(E(z); rho_E)
C_F = Com(F(z); rho_F)
```

親 sumcheck の最後で得た commitment を

```text
C_T = Com(T; rho_T)
T = q * E(z) * F(z)
q = eq(R,z)
```

とする。

このとき、`C_T`, `C_E`, `C_F` の整合性は Pedersen committed multiplication の Sigma protocol で証明できる。

`F = F(z)` を witness scalar として使うと、次の2本の表現を同時に証明すればよい。

```text
C_F = F G + rho_F H
C_T = F * (q C_E) + gamma H
```

ここで、

```text
gamma = rho_T - F * q * rho_E
```

なぜなら、

```text
q C_E = qE G + q rho_E H

F * (q C_E) + gamma H
= qEF G + (F q rho_E + gamma)H
= qEF G + rho_T H
= C_T
```

だから。

この Sigma protocol では、公開値は:

```text
C_E, C_F, C_T, G, H, q
```

秘密値は:

```text
F, rho_F, gamma
```

prover:

```text
a_F, a_rho, a_gamma <- random

T1 = a_F G + a_rho H
T2 = a_F (q C_E) + a_gamma H

e = Hash(transcript, public, T1, T2)

z_F     = a_F     + eF
z_rho   = a_rho   + e rho_F
z_gamma = a_gamma + e gamma
```

verifier:

```text
z_F G + z_rho H
==
T1 + e C_F

z_F (q C_E) + z_gamma H
==
T2 + e C_T
```

この proof により、`E(z)` も `F(z)` も scalar として開かないまま、親の product claim から2つの child claim commitment に分岐できる。

その後は:

```text
C_E:
  A * B = E の sumcheck の initial claim commitment として渡す。

C_F:
  C * D = F の sumcheck の initial claim commitment として渡す。
```

各 child sumcheck は、それぞれの最初の round polynomial 係数 commitment に対して、

```text
C_E == 2C_{E,0} + C_{E,1} + ... + C_{E,d} + B_E
C_F == 2C_{F,0} + C_{F,1} + ... + C_{F,d} + B_F
```

を `beta H` slack 付きでつなぐ。

この流れでは:

```text
- 親 sumcheck の最後を open しない
- E(z), F(z) も open しない
- 乗法分岐だけ Sigma multiplication proof で処理する
- 以降は committed round-polynomial sumcheck に戻る
```

つまり、DAG の internal edge は commitment のまま流し、terminal edge だけ Schnorr opening proof で verifier-known value に接続する。

Sigma protocol のコストは軽い。概算では:

```text
proof size:
  group element 2個 + scalar 3個

prover:
  4 scalar mul + 2 point add + hash

verifier:
  6 scalar mul + 4 point add + hash
```

これは Schnorr opening proof より少し重いが、IPA/Bulletproof よりかなり軽い。GKR の claim reduction では要素ごとではなく分岐 claim ごとに使うので、op 単位なら現実的に見える。

## 片側を開く簡易方式

上の Sigma protocol を使わず、片側を advice として開く簡易案もある。

ここで `E(z)` を advice として scalar で公開する。

すると verifier は、

```text
C_F = (eq(R,z) * E(z))^{-1} * C_T
```

を計算できる。これは `F(z)` への commitment になる。

したがって次の分岐は非対称になる。

```text
E(z):
  scalar advice として公開した値。
  A * B = E の sumcheck を始めるには、
  最初の round sum commitment を E(z) に Schnorr opening proof で open する必要がある。

F(z):
  C_F = Com(F(z)) がすでにある。
  C * D = F の sumcheck の initial claim commitment としてそのまま渡せる。
  F(z) を open する必要はない。
```

この方式では、乗法分岐を Pedersen の線形性だけで完全に処理しているわけではない。片側 `E(z)` を開くことで、もう片側 `F(z)` の commitment を導出している。

## E(z) を開くことの意味

`E(z)` は中間 wire `E` のランダム点評価であり、厳密な zero-knowledge の観点では漏洩である。

ただし `E` が他の計算で再利用されず、漏れる評価が1回だけなら、漏洩は field element 1個分に限られる。

```text
E(z) = sum_i eq(z,i) E_i
```

攻撃面は主に次に依存する。

```text
- E の要素数
- E_i の値域
- E のエントロピー
- E に既知構造があるか
- 同じ E に対して複数点評価が漏れるか
```

`E` が大きく、1回しか使われず、値域も十分広いなら、実用上の漏洩は限定的になる。一方で、値域が小さい場合や同じ中間 wire の評価が何度も漏れる場合は、統計的・総当たり的な攻撃ベクトルが残る。

まとめると、この toy example では:

```text
厳密ZK:
  E(z) の開示は漏洩。

実用的な見方:
  E が一度しか使われないなら、主なリスクは
  E の値域と要素数に依存する統計的 brute force。
```
