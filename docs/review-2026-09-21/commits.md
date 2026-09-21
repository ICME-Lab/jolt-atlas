# Commit review map

Commits are listed in dependency order within each review unit. Source PR identities and deliberate splits are in [sources.json](sources.json).

## 1. tensor-correctness

- [1e29a4f3](https://github.com/ICME-Lab/jolt-atlas/commit/1e29a4f3e32c385c738141825b57c4a69c239765) Restore headroom for additive attention masks
- [26ecdf20](https://github.com/ICME-Lab/jolt-atlas/commit/26ecdf2006fe58a4675790a6dbb0dc3b134b25ed) Match the workspace edition in mask regression formatting
- [7e7a36e0](https://github.com/ICME-Lab/jolt-atlas/commit/7e7a36e00591e0fce4898dc3048101d27264ba79) chore: fix clippy
- [fc8f8390](https://github.com/ICME-Lab/jolt-atlas/commit/fc8f839014fe4867d597e10737c265195a9f0390) Preserve logical element order across padded reshapes
- [f54ecd2d](https://github.com/ICME-Lab/jolt-atlas/commit/f54ecd2d303ea2000a52827e4e3c4bd34fe1826b) Avoid nested worker pools in BN254 G1 commitments
- [bc055cf7](https://github.com/ICME-Lab/jolt-atlas/commit/bc055cf7f8e7c0b0002a44c9cedcff148e2f5a46) fix: handle overflow in softmax centering
- [f0eff454](https://github.com/ICME-Lab/jolt-atlas/commit/f0eff454c0e3050e89db73ed7c0afbf7f511bb43) Preserve raw reshape indices and masks in both ordinary shadows
- [9e1be16e](https://github.com/ICME-Lab/jolt-atlas/commit/9e1be16ebc56fc3263244bfe1a30d9fa78df6163) Correct activation table constant references in ZK tests
- [0b28a6ba](https://github.com/ICME-Lab/jolt-atlas/commit/0b28a6baff0ee3bf499726c0d3e63d703f3b007b) Fix strict lint diagnostics in reshape and ZK opening tests

## 2. ordinary-verifier

- [958d0f9b](https://github.com/ICME-Lab/jolt-atlas/commit/958d0f9b91f102319d1f90e4f02247daf032cbce) Make ONNX import and parallel execution optional for verifier consumers
- [4e8e6d5f](https://github.com/ICME-Lab/jolt-atlas/commit/4e8e6d5faef1a5d3cf3696d1063ff4754228f94b) Document serial feature configuration and update lockfile
- [59e1864d](https://github.com/ICME-Lab/jolt-atlas/commit/59e1864d7f95d91944bb0717fe4e27df81f152df) Combine HyperKZG commitments with one batched MSM
- [6383dc09](https://github.com/ICME-Lab/jolt-atlas/commit/6383dc094d23ad716cfe8388271c2e8d0e767662) Import curve conversion trait in commitment regression
- [04e9efbd](https://github.com/ICME-Lab/jolt-atlas/commit/04e9efbd859fedf935ee8de4ea1c9070132c18bf) Evaluate univariate polynomials with Horner rule
- [bb1f630e](https://github.com/ICME-Lab/jolt-atlas/commit/bb1f630eac28be7fcdd2be4781002f9984e2c3f2) Evaluate unpadded reshape selectors without materializing tensors
- [9565781c](https://github.com/ICME-Lab/jolt-atlas/commit/9565781c74550ca6d6cb050e00649018c7c68b82) Evaluate aligned slice selectors without tensor tables
- [25d60e10](https://github.com/ICME-Lab/jolt-atlas/commit/25d60e10a11350aa0ce83742e986e16aa4c9f85a) Evaluate aligned concat selectors without tensor tables
- [c2f11016](https://github.com/ICME-Lab/jolt-atlas/commit/c2f1101682f0d72713fdb05759ca01be8febd863) Reuse equality weights when verifying softmax max indicators
- [e1369d63](https://github.com/ICME-Lab/jolt-atlas/commit/e1369d63146e6fdce4cb59eccbd861d9cf054060) Interpolate linear polynomials directly
- [5459b140](https://github.com/ICME-Lab/jolt-atlas/commit/5459b140961eb737ceed416b49ceb824493dfac2) Evaluate equality factors with one field product
- [2c73d4e4](https://github.com/ICME-Lab/jolt-atlas/commit/2c73d4e49c326b79487381f2ce95a9571410d7a3) Build clamp buckets without scanning constant weights
- [bd767e46](https://github.com/ICME-Lab/jolt-atlas/commit/bd767e46cc6e7d476664156ea938c08f4552258f) Share activation tables and evaluations across verifier instances
- [1d83af2a](https://github.com/ICME-Lab/jolt-atlas/commit/1d83af2ad4accd71583dce40ddce0144a4817237) Evaluate reduction interpolation with a shared Lagrange basis
- [e65a4d0a](https://github.com/ICME-Lab/jolt-atlas/commit/e65a4d0a64716d7c27335fbcd1f6fdfe4558d341) Check reduction values at zero and one directly
- [b8a7f72e](https://github.com/ICME-Lab/jolt-atlas/commit/b8a7f72e62fd8e73300308e60a34758e13cc6c0b) Embed validated fixed activation tables behind an optional feature
- [247eb106](https://github.com/ICME-Lab/jolt-atlas/commit/247eb1065769630831cc70740f6594ef38762bcf) Order fixed table module declarations for rustfmt
- [8bf537c1](https://github.com/ICME-Lab/jolt-atlas/commit/8bf537c139a4dc9b6a370e8e5760406f0686333f) Evaluate broadcast selectors directly
- [d2196757](https://github.com/ICME-Lab/jolt-atlas/commit/d219675779510648fdc9f5caef45c0a2755a7792) Reuse softmax tables and derive commitment sizes directly
- [3184f6ab](https://github.com/ICME-Lab/jolt-atlas/commit/3184f6ab6c16107371cb77b6298e588baf090a16) Match CI formatting for the softmax cache field
- [ce917126](https://github.com/ICME-Lab/jolt-atlas/commit/ce917126d0ddd001d7a1399a5a36efe2c9a0a417) Add optional affine MSM for serial BN254 verification
- [3106b943](https://github.com/ICME-Lab/jolt-atlas/commit/3106b9435b8989c1bd0c1cad29362039c3f317ab) Add compact softmax advice encoding with bounded direct decoding
- [d6b152eb](https://github.com/ICME-Lab/jolt-atlas/commit/d6b152ebb6b6595a763541e95f83c5e6f84e27c7) Pack repeated proof identifiers and reduction coefficients
- [22dee10f](https://github.com/ICME-Lab/jolt-atlas/commit/22dee10f4a633de0f04651697847f58b6f4c5829) Enforce serial zip lengths and gate ONNX dependent test fixtures
- [d1503200](https://github.com/ICME-Lab/jolt-atlas/commit/d1503200ed1bc3b43e50c92605ce7e1d12220177) Route round two proof helpers through the serial iterator facade
- [168e8c38](https://github.com/ICME-Lab/jolt-atlas/commit/168e8c38ec0926929c08912e43a5ce4a32fe408d) Format the consolidated sources with Rust 1.95
- [cea71000](https://github.com/ICME-Lab/jolt-atlas/commit/cea710000d1bb752c4166941a6f5b8e8703fc0cd) Add a complete proof fixture for verifier feature parity
- [f015b0f5](https://github.com/ICME-Lab/jolt-atlas/commit/f015b0f55841ead02adbc3070fc5ea83ac4069c9) Format serial helpers and the complete parity fixture
- [d6d57db1](https://github.com/ICME-Lab/jolt-atlas/commit/d6d57db10409b66c70a57e5b91cedd455fabea76) Complete serial iterators for round-two tensor and lookup helpers
- [e93028dc](https://github.com/ICME-Lab/jolt-atlas/commit/e93028dcd67356a4864a47f5bfca6357f4b18e1b) Keep tensor iterator implementations distinct in serial builds
- [9bfdc765](https://github.com/ICME-Lab/jolt-atlas/commit/9bfdc765e6e03a00a88cd1d9561c3c3e9ea24f7b) Document serial iterator type and use checked alignment idioms
- [d190e141](https://github.com/ICME-Lab/jolt-atlas/commit/d190e1419f5f1231818e69253cab5b7dda3d7165) Gate ONNX-only test imports while retaining manual ZK fixtures
- [bad074e2](https://github.com/ICME-Lab/jolt-atlas/commit/bad074e20a8d6fa6e24decee663e75cfe9a87f7f) Import the default MSM trait only for the selected verifier backend
- [b70711b0](https://github.com/ICME-Lab/jolt-atlas/commit/b70711b0499dec29badc4d70152c2521b86ed91f) Apply the same commitment length contract to both verifier backends

## 3. native-foundations

- [bb3f5562](https://github.com/ICME-Lab/jolt-atlas/commit/bb3f5562e67280c555ecde14a0ba79da3de02840) Add native hiding openings and derived BlindFold constraints
- [f96259f5](https://github.com/ICME-Lab/jolt-atlas/commit/f96259f5b5047322e462c8df95bfb6ef81e41946) Bind private table relations to hiding tensor commitments
- [1ed5a99f](https://github.com/ICME-Lab/jolt-atlas/commit/1ed5a99f1ea21e2dc71621653ffe60bb212bd39a) Prove exact integer multiplication with rescaling and clamping
- [05020c67](https://github.com/ICME-Lab/jolt-atlas/commit/05020c67e479591e6c715eb5f697fa4d88c47be9) Share hidden tensor commitments across registered graph proofs
- [6b5ad2a0](https://github.com/ICME-Lab/jolt-atlas/commit/6b5ad2a0ee7aff09296feb26ab687557576e4edd) Register public parameters for reusable native graph proofs
- [50fb79ad](https://github.com/ICME-Lab/jolt-atlas/commit/50fb79ad3ab98f37449c1a4f13fa7eb5bddcb64b) Open original hidden graph tensors at external boundaries
- [ae9f00dd](https://github.com/ICME-Lab/jolt-atlas/commit/ae9f00ddd1ee950cf9527708d347ddcca785371d) Keep normalization example with the tensor operator extension
- [a521a4eb](https://github.com/ICME-Lab/jolt-atlas/commit/a521a4eb818e6ba4e793f97ff306b8f7240ee191) Check vector boundary identifiers and setup capacity
- [db58782d](https://github.com/ICME-Lab/jolt-atlas/commit/db58782d75851d78d22e659365312489dd91cae1) Keep witness construction controls with the later uncommitted API
- [85ec9141](https://github.com/ICME-Lab/jolt-atlas/commit/85ec91413cdf7682af88f0af7977cd6a9a5695b0) Read the registered Dory verifier setup capacity
- [c678331a](https://github.com/ICME-Lab/jolt-atlas/commit/c678331a9a218ba42de095a66122edbb7e05481c) Format the consolidated sources with Rust 1.95
- [c5600ad6](https://github.com/ICME-Lab/jolt-atlas/commit/c5600ad637b88aa6db09a08ac1a7d35994ff0b3f) Remove unused experimental cache helpers and avoid test input copies
- [0a7de79f](https://github.com/ICME-Lab/jolt-atlas/commit/0a7de79f358c86ddad43c21cc2431086e82296f3) Format the experimental dispatcher test cleanup

## 4. native-operators

- [b1bfcb9d](https://github.com/ICME-Lab/jolt-atlas/commit/b1bfcb9d177e62dc98e929529554cf5d2835ac3b) Prove clamped addition and subtraction in native graphs
- [e96c3a6a](https://github.com/ICME-Lab/jolt-atlas/commit/e96c3a6a9b2b44b12145880ac8b2fcfdce941282) Prove exact tensor sums and means of squares
- [decfcdb2](https://github.com/ICME-Lab/jolt-atlas/commit/decfcdb2f25ce0ed0b420bf38280f51966f23bd6) Prove exact matrix and tensor contractions with bounded operands
- [1ecf5010](https://github.com/ICME-Lab/jolt-atlas/commit/1ecf5010869ea44c46a1216cea004f465d04cc16) Prove signed clamping before private table lookups
- [014d050b](https://github.com/ICME-Lab/jolt-atlas/commit/014d050b58c93479d4d17caab8e30e1686b7e1ac) Prove exact native reciprocal square roots
- [931619e9](https://github.com/ICME-Lab/jolt-atlas/commit/931619e990b6e7a130a34d2a87521c2e61f7bab3) Prove hidden tensor broadcasts and layout changes
- [e9111fe8](https://github.com/ICME-Lab/jolt-atlas/commit/e9111fe8efa0faa70cf82591bd54fa9d7cd49624) Prove exact native integer softmax
- [487ccd92](https://github.com/ICME-Lab/jolt-atlas/commit/487ccd92d438bfcbadec2b30028ab590efb2aa81) Prove native hidden slices and concatenation
- [f6ef897d](https://github.com/ICME-Lab/jolt-atlas/commit/f6ef897dca3d11aa01801b2788b499c6e8d5cc58) Handle public Boolean coordinates in opening reduction
- [9584d946](https://github.com/ICME-Lab/jolt-atlas/commit/9584d946d94af7c3d185c309be9d28658e01980e) Preserve logical divisors in native means of squares
- [8478b74a](https://github.com/ICME-Lab/jolt-atlas/commit/8478b74a5d699c0593e6655e92e8b1b16d57bf44) Extend registration and boundaries to tensor shapes and restore normalization coverage
- [3a861f4b](https://github.com/ICME-Lab/jolt-atlas/commit/3a861f4bfe4390f9091d3c57c12e72f928d7f456) Format the consolidated sources with Rust 1.95
- [525ee69b](https://github.com/ICME-Lab/jolt-atlas/commit/525ee69b3d5bc02bad97878dab9256a6a692bb8a) Keep uncommitted-witness control with the constructor introduced next
- [fb1b116c](https://github.com/ICME-Lab/jolt-atlas/commit/fb1b116c00b97e0a6855bb8517e4f93b33561bb4) Clarify contraction fixtures for strict lint checks

## 5. native-generation

- [dd33e67e](https://github.com/ICME-Lab/jolt-atlas/commit/dd33e67e23e13e5c94d744028c871f8ed530a68a) Prove lookups into original committed tensor tables
- [a0f7d234](https://github.com/ICME-Lab/jolt-atlas/commit/a0f7d2349d997f48fcb7060d4080147f42024fc6) Prove exact integer division and periodic tables
- [df3f32d2](https://github.com/ICME-Lab/jolt-atlas/commit/df3f32d2e64e74d9964921af1b6db0ec72e6cc44) Project columns before proving committed row gathers
- [59b0b6fa](https://github.com/ICME-Lab/jolt-atlas/commit/59b0b6fac26ff8a405db48f3b6c327ffe1cda85b) Lower integer masks and selection into native graph proofs
- [db23eafd](https://github.com/ICME-Lab/jolt-atlas/commit/db23eafda9fcd0d87c6e5b6201bc3afb60e8c279) Validate native integer witnesses and exact softmax centering
- [fdf764a3](https://github.com/ICME-Lab/jolt-atlas/commit/fdf764a3a9bf7965a83f25275d42b6254b99c5b1) Prove the first maximum index over a logical tensor prefix
- [56e64534](https://github.com/ICME-Lab/jolt-atlas/commit/56e6453458229347e3b05d7e4afe0388a4570216) Prove shifted greedy selection and stopping over hidden tensors
- [02163ea6](https://github.com/ICME-Lab/jolt-atlas/commit/02163ea6302999fc84fb49468c4a064ad3f4466d) Reject uncommitted graph witnesses at external tensor boundaries

## 6. native-prover

- [908d1d39](https://github.com/ICME-Lab/jolt-atlas/commit/908d1d3953fde9a7936c77d5867bdedc16893a49) Share fresh commitments for repeated registered range indicators
- [9ac4f51b](https://github.com/ICME-Lab/jolt-atlas/commit/9ac4f51b942b39f5ad68e3bcfdc2845d4785b103) Compute one-hot commitments in bounded row segments
- [30367557](https://github.com/ICME-Lab/jolt-atlas/commit/303675579375d8780d4ec6bd47fffff5893cf06c) Recover frequent lookup rows from public generator sums
- [27f0dff2](https://github.com/ICME-Lab/jolt-atlas/commit/27f0dff250847385308298d934020a71d236fecc) Reduce native range table memory with tiled histograms
- [f54a6a2d](https://github.com/ICME-Lab/jolt-atlas/commit/f54a6a2d5acde7a9e09d134dcc8d874f01a83a43) Compute independent ZK sumcheck messages in parallel
- [36fe4a4c](https://github.com/ICME-Lab/jolt-atlas/commit/36fe4a4cb70a5f120bfb3cc79ac15a8c743c6cb1) Evaluate sparse indicators with tiled equality weights
- [31e3b796](https://github.com/ICME-Lab/jolt-atlas/commit/31e3b79655cd2a1b96a5f1619ffd8f5a730f233f) Construct native contraction tables with small index maps
- [abb920d7](https://github.com/ICME-Lab/jolt-atlas/commit/abb920d752e38e8b5175ab824b03e12387989bc1) Release consumed native prover buffers before later stages
- [e4eafb2c](https://github.com/ICME-Lab/jolt-atlas/commit/e4eafb2c887bb4714e84e085d46de0cec748c2bb) Commit native polynomials in bounded parallel batches
- [891fd9a0](https://github.com/ICME-Lab/jolt-atlas/commit/891fd9a077a95ff5e9334aa501f8b3f8cb83eb3c) Build dense joint polynomials from overlapping prefixes
- [6e7460a8](https://github.com/ICME-Lab/jolt-atlas/commit/6e7460a8aeb2af7fbf43b1c9b88d9e22f6907c82) Combine row commitments from active hint prefixes
- [4aede59c](https://github.com/ICME-Lab/jolt-atlas/commit/4aede59c1eead0441add38cda4ce00df3b5b2feb) Share immutable rows across cloned Dory hints
- [90b2f549](https://github.com/ICME-Lab/jolt-atlas/commit/90b2f5495dbbe2654be965256c3f747e22a07bd4) Release finalized opening tables before PCS allocation
- [ffec0edd](https://github.com/ICME-Lab/jolt-atlas/commit/ffec0edd04ae72801a745f75bf4f01e8003f1640) Evaluate compact polynomial rows without nested scheduling
- [9c5efe79](https://github.com/ICME-Lab/jolt-atlas/commit/9c5efe7969dc2f3d1a97cb9c94fa5814972a16d5) Evaluate joint dense values without a full vector copy
- [c3e85d58](https://github.com/ICME-Lab/jolt-atlas/commit/c3e85d5808cc5d5c45da01952eb1ccc3b67dd7de) Release dense polynomial inputs after joint construction
- [382b98f8](https://github.com/ICME-Lab/jolt-atlas/commit/382b98f8b6a1cfb6da3ea3ec2cca311650751df1) Bound temporary tables during lookup preparation
- [7a8f87d8](https://github.com/ICME-Lab/jolt-atlas/commit/7a8f87d8dbe121002556387cc03a763e47b01f39) Reclaim bound working polynomial buffers during opening reduction
- [cbe6dad9](https://github.com/ICME-Lab/jolt-atlas/commit/cbe6dad9d0e91a13d0e8dea3c6d80754bbcbf537) Store immutable lookup indices in compact shared arrays
- [cb91027e](https://github.com/ICME-Lab/jolt-atlas/commit/cb91027e9b434a9886998befb9facc5821708d41) Use narrower addresses for fully populated lookup vectors
- [0d2804d3](https://github.com/ICME-Lab/jolt-atlas/commit/0d2804d3cdb037830faff57395c797a10a0fa044) Reclaim owned native arithmetic buffers after binding
- [fc555125](https://github.com/ICME-Lab/jolt-atlas/commit/fc555125c78bd7a3bbe31bce6a334164c7dc2ad9) Format the consolidated sources with Rust 1.95
- [3f5de68a](https://github.com/ICME-Lab/jolt-atlas/commit/3f5de68a85431182daa9d3ea6fd427eefb69557b) Reconcile owned openings and sparse-row fixtures with compact indices
- [d4e170de](https://github.com/ICME-Lab/jolt-atlas/commit/d4e170de5728aabb74c2d11b1803ab8e07e11ee5) Format compact-index reference fixtures
- [0a11f176](https://github.com/ICME-Lab/jolt-atlas/commit/0a11f1763c33a2792d3922e08d95757afe0d1565) Include hidden claim identities in bounded preparation fixtures
- [6d3e3a24](https://github.com/ICME-Lab/jolt-atlas/commit/6d3e3a24daf3a120baa05711e8ebac28d801daa9) Remove redundant cast in dense row reference fixture

## 7. native-verifier

- [9df490f6](https://github.com/ICME-Lab/jolt-atlas/commit/9df490f62855e1ae2486528abd6c936de9a4fba3) Add checked compact transport for native GT commitments
- [56abcc2a](https://github.com/ICME-Lab/jolt-atlas/commit/56abcc2a7567652881dc95d88edee1143bf62a44) Parallelize large Dory commitment combinations
- [34cb85ad](https://github.com/ICME-Lab/jolt-atlas/commit/34cb85adc7533aaaf29c5e4c7a4e47efd4ea9cfc) Use target-group MSM for large Dory commitment sums
- [9d9d9528](https://github.com/ICME-Lab/jolt-atlas/commit/9d9d95283f2d0622773d10475e4d07cf17db1e97) Avoid quadratic opening scans in BlindFold batching
- [328a8399](https://github.com/ICME-Lab/jolt-atlas/commit/328a8399eac869a2689b4261bcf3662f01bcf798) Use Frobenius for exact compact GT subgroup checks
- [cc1480ee](https://github.com/ICME-Lab/jolt-atlas/commit/cc1480ee579c766f0e228d34f0c059e57646c61d) Gate cyclotomic powering with exact membership checks
- [4b0c68cf](https://github.com/ICME-Lab/jolt-atlas/commit/4b0c68cf0ec159851269d048ad36b79ae9d18999) Reduce retained storage during native BlindFold verification
- [2e415e2a](https://github.com/ICME-Lab/jolt-atlas/commit/2e415e2a8bfd71678b7b62766a149b259dc166ce) Borrow single constraints in BlindFold storage tests
