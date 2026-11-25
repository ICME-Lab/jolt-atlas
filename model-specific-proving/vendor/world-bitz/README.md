# World BitZ prover crates

Atlas uses the low-memory versions of `gkr` and `post_gkr` in this directory. The base is
World BitZ revision `34addacd27d63b0929512bcc90fe95d256362b74`. The other BitZ crates remain
Git dependencies at the same revision.

The GKR prover folds the bit-packed witness before allocating dense child buffers. The
post-GKR prover projects the packed witness without expanding it into a full field-element
matrix. Cargo selects these two crates through the BitZ patch in the model crate manifest.

When updating BitZ, update the Git dependencies in both manifests and rebase the local prover
changes on the corresponding versions of the two crates.
