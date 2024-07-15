format: 
	cargo fmt --all
	cargo clippy --all-features -- -D warnings

check_format:
	cargo fmt --all -- --check
	cargo clippy --all-features -- -D warnings

test:
	cargo test --all-features

debug:
	maturin develop --features py

release:
	maturin develop --release --features py