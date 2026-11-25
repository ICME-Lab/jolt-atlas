use std::{io, net::TcpListener};

fn main() -> io::Result<()> {
    let address = std::env::args().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: akita-worker LISTEN_ADDRESS",
        )
    })?;
    let listener = TcpListener::bind(address)?;
    println!("{}", listener.local_addr()?);
    model_specific_proving::protocol::serve_worker(listener)
}
