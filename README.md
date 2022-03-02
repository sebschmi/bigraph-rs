# Bigraph

[![](http://meritbadge.herokuapp.com/bigraph)](https://crates.io/crates/bigraph)
[![](https://docs.rs/bigraph/badge.svg)](https://docs.rs/bigraph)
![](https://github.com/sebschmi/bigraph-rs/workflows/Tests%20and%20Lints/badge.svg?branch=main)

A Rust crate to represent and operate on bigraphs.

The crate defines traits as abstractions from the concrete graph implementation.
At the moment, the only implementation is done via a wrapper over [petgraph](https://crates.io/crates/petgraph), which serves as a prototype.
