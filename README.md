# Changes in this fork

Add conversion from py None to rust unit type ()

Displaying PyErr includes traceback

Implement FromPyObject and IntoPy for various other crates:
Uuid
Smallvec
rustc_hash

Filter tracebacks to only include stack frames with `rust_circuit`

# PyO3

[![actions status](https://github.com/PyO3/pyo3/workflows/CI/badge.svg)](https://github.com/PyO3/pyo3/actions)
[![benchmark](https://github.com/PyO3/pyo3/actions/workflows/bench.yml/badge.svg)](https://pyo3.rs/dev/bench/)
[![codecov](https://codecov.io/gh/PyO3/pyo3/branch/main/graph/badge.svg)](https://codecov.io/gh/PyO3/pyo3)
[![crates.io](https://img.shields.io/crates/v/pyo3)](https://crates.io/crates/pyo3)
[![minimum rustc 1.48](https://img.shields.io/badge/rustc-1.48+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![dev chat](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/PyO3/Lobby)
[![contributing notes](https://img.shields.io/badge/contribute-on%20github-Green)](https://github.com/PyO3/pyo3/blob/main/Contributing.md)

[Rust](https://www.rust-lang.org/) bindings for [Python](https://www.python.org/), including tools for creating native Python extension modules. Running and interacting with Python code from a Rust binary is also supported.

- User Guide: [stable](https://pyo3.rs) | [main](https://pyo3.rs/main)

- API Documentation: [stable](https://docs.rs/pyo3/) | [main](https://pyo3.rs/main/doc)

## Usage

PyO3 supports the following software versions:
  - Python 3.7 and up (CPython and PyPy)
  - Rust 1.48 and up

You can use PyO3 to write a native Python module in Rust, or to embed Python in a Rust binary. The following sections explain each of these in turn.

### Using Rust from Python

PyO3 can be used to generate a native Python module. The easiest way to try this out for the first time is to use [`maturin`](https://github.com/PyO3/maturin). `maturin` is a tool for building and publishing Rust-based Python packages with minimal configuration. The following steps install `maturin`, use it to generate and build a new Python package, and then launch Python to import and execute a function from the package.

First, follow the commands below to create a new directory containing a new Python `virtualenv`, and install `maturin` into the virtualenv using Python's package manager, `pip`:

```bash
# (replace string_sum with the desired package name)
$ mkdir string_sum
$ cd string_sum
$ python -m venv .env
$ source .env/bin/activate
$ pip install maturin
```

Still inside this `string_sum` directory, now run `maturin init`. This will generate the new package source. When given the choice of bindings to use, select pyo3 bindings:

```bash
$ maturin init
✔ 🤷 What kind of bindings to use? · pyo3
  ✨ Done! New project created string_sum
```

The most important files generated by this command are `Cargo.toml` and `lib.rs`, which will look roughly like the following:

**`Cargo.toml`**

```toml
[package]
name = "string_sum"
version = "0.1.0"
edition = "2018"

[lib]
# The name of the native library. This is the name which will be used in Python to import the
# library (i.e. `import string_sum`). If you change this, you must also change the name of the
# `#[pymodule]` in `src/lib.rs`.
name = "string_sum"
# "cdylib" is necessary to produce a shared library for Python to import from.
#
# Downstream Rust code (including code in `bin/`, `examples/`, and `tests/`) will not be able
# to `use string_sum;` unless the "rlib" or "lib" crate type is also included, e.g.:
# crate-type = ["cdylib", "rlib"]
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.17.2", features = ["extension-module"] }
```

**`src/lib.rs`**

```rust
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn string_sum(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
```

Finally, run `maturin develop`. This will build the package and install it into the Python virtualenv previously created and activated. The package is then ready to be used from `python`:

```bash
$ maturin develop
# lots of progress output as maturin runs the compilation...
$ python
>>> import string_sum
>>> string_sum.sum_as_string(5, 20)
'25'
```

To make changes to the package, just edit the Rust source code and then re-run `maturin develop` to recompile.

To run this all as a single copy-and-paste, use the bash script below (replace `string_sum` in the first command with the desired package name):

```bash
mkdir string_sum && cd "$_"
python -m venv .env
source .env/bin/activate
pip install maturin
maturin init --bindings pyo3
maturin develop
```

If you want to be able to run `cargo test` or use this project in a Cargo workspace and are running into linker issues, there are some workarounds in [the FAQ](https://pyo3.rs/latest/faq.html#i-cant-run-cargo-test-or-i-cant-build-in-a-cargo-workspace-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror).

As well as with `maturin`, it is possible to build using [`setuptools-rust`](https://github.com/PyO3/setuptools-rust) or [manually](https://pyo3.rs/latest/building_and_distribution.html#manual-builds). Both offer more flexibility than `maturin` but require more configuration to get started.

### Using Python from Rust

To embed Python into a Rust binary, you need to ensure that your Python installation contains a shared library. The following steps demonstrate how to ensure this (for Ubuntu), and then give some example code which runs an embedded Python interpreter.

To install the Python shared library on Ubuntu:

```bash
sudo apt install python3-dev
```

Start a new project with `cargo new` and add  `pyo3` to the `Cargo.toml` like this:

```toml
[dependencies.pyo3]
version = "0.17.2"
features = ["auto-initialize"]
```

Example program displaying the value of `sys.version` and the current user name:

```rust
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let version: String = sys.getattr("version")?.extract()?;

        let locals = [("os", py.import("os")?)].into_py_dict(py);
        let code = "os.getenv('USER') or os.getenv('USERNAME') or 'Unknown'";
        let user: String = py.eval(code, None, Some(&locals))?.extract()?;

        println!("Hello {}, I'm Python {}", user, version);
        Ok(())
    })
}
```

The guide has [a section](https://pyo3.rs/latest/python_from_rust.html) with lots of examples
about this topic.

## Tools and libraries

- [maturin](https://github.com/PyO3/maturin) _Build and publish crates with pyo3, rust-cpython or cffi bindings as well as rust binaries as python packages_
- [setuptools-rust](https://github.com/PyO3/setuptools-rust) _Setuptools plugin for Rust support_.
- [pyo3-built](https://github.com/PyO3/pyo3-built) _Simple macro to expose metadata obtained with the [`built`](https://crates.io/crates/built) crate as a [`PyDict`](https://docs.rs/pyo3/*/pyo3/types/struct.PyDict.html)_
- [rust-numpy](https://github.com/PyO3/rust-numpy) _Rust binding of NumPy C-API_
- [dict-derive](https://github.com/gperinazzo/dict-derive) _Derive FromPyObject to automatically transform Python dicts into Rust structs_
- [pyo3-log](https://github.com/vorner/pyo3-log) _Bridge from Rust to Python logging_
- [pythonize](https://github.com/davidhewitt/pythonize) _Serde serializer for converting Rust objects to JSON-compatible Python objects_
- [pyo3-asyncio](https://github.com/awestlake87/pyo3-asyncio) _Utilities for working with Python's Asyncio library and async functions_
- [rustimport](https://github.com/mityax/rustimport) _Directly import Rust files or crates from Python, without manual compilation step. Provides pyo3 integration by default and generates pyo3 binding code automatically._

## Examples

- [hyperjson](https://github.com/mre/hyperjson) _A hyper-fast Python module for reading/writing JSON data using Rust's serde-json_
- [html-py-ever](https://github.com/PyO3/setuptools-rust/tree/main/examples/html-py-ever) _Using [html5ever](https://github.com/servo/html5ever) through [kuchiki](https://github.com/kuchiki-rs/kuchiki) to speed up html parsing and css-selecting._
- [point-process](https://github.com/ManifoldFR/point-process-rust/tree/master/pylib) _High level API for pointprocesses as a Python library_
- [autopy](https://github.com/autopilot-rs/autopy) _A simple, cross-platform GUI automation library for Python and Rust._
  - Contains an example of building wheels on TravisCI and appveyor using [cibuildwheel](https://github.com/pypa/cibuildwheel)
- [orjson](https://github.com/ijl/orjson) _Fast Python JSON library_
- [inline-python](https://github.com/fusion-engineering/inline-python) _Inline Python code directly in your Rust code_
- [Rogue-Gym](https://github.com/kngwyu/rogue-gym) _Customizable rogue-like game for AI experiments_
  - Contains an example of building wheels on Azure Pipelines
- [fastuuid](https://github.com/thedrow/fastuuid/) _Python bindings to Rust's UUID library_
- [wasmer-python](https://github.com/wasmerio/wasmer-python) _Python library to run WebAssembly binaries_
- [mocpy](https://github.com/cds-astro/mocpy) _Astronomical Python library offering data structures for describing any arbitrary coverage regions on the unit sphere_
- [tokenizers](https://github.com/huggingface/tokenizers/tree/main/bindings/python) _Python bindings to the Hugging Face tokenizers (NLP) written in Rust_
- [pyre](https://github.com/Project-Dream-Weaver/pyre-http) _Fast Python HTTP server written in Rust_
- [jsonschema-rs](https://github.com/Stranger6667/jsonschema-rs/tree/master/bindings/python) _Fast JSON Schema validation library_
- [css-inline](https://github.com/Stranger6667/css-inline/tree/master/bindings/python) _CSS inlining for Python implemented in Rust_
- [cryptography](https://github.com/pyca/cryptography/tree/main/src/rust) _Python cryptography library with some functionality in Rust_
- [polaroid](https://github.com/daggy1234/polaroid) _Hyper Fast and safe image manipulation library for Python written in Rust_
- [ormsgpack](https://github.com/aviramha/ormsgpack) _Fast Python msgpack library_
- [bed-reader](https://github.com/fastlmm/bed-reader) _Read and write the PLINK BED format, simply and efficiently_
    - Shows Rayon/ndarray::parallel (including capturing errors, controlling thread num), Python types to Rust generics, Github Actions
- [pyheck](https://github.com/kevinheavey/pyheck) _Fast case conversion library, built by wrapping [heck](https://github.com/withoutboats/heck)_
    - Quite easy to follow as there's not much code.
- [polars](https://github.com/pola-rs/polars) _Fast multi-threaded DataFrame library in Rust | Python | Node.js_
- [rust-python-coverage](https://github.com/cjermain/rust-python-coverage) _Example PyO3 project with automated test coverage for Rust and Python_
- [forust](https://github.com/jinlow/forust) _A lightweight gradient boosted decision tree library written in Rust._
- [ril-py](https://github.com/Cryptex-github/ril-py) _A performant and high-level image processing library for Python written in Rust_
- [fastbloom](https://github.com/yankun1992/fastbloom) _A fast [bloom filter](https://github.com/yankun1992/fastbloom#BloomFilter) | [counting bloom filter](https://github.com/yankun1992/fastbloom#countingbloomfilter) implemented by Rust for Rust and Python!_
- [river](https://github.com/online-ml/river) _Online machine learning in python, the computationally heavy statistics algorithms are implemented in Rust_
- [feos](https://github.com/feos-org/feos) _Lightning fast thermodynamic modeling in Rust with fully developed Python interface_

## Articles and other media

- [Nine Rules for Writing Python Extensions in Rust](https://towardsdatascience.com/nine-rules-for-writing-python-extensions-in-rust-d35ea3a4ec29?sk=f8d808d5f414154fdb811e4137011437) - Dec 31, 2021
- [Calling Rust from Python using PyO3](https://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/) - Nov 18, 2021
- [davidhewitt's 2021 talk at Rust Manchester meetup](https://www.youtube.com/watch?v=-XyWG_klSAw&t=320s) - Aug 19, 2021
- [Incrementally porting a small Python project to Rust](https://blog.waleedkhan.name/port-python-to-rust/) - Apr 29, 2021
- [Vortexa - Integrating Rust into Python](https://www.vortexa.com/insight/integrating-rust-into-python) - Apr 12, 2021
- [Writing and publishing a Python module in Rust](https://blog.yossarian.net/2020/08/02/Writing-and-publishing-a-python-module-in-rust) - Aug 2, 2020

## Contributing

Everyone is welcomed to contribute to PyO3! There are many ways to support the project, such as:

- help PyO3 users with issues on GitHub and Gitter
- improve documentation
- write features and bugfixes
- publish blogs and examples of how to use PyO3

Our [contributing notes](https://github.com/PyO3/pyo3/blob/main/Contributing.md) and [architecture guide](https://github.com/PyO3/pyo3/blob/main/Architecture.md) have more resources if you wish to volunteer time for PyO3 and are searching where to start.

If you don't have time to contribute yourself but still wish to support the project's future success, some of our maintainers have GitHub sponsorship pages:

- [davidhewitt](https://github.com/sponsors/davidhewitt)
- [messense](https://github.com/sponsors/messense)

## License

PyO3 is licensed under the [Apache-2.0 license](https://opensource.org/licenses/APACHE-2.0).
Python is licensed under the [Python License](https://docs.python.org/3/license.html).

<a href="https://www.netlify.com"> <img src="https://www.netlify.com/v3/img/components/netlify-color-accent.svg" alt="Deploys by Netlify" /> </a>
