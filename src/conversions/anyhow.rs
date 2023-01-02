#![cfg(feature = "anyhow")]

//! A conversion from
//! [anyhow](https://docs.rs/anyhow/ "A trait object based error system for easy idiomatic error handling in Rust applications.")’s
//! [`Error`](https://docs.rs/anyhow/latest/anyhow/struct.Error.html "Anyhows `Error` type, a wrapper around a dynamic error type")
//! type to [`PyErr`].
//!
//! Use of an error handling library like [anyhow] is common in application code and when you just
//! want error handling to be easy. If you are writing a library or you need more control over your
//! errors you might want to design your own error type instead.
//!
//! This implementation always creates a Python [`RuntimeError`]. You might find that you need to
//! map the error from your Rust code into another Python exception. See [`PyErr::new`] for more
//! information about that.
//!
//! For information about error handling in general, see the [Error handling] chapter of the Rust
//! book.
//!
//! # Setup
//!
//! To use this feature, add this to your **`Cargo.toml`**:
//!
//! ```toml
//! [dependencies]
//! ## change * to the version you want to use, ideally the latest.
//! anyhow = "*"
// workaround for `extended_key_value_attributes`: https://github.com/rust-lang/rust/issues/82768#issuecomment-803935643
#![cfg_attr(docsrs, cfg_attr(docsrs, doc = concat!("pyo3 = { version = \"", env!("CARGO_PKG_VERSION"),  "\", features = [\"anyhow\"] }")))]
#![cfg_attr(
    not(docsrs),
    doc = "pyo3 = { version = \"*\", features = [\"anyhow\"] }"
)]
//! ```
//!
//! Note that you must use compatible versions of anyhow and PyO3.
//! The required anyhow version may vary based on the version of PyO3.
//!
//! # Example: Propagating a `PyErr` into [`anyhow::Error`]
//!
//! ```rust
//! use pyo3::prelude::*;
//! use pyo3::wrap_pyfunction;
//! use std::path::PathBuf;
//!
//! // A wrapper around a Rust function.
//! // The pyfunction macro performs the conversion to a PyErr
//! #[pyfunction]
//! fn py_open(filename: PathBuf) -> anyhow::Result<Vec<u8>> {
//!     let data = std::fs::read(filename)?;
//!     Ok(data)
//! }
//!
//! fn main() {
//!     let error = Python::with_gil(|py| -> PyResult<Vec<u8>> {
//!         let fun = wrap_pyfunction!(py_open, py)?;
//!         let text = fun.call1(("foo.txt",))?.extract::<Vec<u8>>()?;
//!         Ok(text)
//!     }).unwrap_err();
//!
//!     println!("{}", error);
//! }
//! ```
//!
//! # Example: Using `anyhow` in general
//!
//! Note that you don't need this feature to convert a [`PyErr`] into an [`anyhow::Error`], because
//! it can already convert anything that implements [`Error`](std::error::Error):
//!
//! ```rust
//! use pyo3::prelude::*;
//! use pyo3::types::PyBytes;
//!
//! // An example function that must handle multiple error types.
//! //
//! // To do this you usually need to design your own error type or use
//! // `Box<dyn Error>`. `anyhow` is a convenient alternative for this.
//! pub fn decompress(bytes: &[u8]) -> anyhow::Result<String> {
//!     // An arbitrary example of a Python api you
//!     // could call inside an application...
//!     // This might return a `PyErr`.
//!     let res = Python::with_gil(|py| {
//!         let zlib = PyModule::import(py, "zlib")?;
//!         let decompress = zlib.getattr("decompress")?;
//!         let bytes = PyBytes::new(py, bytes);
//!         let value = decompress.call1((bytes,))?;
//!         value.extract::<Vec<u8>>()
//!     })?;
//!
//!     // This might be a `FromUtf8Error`.
//!     let text = String::from_utf8(res)?;
//!
//!     Ok(text)
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     let bytes: &[u8] = b"x\x9c\x8b\xcc/U(\xce\xc8/\xcdIQ((\xcaOJL\xca\xa9T\
//!                         (-NU(\xc9HU\xc8\xc9LJ\xcbI,IUH.\x02\x91\x99y\xc5%\
//!                         \xa9\x89)z\x00\xf2\x15\x12\xfe";
//!     let text = decompress(bytes)?;
//!
//!     println!("The text is \"{}\"", text);
//! # assert_eq!(text, "You should probably use the libflate crate instead.");
//!     Ok(())
//! }
//! ```
//!
//! [`RuntimeError`]: https://docs.python.org/3/library/exceptions.html#RuntimeError "Built-in Exceptions — Python documentation"
//! [Error handling]: https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html "Recoverable Errors with Result - The Rust Programming Language"

use crate::{
    exceptions::PyRuntimeError, once_cell::GILLazy, types::PyType, Py, PyErr, PyTypeInfo, Python,
};
use std::sync::Mutex;

// get exception stuff is Redwood hack. Could maybe be upstreamed, but would need to be improved...

// fallback only used on < 3.11
pub fn anyhow_to_py_err<F>(err: &anyhow::Error, fallback: F) -> PyErr
where
    F: for<'py> FnOnce(Python<'py>, PyErr, Vec<String>) -> PyErr,
{
    Python::with_gil(|py| {
        if let Some(py_err) = err.root_cause().downcast_ref::<PyErr>() {
            if err.chain().nth(1).is_none() {
                py_err.clone_ref(py)
            } else {
                let mut stack: Vec<String> = Vec::new();
                let mut iter = err.chain().peekable();
                while let Some(x) = iter.next() {
                    if iter.peek().is_none() {
                        break;
                    };
                    stack.push(format!("  rust context: {}", x));
                }
                if cfg!(Py_3_11) {
                    let py_err = py_err.clone_ref(py).into_value(py);
                    let py_err = py_err.as_ref(py);
                    for x in stack.iter().rev() {
                        py_err.call_method1("add_note", (x,)).unwrap();
                    }
                    PyErr::from_value(py_err)
                } else {
                    fallback(py, py_err.clone_ref(py), stack)
                }
            }
        } else {
            PyRuntimeError::new_err(format!("{:?}", err))
        }
    })
}

pub fn default_anyhow_to_py_err(err: anyhow::Error) -> PyErr {
    anyhow_to_py_err(&err, |_, _, _| PyRuntimeError::new_err(format!("{:?}", err)))
}

static ANYHOW_TO_PY_ERR: GILLazy<
    Mutex<Box<dyn Fn(anyhow::Error) -> PyErr + Send + Sync + 'static>>,
> = GILLazy::new(|| Mutex::new(Box::new(default_anyhow_to_py_err)));

/// set anyhow exception getter
pub fn set_anyhow_to_py_err(f: Box<dyn Fn(anyhow::Error) -> PyErr + Send + Sync + 'static>) {
    *ANYHOW_TO_PY_ERR.lock().unwrap() = f;
}

impl From<anyhow::Error> for PyErr {
    fn from(err: anyhow::Error) -> Self {
        ANYHOW_TO_PY_ERR.lock().unwrap()(err)
    }
}

#[cfg(test)]
mod test_anyhow {
    use crate::exceptions::PyTypeError;
    use crate::prelude::*;
    use crate::types::{IntoPyDict, PyType};

    use anyhow::{anyhow, bail, Context, Error, Result};

    fn f() -> Result<()> {
        use std::io;
        bail!(io::Error::new(io::ErrorKind::PermissionDenied, "oh no!"));
    }

    fn g() -> Result<()> {
        f().context("f failed")
    }

    fn h() -> Result<()> {
        g().context("g failed")
    }

    #[test]
    fn test_pyo3_exception_contents() {
        let err = h().unwrap_err();
        let expected_contents = format!("{:?}", err);
        let pyerr = PyErr::from(err);

        Python::with_gil(|py| {
            let locals = [("err", pyerr)].into_py_dict(py);
            let pyerr = py.run("raise err", None, Some(locals)).unwrap_err();
            assert_eq!(pyerr.value(py).to_string(), expected_contents);
        })
    }

    fn k() -> Result<()> {
        Err(anyhow!("Some sort of error"))
    }

    #[test]
    fn test_pyo3_exception_contents2() {
        let err = k().unwrap_err();
        let expected_contents = format!("{:?}", err);
        let pyerr = PyErr::from(err);

        Python::with_gil(|py| {
            let locals = [("err", pyerr)].into_py_dict(py);
            let pyerr = py.run("raise err", None, Some(locals)).unwrap_err();
            assert_eq!(pyerr.value(py).to_string(), expected_contents);
        })
    }

    #[test]
    fn test_pyo3_exception_different_type() {
        let err: PyErr = Error::new(PyTypeError::new_err("Some sort of error")).into();
        Python::with_gil(|py| {
            assert!(err.get_type(py).is(PyType::new::<PyTypeError>(py)));
        })
    }
}
