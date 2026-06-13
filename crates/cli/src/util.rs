use std::{
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use log::error;
use tree_sitter::{Parser, Tree};
use tree_sitter_config::Config;
use tree_sitter_loader::Config as LoaderConfig;

use crate::error::{
    GrammarLookupError, IoError, LanguageConfigForFileNotFound, UtilError, UtilResult,
};

const HTML_HEADER: &[u8] = b"
<!DOCTYPE html>

<style>
svg { width: 100%; }
</style>

";

#[must_use]
pub fn lang_not_found_for_path(path: &Path, loader_config: &LoaderConfig) -> GrammarLookupError {
    GrammarLookupError::LanguageConfigForFileNotFound(LanguageConfigForFileNotFound {
        file: path.to_path_buf(),
        parser_directories: loader_config.parser_directories.clone(),
        config_location: Config::find_config_file().ok().flatten(),
    })
}

#[must_use]
pub fn cancel_on_signal() -> Arc<AtomicUsize> {
    let result = Arc::new(AtomicUsize::new(0));
    ctrlc::set_handler({
        let flag = result.clone();
        move || {
            flag.store(1, Ordering::Relaxed);
        }
    })
    .expect("Error setting Ctrl-C handler");
    result
}

pub struct LogSession {
    path: PathBuf,
    dot_process: Option<Child>,
    dot_process_stdin: Option<ChildStdin>,
    open_log: bool,
}

pub fn print_tree_graph(tree: &Tree, path: &str, quiet: bool) -> UtilResult<()> {
    let session = LogSession::new(path, quiet)?;
    tree.print_dot_graph(session.dot_process_stdin.as_ref().unwrap());
    Ok(())
}

pub fn log_graphs(parser: &mut Parser, path: &str, open_log: bool) -> UtilResult<LogSession> {
    let session = LogSession::new(path, open_log)?;
    parser.print_dot_graphs(session.dot_process_stdin.as_ref().unwrap());
    Ok(session)
}

impl LogSession {
    fn new(path: &str, open_log: bool) -> UtilResult<Self> {
        use std::io::Write;

        let path_buf = PathBuf::from(path);
        let mut dot_file = std::fs::File::create(path).map_err(IoError::at(&path_buf))?;
        dot_file
            .write_all(HTML_HEADER)
            .map_err(IoError::at(&path_buf))?;
        let mut dot_process = Command::new("dot")
            .arg("-Tsvg")
            .stdin(Stdio::piped())
            .stdout(dot_file)
            .spawn()
            .map_err(|source| UtilError::DotSpawn { source })?;
        let dot_stdin = dot_process.stdin.take().ok_or(UtilError::DotStdin)?;
        Ok(Self {
            path: path_buf,
            dot_process: Some(dot_process),
            dot_process_stdin: Some(dot_stdin),
            open_log,
        })
    }
}

impl Drop for LogSession {
    fn drop(&mut self) {
        use std::fs;

        drop(self.dot_process_stdin.take().unwrap());
        let output = self.dot_process.take().unwrap().wait_with_output().unwrap();
        if output.status.success() {
            if self.open_log && fs::metadata(&self.path).unwrap().len() > HTML_HEADER.len() as u64 {
                webbrowser::open(&self.path.to_string_lossy()).unwrap();
            }
        } else {
            error!(
                "Dot failed: {} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
