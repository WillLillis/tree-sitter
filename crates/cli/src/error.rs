use std::{
    io,
    num::ParseIntError,
    path::{Path, PathBuf},
};

use thiserror::Error;
use tree_sitter::QueryError;
use tree_sitter_config::ConfigError;
use tree_sitter_generate::GenerateError;
use tree_sitter_loader::LoaderError;

use crate::{test_highlight, test_tags, version};

pub type CliResult<T> = Result<T, CliError>;
pub type GrammarLookupResult<T> = Result<T, GrammarLookupError>;
pub type InputSelectionResult<T> = Result<T, InputSelectionError>;
pub type UtilResult<T> = Result<T, UtilError>;
pub type QueryTestingResult<T> = Result<T, QueryTestingError>;
pub type InitConfigResult<T> = Result<T, InitConfigError>;
pub type InitResult<T> = Result<T, InitError>;
pub type GenerateCmdResult<T> = Result<T, GenerateCmdError>;
pub type BuildCmdResult<T> = Result<T, BuildCmdError>;
pub type WasmCmdResult<T> = Result<T, WasmCmdError>;
pub type ParseCmdResult<T> = Result<T, ParseCmdError>;
pub type TestCmdResult<T> = Result<T, TestCmdError>;
pub type FuzzCmdResult<T> = Result<T, FuzzCmdError>;
pub type QueryCmdResult<T> = Result<T, QueryCmdError>;
pub type HighlightCmdResult<T> = Result<T, HighlightCmdError>;
pub type TagsCmdResult<T> = Result<T, TagsCmdError>;
pub type PlaygroundResult<T> = Result<T, PlaygroundError>;
pub type DumpLanguagesResult<T> = Result<T, DumpLanguagesError>;

#[derive(Debug, Error)]
pub struct IoError {
    pub error: io::Error,
    pub path: Option<PathBuf>,
}

impl IoError {
    pub fn new(error: io::Error, path: Option<&Path>) -> Self {
        Self {
            error,
            path: path.map(Path::to_path_buf),
        }
    }

    pub fn at(path: impl Into<PathBuf>) -> impl FnOnce(io::Error) -> Self {
        let path = path.into();
        move |error| Self {
            error,
            path: Some(path),
        }
    }
}

// Implicit conversion for path-less io errors (e.g., writes to stdout, stdin reads).
// For io ops with a known path, always use `IoError::at(&p)` explicitly.
impl From<io::Error> for IoError {
    fn from(error: io::Error) -> Self {
        Self { error, path: None }
    }
}

// `From` is not transitive in Rust, so this short-circuit impl is needed on every
// subcommand error that has an `Io(IoError)` variant to let `?` work directly on
// `io::Error`.
macro_rules! impl_from_io_error {
    ($($ty:ty),* $(,)?) => {
        $(impl From<io::Error> for $ty {
            fn from(error: io::Error) -> Self {
                Self::Io(IoError::from(error))
            }
        })*
    };
}

impl_from_io_error!(
    InitConfigError,
    InitError,
    GenerateCmdError,
    BuildCmdError,
    WasmCmdError,
    ParseCmdError,
    TestCmdError,
    FuzzCmdError,
    QueryCmdError,
    HighlightCmdError,
    TagsCmdError,
    PlaygroundError,
    UtilError,
    InputSelectionError,
    QueryTestingError,
);

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)?;
        if let Some(path) = &self.path {
            write!(f, " ({})", path.display())?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub struct LanguageConfigForFileNotFound {
    pub file: PathBuf,
    pub parser_directories: Vec<PathBuf>,
    pub config_location: Option<PathBuf>,
}

impl std::fmt::Display for LanguageConfigForFileNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let file = self.file.display();
        writeln!(f, "No language found for path `{file}`\n")?;
        writeln!(
            f,
            "If a language should be associated with this file extension, please ensure the path to `{file}` is inside one of the following directories as specified by your 'config.json':\n"
        )?;
        for (i, dir) in self.parser_directories.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, dir.display())?;
        }
        writeln!(f)?;
        let trailer = match &self.config_location {
            Some(p) => format!("located at {}", p.display()),
            None => "which you need to create by running `tree-sitter init-config`".to_string(),
        };
        write!(
            f,
            "If the directory that contains the relevant grammar for `{file}` is not listed above, please add the directory to the list of directories in your config file, {trailer}"
        )
    }
}

#[derive(Debug, Error)]
pub enum GrammarLookupError {
    #[error("No language found in current path: {path}")]
    LanguageNotFound { path: PathBuf },
    #[error("No language configuration found in current path: {path}")]
    LanguageConfigNotFound { path: PathBuf },
    #[error("Language not found: {name}")]
    LanguageNotFoundByName { name: String },
    #[error(transparent)]
    LanguageConfigForFileNotFound(#[from] LanguageConfigForFileNotFound),
    #[error("Failed to load language for path {path}: {source}")]
    LanguageLoadFailed {
        path: PathBuf,
        #[source]
        source: LoaderError,
    },
    #[error("No language name found for {lib_path}")]
    LanguageNameNotInferred { lib_path: PathBuf },
    #[error("Unknown scope: {scope}")]
    UnknownScope { scope: String },
}

#[derive(Debug, Error)]
pub enum QueryTestingError {
    #[error(
        "Error: could not find a line that corresponds to the assertion `{capture}` located at {position}"
    )]
    AssertionLineNotFound {
        capture: String,
        position: crate::query_testing::Utf8Point,
    },
    #[error("Assertion failed: at {position}, found {found}, expected {expected}")]
    AssertionMismatch {
        position: crate::query_testing::Utf8Point,
        found: String,
        expected: String,
    },
    #[error("Assertion failed: could not match {capture} at row {row}, column {column}")]
    AssertionUnmatched {
        capture: String,
        row: usize,
        column: usize,
    },
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum UtilError {
    #[error("Failed to run the `dot` command. Check that graphviz is installed. ({source})")]
    DotSpawn {
        #[source]
        source: io::Error,
    },
    #[error("Failed to open stdin for `dot` process")]
    DotStdin,
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum InputSelectionError {
    #[error(
        "Test corpus directory not found in current directory, see https://tree-sitter.github.io/tree-sitter/creating-parsers/5-writing-tests"
    )]
    TestCorpusNotFound,
    #[error("Failed to fetch contents of test #{number}")]
    TestNumberNotFound { number: u32 },
    #[error("Invalid path: {path}")]
    InvalidPath { path: PathBuf },
    #[error("Invalid glob pattern {pattern}: {source}")]
    InvalidGlob {
        pattern: String,
        #[source]
        source: glob::PatternError,
    },
    #[error("No files were found at or matched by the provided pathname/glob")]
    NoFilesFound,
    #[error("")]
    Interrupted,
    #[error(transparent)]
    Glob(#[from] glob::GlobError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum InitConfigError {
    #[error("Configuration file already exists: {path}")]
    ConfigExists { path: PathBuf },
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("Failed to parse {path}: {source}")]
    ParseConfigFile {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "Failed to locate a '{}' file in '{}' or any parent directory. Please ensure you have one; consult the docs.",
        filename.display(), search_root.display()
    )]
    ConfigFileNotFound {
        filename: PathBuf,
        search_root: PathBuf,
    },
    #[error(transparent)]
    Generate(#[from] tree_sitter_generate::GenerateError),
    #[error(transparent)]
    Dialoguer(#[from] dialoguer::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum GenerateCmdError {
    #[error(transparent)]
    Generate(#[from] GenerateError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
#[error(
    "--wasm flag specified, but this build of tree-sitter-cli does not include the wasm feature"
)]
pub struct WasmFeatureDisabled;

#[derive(Debug, Error)]
pub enum BuildCmdError {
    #[error("Output path must have a parent: {path}")]
    OutputPathNoParent { path: PathBuf },
    #[error("Output path must have a filename: {path}")]
    OutputPathNoFilename { path: PathBuf },
    #[error("Failed to compile parser: {0}")]
    Compile(#[source] LoaderError),
    #[error(transparent)]
    WasmDisabled(#[from] WasmFeatureDisabled),
    #[error(transparent)]
    Wasm(#[from] WasmCmdError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum WasmCmdError {
    #[error("Failed to read {path}. Run `tree-sitter build --wasm` first. ({source})")]
    ReadWasmBinary {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Failed to parse grammar file {path}: {source}")]
    ParseGrammarFile {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "This external scanner uses a symbol that isn't available to Wasm parsers.\n\nMissing symbols:\n    {}\n\nAvailable symbols:\n    {}",
        missing.join("\n    "),
        available.join("\n    ")
    )]
    DisallowedScannerSymbols {
        missing: Vec<String>,
        available: Vec<String>,
    },
    #[error(transparent)]
    LoadGrammar(#[from] tree_sitter_generate::LoadGrammarError),
    #[error(transparent)]
    WasmParser(#[from] wasmparser::BinaryReaderError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum ParseCmdError {
    #[error(transparent)]
    GrammarLookup(#[from] GrammarLookupError),
    #[error(transparent)]
    InputSelection(#[from] InputSelectionError),
    #[error("Failed to parse CST theme: {0}")]
    ParseCstTheme(#[source] serde_json::Error),
    #[error(transparent)]
    ParseRange(#[from] ParseRangeError),
    #[error("Failed to address a row: {row}")]
    PositionRowOutOfBounds { row: usize },
    #[error("Failed to address a column: {column}")]
    PositionColumnOutOfBounds { column: usize },
    #[error("Failed to address a column over the end")]
    PositionPastEnd,
    #[error("Failed to address an offset: {offset}")]
    OffsetOutOfBounds { offset: usize },
    // Marker error only. Parse-failure summary and its display are owned by the parser
    // (see options.stats printing), not this variant.
    #[error("")]
    ParseHadErrors,
    #[error(
        "Invalid edit string '{flag}'. Edit strings must match the pattern '<START_BYTE_OR_POSITION> <REMOVED_LENGTH> <NEW_TEXT>'"
    )]
    InvalidEditFlag { flag: String },
    #[error(transparent)]
    WasmDisabled(#[from] WasmFeatureDisabled),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Language(#[from] tree_sitter::LanguageError),
    #[error(transparent)]
    Util(#[from] UtilError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Clone, Copy)]
pub enum TestKind {
    Parse,
    Highlight,
    Tags,
}

impl std::fmt::Display for TestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse => write!(f, "parse"),
            Self::Highlight => write!(f, "highlight"),
            Self::Tags => write!(f, "tags"),
        }
    }
}

#[derive(Debug, Error)]
pub enum TestCmdError {
    // Marker error only. Pass/fail counts and their display are owned by the
    // test runner (see test_summary printing in main), not this variant.
    #[error("")]
    TestsFailed { kind: TestKind },
    #[error(
        "Some tests failed to parse with unexpected ERROR or MISSING nodes, and cannot be updated automatically. Either fix the grammar or manually update the test files."
    )]
    ParseTestsHaveUnexpectedErrors,
    #[error("Error in query file {path}: {source}")]
    InvalidQueryFile {
        path: PathBuf,
        #[source]
        source: QueryError,
    },
    #[error("No highlighting config found for {path}")]
    NoHighlightConfig { path: PathBuf },
    #[error("No tags config found for {path}")]
    NoTagsConfig { path: PathBuf },
    #[error(transparent)]
    GrammarLookup(#[from] GrammarLookupError),
    #[error(transparent)]
    InputSelection(#[from] InputSelectionError),
    #[error(transparent)]
    Util(#[from] UtilError),
    #[error(transparent)]
    QueryTesting(#[from] QueryTestingError),
    #[error(transparent)]
    HighlightFailure(#[from] test_highlight::Failure),
    #[error(transparent)]
    TagFailure(#[from] test_tags::Failure),
    #[error(transparent)]
    Query(#[from] QueryCmdError),
    #[error(transparent)]
    Tags(#[from] tree_sitter_tags::Error),
    #[error(transparent)]
    Highlight(#[from] tree_sitter_highlight::Error),
    #[error(transparent)]
    WasmDisabled(#[from] WasmFeatureDisabled),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Language(#[from] tree_sitter::LanguageError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum FuzzCmdError {
    #[error(transparent)]
    GrammarLookup(#[from] GrammarLookupError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum ParseRangeError {
    #[error("Invalid range '{input}': missing colon separator")]
    MissingColon { input: String },
    #[error("Invalid range '{input}': start is not a number")]
    BadStart { input: String },
    #[error("Invalid range '{input}': missing end")]
    MissingEnd { input: String },
    #[error("Invalid range '{input}': end is not a number")]
    BadEnd { input: String },
}

pub type ParseRangeResult<T> = Result<T, ParseRangeError>;

#[derive(Debug, Error)]
pub enum QueryCmdError {
    #[error("Query compilation failed for {path}: {source}")]
    InvalidQueryFile {
        path: PathBuf,
        #[source]
        source: QueryError,
    },
    #[error(transparent)]
    Language(#[from] tree_sitter::LanguageError),
    #[error(transparent)]
    QueryTesting(#[from] QueryTestingError),
    #[error(transparent)]
    GrammarLookup(#[from] GrammarLookupError),
    #[error(transparent)]
    InputSelection(#[from] InputSelectionError),
    #[error(transparent)]
    ParseRange(#[from] ParseRangeError),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum HighlightCmdError {
    #[error(transparent)]
    GrammarLookup(#[from] GrammarLookupError),
    #[error(transparent)]
    InputSelection(#[from] InputSelectionError),
    #[error(transparent)]
    Highlight(#[from] tree_sitter_highlight::Error),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum TagsCmdError {
    #[error(transparent)]
    GrammarLookup(#[from] GrammarLookupError),
    #[error(transparent)]
    InputSelection(#[from] InputSelectionError),
    #[error(transparent)]
    Tags(#[from] tree_sitter_tags::Error),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum PlaygroundError {
    #[error("Failed to write HTTP response: {0}")]
    WriteHttpResponse(#[source] io::Error),
    #[error("Invalid port specification '{value}': {source}")]
    InvalidPort {
        value: String,
        #[source]
        source: ParseIntError,
    },
    #[error("Failed to bind to {addr}:{port}: {source}")]
    BindPort {
        addr: String,
        port: u16,
        #[source]
        source: io::Error,
    },
    #[error("Failed to find a free port to bind to on {addr}")]
    NoFreePort { addr: String },
    #[error("Failed to start web server: {0}")]
    ServerStart(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Wasm(#[from] WasmCmdError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Debug, Error)]
pub enum DumpLanguagesError {
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Config(#[from] ConfigError),
}

#[derive(Debug, Error)]
pub enum CliError {
    #[error(transparent)]
    InitConfig(#[from] InitConfigError),
    #[error(transparent)]
    Init(#[from] InitError),
    #[error(transparent)]
    Generate(#[from] GenerateCmdError),
    #[error(transparent)]
    Build(#[from] BuildCmdError),
    #[error(transparent)]
    Parse(#[from] ParseCmdError),
    #[error(transparent)]
    Test(#[from] TestCmdError),
    #[error(transparent)]
    Version(#[from] version::VersionError),
    #[error(transparent)]
    Fuzz(#[from] FuzzCmdError),
    #[error(transparent)]
    Query(#[from] QueryCmdError),
    #[error(transparent)]
    Highlight(#[from] HighlightCmdError),
    #[error(transparent)]
    Tags(#[from] TagsCmdError),
    #[error(transparent)]
    Playground(#[from] PlaygroundError),
    #[error(transparent)]
    DumpLanguages(#[from] DumpLanguagesError),
    #[error(transparent)]
    Args(#[from] clap::Error),
    #[error("Failed to initialize loader: {0}")]
    LoaderInit(#[from] LoaderError),
}

impl CliError {
    #[must_use]
    pub fn is_broken_pipe(&self) -> bool {
        let io_err = match self {
            Self::Parse(ParseCmdError::Io(e))
            | Self::Build(BuildCmdError::Io(e))
            | Self::Query(QueryCmdError::Io(e))
            | Self::Highlight(HighlightCmdError::Io(e))
            | Self::Tags(TagsCmdError::Io(e))
            | Self::Playground(PlaygroundError::Io(e))
            | Self::Test(TestCmdError::Io(e)) => Some(&e.error),
            _ => None,
        };
        io_err.is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
    }
}
