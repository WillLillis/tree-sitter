use crate::{bail_on_err, root_dir, FetchFixtures, EMSCRIPTEN_VERSION};
use anyhow::Result;
use std::{
    fs,
    process::{Command, Stdio},
};

pub fn run_fixtures(args: &FetchFixtures) -> Result<()> {
    let fixtures_dir = root_dir().join("test").join("fixtures");
    let grammars_dir = fixtures_dir.join("grammars");
    let fixtures_path = fixtures_dir.join("fixtures.json");

    // grammar name, branch, commit sha
    let mut fixtures: Vec<(String, String, String)> =
        serde_json::from_str(&fs::read_to_string(&fixtures_path)?)?;

    for (grammar, branch, sha) in &mut fixtures {
        let grammar_dir = grammars_dir.join(&grammar);
        let grammar_url = format!("https://github.com/tree-sitter/tree-sitter-{grammar}");

        println!("Fetching the {grammar} grammar...");

        if !grammar_dir.exists() {
            let mut command = Command::new("git");
            command.args([
                "clone",
                "--depth",
                "1",
                &grammar_url,
                &grammar_dir.to_string_lossy(),
            ]);
            bail_on_err(
                &command.spawn()?.wait_with_output()?,
                &format!("Failed to clone the {grammar} grammar"),
            )?;
        }

        std::env::set_current_dir(&grammar_dir)?;

        let mut command = Command::new("git");
        command.args(["fetch", "origin", sha, "--depth", "1"]);
        bail_on_err(
            &command.spawn()?.wait_with_output()?,
            &format!("Failed to fetch the {grammar} grammar"),
        )?;

        let mut command = Command::new("git");
        let reset_ref = if args.update {
            format!("origin/{branch}")
        } else {
            sha.clone()
        };
        command.args(["reset", "--hard", &reset_ref]);
        bail_on_err(
            &command.spawn()?.wait_with_output()?,
            &format!("Failed to reset the {grammar} grammar"),
        )?;
        if args.update {
            let mut command = Command::new("git");
            command.args(["rev-parse", "HEAD"]).stdout(Stdio::piped());
            let update_out = command.spawn()?.wait_with_output()?;
            bail_on_err(
                &update_out,
                &format!("Failed to parse the {grammar} grammar's latest commit"),
            )?;
            let new_sha = String::from_utf8(update_out.stdout)?.trim().to_string();
            if !new_sha.eq(sha) {
                println!("Updating the {grammar} grammar from {sha} to {new_sha}...");
                *sha = new_sha;
            }
        }
    }

    if args.update {
        fs::write(&fixtures_path, serde_json::to_string_pretty(&fixtures)?)?;
    }

    Ok(())
}

pub fn run_emscripten() -> Result<()> {
    let emscripten_dir = root_dir().join("target").join("emsdk");
    if emscripten_dir.exists() {
        println!("Emscripten SDK already exists");
        return Ok(());
    }
    println!("Cloning the Emscripten SDK...");

    let mut command = Command::new("git");
    command.args([
        "clone",
        "https://github.com/emscripten-core/emsdk.git",
        &emscripten_dir.to_string_lossy(),
    ]);
    bail_on_err(
        &command.spawn()?.wait_with_output()?,
        "Failed to clone the Emscripten SDK",
    )?;

    std::env::set_current_dir(&emscripten_dir)?;

    let emsdk = if cfg!(windows) {
        "emsdk.bat"
    } else {
        "./emsdk"
    };

    let mut command = Command::new(emsdk);
    command.args(["install", EMSCRIPTEN_VERSION]);
    bail_on_err(
        &command.spawn()?.wait_with_output()?,
        "Failed to install Emscripten",
    )?;

    let mut command = Command::new(emsdk);
    command.args(["activate", EMSCRIPTEN_VERSION]);
    bail_on_err(
        &command.spawn()?.wait_with_output()?,
        "Failed to activate Emscripten",
    )
}
