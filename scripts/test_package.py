#!/usr/bin/env python3
"""Build-script/source-integrity tests, not substitutes for compiling Rust.

Each workflow runs in a disposable checkout and HOME with an explicit Cargo
fixture. That fixture only records arguments and makes a labelled test executable.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('build.sh', 'clean-build.sh', 'clean.sh', 'check.sh', 'test.sh',
           'format.sh', 'run.sh', 'verify.sh', 'install.sh',
           'rebuild-and-install.sh', 'rebuild-and-run.sh', 'setup.sh')

FAKE_CARGO = r'''#!/usr/bin/env -S python3 -S
import json, os, pathlib, shutil, sys, tomllib
args = sys.argv[1:]
with open(os.environ['FAKE_CARGO_LOG'], 'a') as f:
    f.write(json.dumps({'args': args, 'cwd': os.getcwd(),
                       'target': os.environ.get('CARGO_TARGET_DIR'),
                       'build_dir': os.environ.get('CARGO_BUILD_BUILD_DIR')}) + '\n')
if args and args[0] == os.environ.get('FAKE_CARGO_FAIL'):
    print('Intentional packaging-test fixture failure', file=sys.stderr)
    sys.exit(41)
if args and args[0] == 'clean':
    path = pathlib.Path(args[args.index('--target-dir') + 1])
    if '--dry-run' not in args and path.exists():
        shutil.rmtree(path)
if args and args[0] == 'build':
    profile = 'release' if '--release' in args else 'debug'
    out = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / profile / 'ollama-workbench'
    out.parent.mkdir(parents=True, exist_ok=True)
    version = tomllib.loads(pathlib.Path('Cargo.toml').read_text())['package']['version']
    identity = '0.0.0-STALE' if os.environ.get('FAKE_STALE_IDENTITY') else version
    out.write_text('#!/usr/bin/env -S python3 -S\n'
                   'import json, sys\n'
                   'if sys.argv[1:] == ["--version"]:\n'
                   f'    print("AI Workbench {identity} (source 0123456789abcdef)")\n'
                   'else:\n'
                   '    print("FRESH_BUILD " + json.dumps(sys.argv[1:]))\n')
    out.chmod(0o755)
print('Packaging fixture only: cargo', ' '.join(args))
'''

class LaunchAndInstall(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='workbench packaging ')
        self.base = Path(self.temp.name)
        self.project = self.base / 'checkout with spaces'
        self.project.mkdir()
        for name in SCRIPTS:
            shutil.copy2(ROOT / name, self.project / name)
        (self.project / 'scripts').mkdir()
        shutil.copy2(ROOT / 'scripts/workbench.sh', self.project / 'scripts/workbench.sh')
        (self.project / 'assets').mkdir()
        (self.project / 'assets/ollama-workbench.svg').write_text('<svg/>')
        for name in ['Cargo.toml', 'Cargo.lock']:
            shutil.copy2(ROOT / name, self.project / name)
        self.home = self.base / 'home with spaces'
        self.home.mkdir()
        self.tools = self.base / 'fixture commands'
        self.tools.mkdir()
        (self.tools / 'cargo').write_text(FAKE_CARGO)
        (self.tools / 'cargo').chmod(0o755)
        (self.tools / 'rustc').write_text('#!/bin/sh\necho "rustc PACKAGING_FIXTURE_NOT_A_COMPILER"\n')
        (self.tools / 'rustc').chmod(0o755)
        self.log = self.base / 'cargo.jsonl'
        self.env = {**os.environ, 'HOME': str(self.home), 'XDG_DATA_HOME': str(self.home / 'data'),
                    'PATH': str(self.tools) + os.pathsep + os.environ.get('PATH', ''),
                    'FAKE_CARGO_LOG': str(self.log)}
        for key in ['FAKE_CARGO_FAIL', 'FAKE_STALE_IDENTITY', 'CARGO_BUILD_TARGET',
                    'CARGO_NET_OFFLINE', 'CARGO_BUILD_JOBS']:
            self.env.pop(key, None)

    def tearDown(self):
        self.temp.cleanup()

    def command(self, name, *args):
        return subprocess.run([str(self.project / name), *args], cwd=self.base,
                              env=self.env, text=True, capture_output=True, timeout=30)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def commands(self):
        return [call['args'][0] for call in self.calls() if call['args'][0] != '--version']

    def old_binary(self):
        target = self.home / '.local/bin/ollama-workbench'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#!/bin/sh\necho PREVIOUS_BUILD\n')
        target.chmod(0o755)
        return target

    def marker(self, path, content='retain this'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_setup_installs_zstd_before_installing_local_ollama(self):
        # Exercise setup even on hosts that already have Ollama or run as root.
        for name in ('bash', 'dirname', 'uname', 'sort', 'head', 'awk'):
            (self.tools / name).symlink_to(shutil.which(name))
        for name in ('apt-get', 'rustup'):
            (self.tools / name).symlink_to(shutil.which('true'))
        (self.tools / 'id').write_text('#!/bin/sh\necho 1000\n')
        (self.tools / 'id').chmod(0o755)
        self.env['PATH'] = str(self.tools)
        self.assertFalse((self.home / '.cargo/env').exists())
        flags = ('--dry-run', '--no-optional', '--no-install')
        local = self.command('setup.sh', '--ollama', *flags)
        self.assertEqual(local.returncode, 0, local.stderr)
        install_zstd = 'apt-get install -y --no-install-recommends zstd'
        self.assertLess(local.stdout.index(install_zstd),
                        local.stdout.index('https://ollama.com/install.sh'))
        remote = self.command('setup.sh', '--no-ollama', *flags)
        self.assertEqual(remote.returncode, 0, remote.stderr)
        self.assertNotIn(install_zstd, remote.stdout)

    def test_release_build_is_locked_and_does_not_clean_or_install(self):
        old = self.old_binary()
        result = self.command('build.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ['build'])
        self.assertEqual(self.calls()[-1]['args'], ['build', '--bin', 'ollama-workbench', '--release',
                                                  '--locked', '--target-dir', str(self.project / 'target')])
        self.assertIn('PREVIOUS_BUILD', old.read_text())
        self.assertTrue((self.project / 'target/release/ollama-workbench').exists())
        self.assertTrue(list((self.project / 'reports/local-build').glob('build-*.log')))

    def test_debug_build_offline_jobs_and_checkout_with_spaces(self):
        result = self.command('build.sh', '--debug', '--offline', '--jobs', '2')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.calls()[-1]['args']
        self.assertNotIn('--release', args)
        self.assertIn('--offline', args)
        self.assertEqual(args[-2:], ['--jobs', '2'])
        self.assertEqual(self.calls()[-1]['cwd'], str(self.project))
        self.assertTrue((self.project / 'target/debug/ollama-workbench').exists())

    def test_clean_build_cleans_before_build_and_preserves_data(self):
        artifact = self.marker(self.project / 'target/debug/old')
        source = self.marker(self.project / 'src/keep.rs')
        data = self.marker(self.home / 'data/ollama-workbench/settings.json')
        model = self.marker(self.home / '.ollama/models/weight')
        cache = self.marker(self.home / '.cargo/registry/keep')
        lock = (self.project / 'Cargo.lock').read_bytes()
        result = self.command('clean-build.sh', '--offline', '-j', '2')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ['clean', 'build'])
        self.assertFalse(artifact.exists())
        self.assertTrue(all(p.read_text() == 'retain this' for p in [source, data, model, cache]))
        self.assertEqual((self.project / 'Cargo.lock').read_bytes(), lock)

    def test_clean_only_never_builds_and_overrides_external_output_paths(self):
        artifact = self.marker(self.project / 'target/release/old')
        external = self.marker(self.base / 'shared-target/keep')
        self.env['CARGO_TARGET_DIR'] = str(external.parent)
        self.env['CARGO_BUILD_BUILD_DIR'] = str(external.parent)
        result = self.command('clean.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ['clean'])
        self.assertFalse(artifact.exists())
        self.assertTrue(external.exists())
        self.assertEqual(self.calls()[-1]['target'], str(self.project / 'target'))
        self.assertEqual(self.calls()[-1]['build_dir'], str(self.project / 'target'))

    def test_clean_dry_run_does_not_remove_output(self):
        artifact = self.marker(self.project / 'target/release/keep')
        result = self.command('clean.sh', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(artifact.exists())
        self.assertIn('--dry-run', self.calls()[-1]['args'])

    def test_symlinked_target_is_refused_before_clean(self):
        external = self.marker(self.base / 'shared-target/keep')
        (self.project / 'target').symlink_to(external.parent, target_is_directory=True)
        result = self.command('clean.sh')
        self.assertEqual(result.returncode, 2)
        self.assertIn('symlinked build directory', result.stdout)
        self.assertTrue(external.exists())
        self.assertNotIn('clean', self.commands())

    def test_dangling_symlink_target_is_refused_before_build(self):
        (self.project / 'target').symlink_to(self.base / 'missing-output')
        result = self.command('build.sh')
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('build', self.commands())

    def test_invalid_options_never_clean(self):
        artifact = self.marker(self.project / 'target/release/keep')
        for args in [('--jobs', '0'), ('--jobs',), ('--jobs=-1',), ('--unknown',)]:
            result = self.command('clean-build.sh', *args)
            self.assertEqual(result.returncode, 2, str(args))
        self.assertTrue(artifact.exists())
        self.assertFalse(self.calls())

    def test_failed_clean_prevents_rebuild(self):
        self.env['FAKE_CARGO_FAIL'] = 'clean'
        result = self.command('clean-build.sh')
        self.assertEqual(result.returncode, 41)
        self.assertEqual(self.commands(), ['clean'])

    def test_failure_exit_code_and_error_survive_logging(self):
        self.env['FAKE_CARGO_FAIL'] = 'build'
        result = self.command('build.sh')
        self.assertEqual(result.returncode, 41)
        logs = list((self.project / 'reports/local-build').glob('build-*.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('Intentional packaging-test fixture failure', logs[0].read_text())
        self.assertNotIn('Executable:', result.stdout)

    def test_source_launcher_ignores_old_bundle_and_preserves_arguments(self):
        for sub in ['bin', 'target/release']:
            folder = self.project / sub
            folder.mkdir(parents=True)
            old = folder / 'ollama-workbench'
            old.write_text('#!/bin/sh\necho WRONG_OLD_BINARY\nexit 99\n')
            old.chmod(0o755)
        self.env['CARGO_TARGET_DIR'] = '/not/the/checkout'
        args = ['--data-dir', '/tmp/data with spaces', '--check']
        result = self.command('run.sh', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), ['build'])
        self.assertEqual(self.calls()[-1]['target'], str(self.project / 'target'))
        self.assertIn('FRESH_BUILD ' + json.dumps(args), result.stdout)
        self.assertNotIn('WRONG_OLD_BINARY', result.stdout)

    def test_failed_test_gate_never_replaces_installed_executable(self):
        target = self.old_binary()
        previous = target.read_bytes()
        self.env['FAKE_CARGO_FAIL'] = 'test'
        result = self.command('rebuild-and-install.sh')
        self.assertEqual(result.returncode, 41, result.stdout + result.stderr)
        self.assertEqual(target.read_bytes(), previous)
        self.assertFalse(target.with_name('ollama-workbench.previous').exists())
        self.assertFalse((self.home / 'data/applications').exists())
        self.assertNotIn('build', self.commands())

    def test_failed_lint_gate_never_installs_a_stale_target(self):
        target = self.old_binary()
        previous = target.read_bytes()
        self.env['FAKE_CARGO_FAIL'] = 'clippy'
        result = self.command('install.sh')
        self.assertEqual(result.returncode, 41)
        self.assertEqual(target.read_bytes(), previous)
        self.assertIn('FAILED: install', result.stderr)

    def test_successful_fixture_gate_installs_fresh_target_and_backs_up_old(self):
        target = self.old_binary()
        previous = target.read_bytes()
        result = self.command('rebuild-and-install.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('FRESH_BUILD', target.read_text())
        self.assertEqual(target.with_name('ollama-workbench.previous').read_bytes(), previous)
        self.assertTrue(os.access(target, os.X_OK))
        desktop = (self.home / 'data/applications/org.localtools.AIWorkbench.desktop').read_text()
        self.assertIn('Exec="' + str(target) + '"', desktop)
        self.assertIn('Name=AI Workbench', desktop)
        self.assertFalse((self.home / 'data/applications/org.localtools.OllamaWorkbench.desktop').exists())
        self.assertEqual(self.commands(), ['check', 'test', 'test', 'clippy', 'build'])
        self.assertFalse(list((self.home / '.local/bin').glob('.ollama-workbench.*')))

    def test_install_clean_offline_jobs_forward_without_format_mutation(self):
        result = self.command('install.sh', '--clean', '--offline', '--jobs=2')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ['clean', 'check', 'test', 'test', 'clippy', 'build'])
        for call in self.calls():
            if call['args'][0] in ['check', 'test', 'clippy', 'build']:
                self.assertIn('--offline', call['args'])
                self.assertIn('--locked', call['args'])
                self.assertIn('--jobs', call['args'])

    def test_stale_binary_identity_never_replaces_installed_executable(self):
        target = self.old_binary()
        previous = target.read_bytes()
        self.env['FAKE_STALE_IDENTITY'] = 'yes'
        result = self.command('install.sh')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(target.read_bytes(), previous)
        self.assertIn('Refusing unexpected build identity', result.stdout)

    def test_rebuild_and_run_only_starts_after_all_gates(self):
        result = self.command('rebuild-and-run.sh', '--demo')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('FRESH_BUILD ["--demo"]', result.stdout)
        self.assertEqual(self.commands(), ['check', 'test', 'test', 'clippy', 'build'])

    def test_failing_gate_does_not_run_any_binary(self):
        self.env['FAKE_CARGO_FAIL'] = 'check'
        result = self.command('rebuild-and-run.sh')
        self.assertEqual(result.returncode, 41)
        self.assertNotIn('FRESH_BUILD', result.stdout)
        self.assertFalse((self.project / 'target/release/ollama-workbench').exists())

    def test_install_rejects_skip_tests_before_build(self):
        result = self.command('install.sh', '--skip-tests')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_format_is_explicit_and_check_option_does_not_write(self):
        result = self.command('format.sh', '--check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls()[-1]['args'], ['fmt', '--all', '--', '--check'])
        result = self.command('format.sh')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls()[-1]['args'], ['fmt', '--all'])

    def test_verify_can_explicitly_add_format_gate(self):
        self.env['FAKE_CARGO_FAIL'] = 'fmt'
        result = self.command('verify.sh', '--format-check')
        self.assertEqual(result.returncode, 41)
        self.assertEqual(self.commands(), ['fmt'])
        self.assertEqual(self.calls()[-1]['args'], ['fmt', '--all', '--', '--check'])

    def test_check_and_test_scripts_do_not_build_release_or_install(self):
        result = self.command('check.sh', '--offline')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ['check'])
        result = self.command('test.sh', '-j', '2')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ['check', 'test', 'test'])

    def test_help_has_no_cargo_or_clean_side_effects(self):
        for name in SCRIPTS:
            if name in ['run.sh', 'rebuild-and-run.sh']:
                continue
            result = self.command(name, '--help')
            self.assertEqual(result.returncode, 0, name)
            self.assertIn('Usage:', result.stdout)
        self.assertFalse(self.calls())
        self.assertFalse((self.project / 'reports').exists())

class SourceIntegrity(unittest.TestCase):
    def test_all_shell_scripts_parse(self):
        for name in SCRIPTS:
            result = subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, name + ': ' + result.stderr)

    def test_explicit_domain_modules_exist(self):
        text = (ROOT / 'src/lib.rs').read_text()
        modules = re.findall(r'#\[path = "([^"]+)"\]\s*pub mod (\w+);', text)
        self.assertGreaterEqual(len(modules), 36)
        for relative, _name in modules:
            self.assertTrue((ROOT / 'src' / relative).is_file(), relative)
        baseline = json.loads((ROOT / 'scripts/package_baseline.json').read_text())
        self.assertTrue(set(baseline['public_modules']).issubset({name for _, name in modules}))

    def test_split_child_modules_resolve(self):
        for path in (ROOT / 'src').rglob('*.rs'):
            # Explicit path attributes belong to lib.rs, checked above.
            if path.name == 'lib.rs':
                continue
            parent = path.parent if path.name in {'mod.rs', 'main.rs'} else path.with_suffix('')
            for name in re.findall(r'^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;', path.read_text(), re.M):
                self.assertTrue((parent / f'{name}.rs').is_file() or (parent / name / 'mod.rs').is_file(), f'{path}: {name}')

    def test_all_original_regression_test_names_remain(self):
        baseline = json.loads((ROOT / 'scripts/package_baseline.json').read_text())
        current = '\n'.join(path.read_text() for area in ['src', 'tests'] for path in (ROOT / area).rglob('*.rs'))
        for name in baseline['test_names']:
            self.assertRegex(current, rf'fn\s+{re.escape(name)}\s*\(', name)

    def test_pinned_dependency_graph_is_unchanged(self):
        baseline = json.loads((ROOT / 'scripts/package_baseline.json').read_text())
        lock = tomllib.loads((ROOT / 'Cargo.lock').read_text())
        packages = [entry for entry in lock['package'] if entry['name'] != 'ollama-workbench']
        digest = hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, baseline['dependency_packages_sha256'])
        own = next(entry for entry in lock['package'] if entry['name'] == 'ollama-workbench')
        manifest = tomllib.loads((ROOT / 'Cargo.toml').read_text())
        self.assertEqual(own['version'], manifest['package']['version'])
        self.assertEqual(manifest['dependencies']['eframe']['version'], '=0.31.1')

    def test_no_stale_binary_or_known_unsupported_theme_field_is_shipped(self):
        self.assertFalse((ROOT / 'bin/ollama-workbench').exists())
        for path in (ROOT / 'src').rglob('*.rs'):
            self.assertNotIn('.weak_text_alpha', path.read_text(), str(path))


    def test_shared_script_parses_and_wrappers_are_executable(self):
        result = subprocess.run(['bash', '-n', str(ROOT / 'scripts/workbench.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in SCRIPTS + ('scripts/workbench.sh',):
            self.assertTrue(os.access(ROOT / name, os.X_OK), name)

    def test_ui_modules_do_not_import_same_named_backend_as_self(self):
        for name, rel in [('agents', 'src/desktop/advanced'), ('media', 'src/desktop/studio')]:
            text = (ROOT / rel / 'mod.rs').read_text()
            self.assertNotRegex(text, rf'{name}::\{{\s*self[,}}]')
            for path in (ROOT / rel).glob('*.rs'):
                self.assertNotRegex(path.read_text(), rf'(?<![:\w]){name}::[A-Za-z_]')

    def test_model_mutation_event_has_endpoint_and_request_identity(self):
        text = (ROOT / 'src/desktop/app/models.rs').read_text()
        body = re.search(r'WorkEvent::Models\s*\{([^}]+)\}', text).group(1)
        self.assertIn('request_id', body)
        self.assertIn('endpoint', body)
        self.assertIn('let request_id = self.connection_request_id;', text)
        self.assertIn('self.connection_request_id.wrapping_add(1)', text)

    def test_reported_formatter_whitespace_and_stroke_warnings_are_removed(self):
        text = (ROOT / 'src/desktop/app/chat.rs').read_text()
        self.assertFalse(any(line != line.rstrip(' \t') for line in text.splitlines()))
        theme = (ROOT / 'src/desktop/theme.rs').read_text()
        self.assertNotIn('Stroke::new(1.0,', theme)
        self.assertEqual(theme.count('Stroke::new(1.0_f32,'), 2)

    def test_rc7_tool_recovery_sources_are_packaged(self):
        # The turn loop's helpers live beside it since the execution.rs split.
        runtime = (ROOT / 'src/inference/runtime/execution.rs').read_text() + (ROOT / 'src/inference/runtime/turn_helpers.rs').read_text() + (ROOT / 'src/inference/runtime/execution/tests.rs').read_text()
        routing = (ROOT / 'src/inference/routing.rs').read_text()
        protocol = (ROOT / 'src/inference/runtime/tool_protocol.rs').read_text()
        self.assertIn('route_for_task_excluding', runtime)
        self.assertIn('repeated_only_rounds', runtime)
        self.assertIn('"duplicate": true', runtime)
        self.assertIn('pub fn choose_excluding', routing)
        self.assertIn('unwrap_tool_call_tags', protocol)

    def test_rc7_regression_cases_are_present(self):
        source = '\n'.join(
            path.read_text()
            for area in ('src', 'tests')
            for path in (ROOT / area).rglob('*.rs')
        )
        for name in (
            'runtime_protocol_failure_excludes_the_failed_model_for_bounded_failover',
            'repeated_native_reads_are_skipped_and_force_an_early_final_response',
            'successful_write_opens_a_new_epoch_for_a_real_verification_reread',
            'unfinished_report_does_not_count_duplicate_requests_as_successes',
            'singular_unexecuted_tool_warning_uses_correct_grammar',
        ):
            self.assertRegex(source, rf'fn\s+{re.escape(name)}\s*\(', name)

    def test_rc8_mcp_web_search_sources_are_packaged(self):
        manifest = tomllib.loads((ROOT / 'Cargo.toml').read_text())
        version = manifest['package']['version']
        self.assertIn(f'Version {version} ', (ROOT / 'README.md').read_text())
        lock = tomllib.loads((ROOT / 'Cargo.lock').read_text())
        own = next(entry for entry in lock['package'] if entry['name'] == 'ollama-workbench')
        self.assertEqual(own['version'], version)

        mcp = (ROOT / 'src/services/mcp.rs').read_text()
        tools = (ROOT / 'src/workspace/tools/mod.rs').read_text()
        preparation = (ROOT / 'src/workspace/tools/preparation.rs').read_text()
        execution = (ROOT / 'src/workspace/tools/execution.rs').read_text()
        # The turn loop's helpers live beside it since the execution.rs split.
        runtime = (ROOT / 'src/inference/runtime/execution.rs').read_text() + (ROOT / 'src/inference/runtime/turn_helpers.rs').read_text() + (ROOT / 'src/inference/runtime/execution/tests.rs').read_text()
        configuration = (ROOT / 'src/inference/runtime/configuration.rs').read_text()
        diagnostics = (ROOT / 'src/core/diagnostics.rs').read_text()
        preferences = (ROOT / 'src/desktop/app/preferences.rs').read_text()
        settings = (ROOT / 'src/core/settings.rs').read_text()

        self.assertIn('@brave/brave-search-mcp-server@2.1.4', mcp)
        self.assertIn('BRAVE_MCP_ENABLED_TOOLS', mcp)
        self.assertIn('pub web_search: Option<McpWebSearchConfig>', mcp)
        self.assertIn('pub max_query_chars: usize', mcp)
        self.assertIn('pub max_query_words: Option<usize>', mcp)
        self.assertIn('pub max_results: usize', mcp)
        self.assertIn('McpWebSearch', tools)
        self.assertIn('SearchProvider::Mcp', preparation)
        self.assertIn('connected_or_start_mcp', execution)
        self.assertIn('configured_network_engine', configuration)
        self.assertIn('get_chat_engine', (ROOT / 'src/desktop/app/tools.rs').read_text())
        self.assertIn('explicit_web_search_intent', runtime)
        self.assertIn('workspace_tools_enabled', tools)
        self.assertIn('Web search', diagnostics)
        # The Settings page shows an API-key field for every search preset
        # (Brave, Exa, Kagi); each preset names its key variable.
        self.assertIn('SEARCH_PRESETS', preferences)
        self.assertIn('preset.key_env', preferences)
        for key in ('BRAVE_API_KEY', 'EXA_API_KEY', 'KAGI_API_KEY'):
            self.assertIn(key, mcp)
        self.assertIn('TOOL_CAPABILITY_SYSTEM', settings)

    def test_rc8_restricted_engine_names_borrow_from_live_definitions(self):
        tools = (ROOT / 'src/workspace/tools/mod.rs').read_text()
        match = re.search(
            r'fn restricted_engine_hides_and_rejects_workspace_actions\(\) \{(?P<body>.*?)\n    \#\[test\]',
            tools,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group('body')
        self.assertIn('let definitions = e.definitions();', body)
        self.assertRegex(body, r'let names = definitions\s*\.iter\(\)')
        self.assertNotRegex(
            body,
            r'let names = e\s*\.definitions\(\)\s*\.iter\(\)',
        )

    def test_rc8_mcp_web_search_regression_cases_are_present(self):
        source = '\n'.join(
            path.read_text()
            for area in ('src', 'tests')
            for path in (ROOT / area).rglob('*.rs')
        )
        for name in (
            'mcp_web_search_is_first_class_and_reuses_connection',
            'mcp_search_requires_network_opt_in_and_mcp_is_default_provider',
            'restricted_engine_hides_and_rejects_workspace_actions',
            'explicit_web_search_intent_is_narrow_and_honors_opt_outs',
            'web_search_diagnostics_report_configuration_without_launching_it',
            'brave_search_mapping_enforces_provider_query_bounds',
            'older_search_mapping_json_gets_safe_limits',
            'exited_server_is_not_reported_as_connected',
        ):
            self.assertRegex(source, rf'fn\s+{re.escape(name)}\s*\(', name)

if __name__ == '__main__':
    print('Packaging/source-integrity checks only; fixture Cargo is NOT Rust validation.', flush=True)
    unittest.main(verbosity=2)
