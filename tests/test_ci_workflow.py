"""CI-01/CI-02 contract: checks exposed to pull requests and branch protection.

CI is also checked with a YAML parser and GitHub Actions after publication.
"""
import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml'


class CIWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WORKFLOW.read_text(encoding='utf-8')

    def test_push_and_pull_request_triggers_cover_modernization_and_main(self):
        self.assertRegex(self.source, r'(?m)^on:\s*$')
        self.assertIn('push:', self.source)
        self.assertIn('pull_request:', self.source)
        self.assertGreaterEqual(self.source.count('feature/hand-d-v2-modernization'), 2)
        self.assertGreaterEqual(self.source.count('main'), 2)
        self.assertNotIn('pull_request_target:', self.source)

    def test_three_independent_jobs_and_one_stable_merge_gate(self):
        for job in ('python', 'frontend', 'rust', 'required'):
            self.assertRegex(self.source, rf'(?m)^  {job}:\s*$')
        self.assertIn('name: Required CI', self.source)
        self.assertIn('needs: [python, frontend, rust]', self.source)
        self.assertRegex(self.source, r'(?m)^    if: \$\{\{ always\(\) \}\}$')
        for job in ('python', 'frontend', 'rust'):
            self.assertIn(f'needs.{job}.result', self.source)

    def test_lockfiles_and_offline_contract_suites_are_required(self):
        for command in (
            'uv lock --check',
            'uv sync --frozen --group training',
            "python -m unittest discover -s tests -p 'test_*.py' -q",
            'npm ci --prefix desktop',
            'npm --prefix desktop run check',
            'node --test desktop/tests/*.test.mjs',
            'cargo test --manifest-path desktop/src-tauri/Cargo.toml --locked -q',
        ):
            self.assertIn(command, self.source)

    def test_pinned_actions_use_node24_capable_major_versions(self):
        # Checkout v5 and setup-node v5 no longer require the deprecated Node 20 runner.
        self.assertIn('actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09', self.source)
        self.assertIn('actions/setup-node@a0853c24544627f65ddf259abe73b1d18a591444', self.source)
        self.assertNotIn('actions/checkout@11d5960a326750d5838078e36cf38b85af677262', self.source)
        used = re.findall(r'(?m)^\s+- uses: ([^\s#]+)', self.source)
        self.assertTrue(used)
        self.assertTrue(all(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}', value)
                            for value in used), 'Every action must be pinned to a commit SHA')

    def test_pr_jobs_are_read_only_and_cancel_superseded_runs(self):
        self.assertRegex(self.source, r'(?m)^permissions:\s*\n  contents: read\s*$')
        self.assertNotRegex(self.source, r'(?m)^\s+(?:contents|actions|checks|pull-requests): write\s*$')
        self.assertIn('cancel-in-progress: true', self.source)
        self.assertNotIn('secrets.', self.source)
        self.assertIn('persist-credentials: false', self.source)
        self.assertRegex(self.source, r'(?m)^    runs-on: ubuntu-24\.04\s*$')


if __name__ == '__main__':
    unittest.main()
