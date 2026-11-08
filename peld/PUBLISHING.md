# Publishing pELD

1. Choose and add a project `LICENSE`, then declare it in `pyproject.toml`.
2. Reserve or verify the `peld` project on PyPI.
3. Update the version in `pyproject.toml` and `src/peld/__init__.py` together.
4. Run the local checks:

   ```bash
   python -m pip install -e '.[test,publish]'
   pytest
   python -m build
   python -m twine check dist/*
   ```

5. Test the artifacts in a fresh environment:

   ```bash
   python -m venv /tmp/peld-release-test
   /tmp/peld-release-test/bin/python -m pip install dist/*.whl
   /tmp/peld-release-test/bin/python -c 'import peld; print(peld.__version__)'
   ```

6. Publish to TestPyPI first, or configure the included GitHub Actions workflow
   as a PyPI Trusted Publisher. The publisher must match the final GitHub owner,
   repository, workflow filename, and `pypi` environment exactly.

7. Publish a GitHub release whose tag begins with `peld-v`, for example
   `peld-v0.1.0`. Other repository releases, including checkpoint releases, do
   not trigger the package publisher.

Do not store PyPI tokens in the repository. Trusted Publishing uses short-lived
OIDC credentials and is preferred for the release workflow.
