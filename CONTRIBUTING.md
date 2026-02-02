# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## PR Guidelines

### Linting
We use pre-commit hooks to enforce code style and linting with `ruff`. Run `pre-commit install` before committing any changes, this will enable pre-commit hooks to run automatically on every git commit!

You can also run `pre-commit run --all-files` to run the hooks manually.

### PR Titles
We follow the Conventional Commits specification for PR titles. This helps us maintain a clean and readable git history.

**Format:**
```
<type>(<scope>): <description>
```

**Core types:**
- feat: - New feature
- fix: - Bug fix
- docs: - Documentation changes
- style: - Code style changes (formatting, semicolons, etc.)
- refactor: - Code changes that neither fix bugs nor add features
- perf: - Performance improvements
- test: - Adding or updating tests
- chore: - Maintenance tasks (dependencies, build config, etc.)

**Scope (optional):**
The scope should specify the area of the codebase affected

**Description:**

- Use imperative, present tense
- Don't capitalize the first letter
- No period (.) at the end
- Keep it concise but descriptive

**Examples:**
- ✅ feat(tools): add task management tool
- ✅ docs: update installation guide with uv


## Questions?
Feel free to open an issue or reach me at ykzhang@cs.wisc.edu if you have any questions or suggestions!
