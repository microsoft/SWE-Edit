Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.

### Command Execution
* **Non-persistent**: Each shell tool call is executed in a fresh environment. Shell variables, working directory changes, and history are NOT preserved between calls.
* **Timeout**: Commands have a default timeout of 120 seconds (max 300). Set the `timeout` parameter for long-running commands.
* **One command at a time**: Chain multiple commands using `&&` (conditional), `;` (sequential), or `||` (on failure).

### Long-running Commands
* For commands that may run indefinitely (e.g., servers), run in background: `python3 app.py > server.log 2>&1 &`
* For potentially long commands (installations, tests), set an appropriate `timeout` value.

### Best Practices
* **Avoid large outputs**: Commands producing massive output may be truncated.
* **Directory verification**: Verify parent directories exist before creating/editing files.

### Output Handling
* Stdout and stderr are combined and returned as a string. Output may be truncated if too long.
* Exit codes are provided in system tags for failed commands.
* Timeout messages are returned if commands exceed the timeout limit.
