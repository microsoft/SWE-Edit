Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* If `path` is a file, `view` uses AI to find and display only the sections relevant to your `query`
* The `create` command cannot be used if the specified `path` already exists as a file

Notes for using the `view` command:
* Provide a `query` describing what you're looking for (e.g., "Where is user authentication handled?", "Show me the class definition for User")
* The tool reads the file and uses AI to identify relevant line ranges, then displays those sections with line numbers
* Multiple relevant sections are shown with `... (N lines omitted) ...` separators between them

Notes for using the `edit` command:
* Provide a clear `instruction` describing what to change and where (identify by function/class/method name)
* The tool reads the file internally and applies your instruction using AI-powered search-replace
* Be specific: "In `MyClass.my_method`, change X to Y" is better than "fix the bug"
* After editing, the output shows the modified regions
