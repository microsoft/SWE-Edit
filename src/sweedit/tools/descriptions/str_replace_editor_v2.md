Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* If `path` is a file, `view` uses AI to find and display only the sections relevant to your `query`
* The `create` command cannot be used if the specified `path` already exists as a file

Notes for using the `view` command:
* Provide a `query` describing what you're looking for (e.g., "Where is user authentication handled?", "Show me the class definition for User")
* The tool reads the file and uses AI to identify relevant line ranges, then displays those sections with line numbers
* Multiple relevant sections are shown with `... (N lines omitted) ...` separators between them

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
