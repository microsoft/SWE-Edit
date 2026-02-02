Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* Large file views will be truncated, showing only the first and last 250 lines

Notes for using the `edit` command:
* Provide a clear `instruction` describing what to change and where (identify by function/class/method name)
* The tool reads the file internally and applies your instruction using AI-powered search-replace
* Be specific: "In `MyClass.my_method`, change X to Y" is better than "fix the bug"
* After editing, the output shows the modified regions
