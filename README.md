# README
Please keep this updated as you add things.  We can fill in details about the project later but keep it updated with project structure and any code features (uv, formatter, file structure, ...)

## Environment Manager: `uv`
`uv` is a nice js-like package and environment manager for python.  Note the following common commands:
Sync / install deps (like `npm i`): `uv sync`
Create a virtual environment at path/to/.venv: `uv venv path/to/.venv` (activate as normal, note to deactivate conda if you do use this)
Pretend you are using pip: `uv pip xxx`
Add `requests` as a dependency: `uv add requests`
Run the script `myscript.py`: `uv run myscript.py`

And more: https://mathspp.com/blog/uv-cheatsheet

## Pre-commit black formatter 
I like black formatter, which is rare since I don't like formatters.  It may cause some unexpected behavior when you go to commit code, but it is for the best to keep code clean.  Would recommend installing the vscode extension as well.

## File Structure 
Anything not directly applicable to current work should not be in the root -- put it in a folder corresponding to the paper it was used for or an otherwise informative name. 

@neskech can add more here since he made the models folder but what i have in mind are losses, `models`, `metrics`, `train`, and `test` folders along with a script to run everything at root.
