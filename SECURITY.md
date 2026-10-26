# Security and data handling

This project is intended for a trusted user on a local machine. It executes model-generated Python, shell commands, and ROOT macros with the current user's permissions. It also loads serialized model checkpoints using mechanisms capable of executing code. Prompt instructions, restricted calculator builtins, and tool allowlists are not a sandbox.

Use a disposable VM or isolated environment without valuable credentials or unrelated private files. Keep the FastAPI listener on loopback. There is no authentication, production-grade HTML sanitization, per-user isolation, upload limit, or protection suitable for public hosting.

Gemini receives prompts and tool outputs, which may include dataset samples and local paths. Search services receive search queries. Generated scripts can access files and network resources available to your account. Review your data-sharing permissions before use.

`.env`, local environments, `outputs/`, `data/`, and common model/data binaries are ignored by Git. Other files, including CSVs outside those directories, still require manual review before staging. Outputs may contain input copies, sensitive logs, model weights, and code. If a credential is accidentally published, revoke or rotate it immediately; deleting a file does not remove Git history.

For a suspected vulnerability, contact the repository owner privately through contact information on their GitHub profile if available. Do not post credentials or private datasets in public issues.
