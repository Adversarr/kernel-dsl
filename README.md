# Kernel DSL

Kernel DSL is a collection of skills to write and improve deep learning operators in DSLs such as TileLang, Triton, and CuTeDSL.
It is distilled from many existing repos, see [Acknowledgements](#acknowledgements).

Usage: copy `skills/*` into your agents' `skills` folder. (e.g. `<your_repo>/.claude/skills/`, `<your_repo>/.agents/skills/`)

## Develop

We use `trial/<some_dsl>` to validate our skill can actually work as expected, and improve the skill documents if possible.

## TODO

Languages
- [x] tilelang-wiki
- [ ] triton
- [ ] CuTeDSL

Harness
- [x] Profiler (torch)
- [x] Profiler (ncu)

Documents
- [x] KernelWiki from mit-han-lab.

Misc
- [ ] Shorten the skill documents.

# Acknowledgements

- [KernelWiki](https://github.com/mit-han-lab/KernelWiki/)
- [kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)
- [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill/tree/main)

Built with Codex and TRAE.