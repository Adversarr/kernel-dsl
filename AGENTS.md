# AGENTS.md

This repository is a skill-building workspace, not a typical application/library repo.

## Repository Layout

- `./3rdparty/` stores the upstream package source trees that we build skills for.
- `./skills/` stores the installable skills we author in this repo.
- Each skill lives under `./skills/<skill-name>/`.

## Authoring Rules

- Treat `./3rdparty/` as source material for understanding, extracting, and packaging knowledge into a skill.
- Treat each directory under `./skills/` as an independent artifact.
- A skill directory such as `./skills/some-skill/` must not reference, import, include, or cite files outside that directory.
- Do not rely on sibling skills, root-level helper files, or files under `./3rdparty/` at install time.
- If a skill needs examples, docs, snippets, metadata, or other assets, copy or rewrite the needed material into that skill's own directory.

## Practical Implication

When building or updating a skill, assume installation only includes the contents of that skill's directory. If something is required for the skill to work, it must live inside `./skills/<skill-name>/`.
