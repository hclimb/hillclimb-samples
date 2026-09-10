# Implementation note — guidelines

What each `wiki/implementations/*.md` note should cover. These are guidelines, not a rigid
form — include what's relevant, drop what isn't, keep it terse and specific. Lead with the
outcome: a colleague should grasp what changed and why without reading to the end.

A good note makes the following clear:

- **What changed, in a sentence or two** — put it first, so the note is skimmable.
- **Motivation & context** — the problem or limitation that prompted the change, and what was
  true before. Link the reference page(s) that describe the prior state.
- **Options weighed & tradeoffs** — the alternatives considered and why the chosen one won.
  Record what you rejected and the one-line reason; that reasoning is exactly what a reference
  page omits, and the main reason this note exists.
- **How it was built & integrated** — the key files/functions (use the codebase's `::`
  notation), where it hooks into the existing flow, what stayed unchanged, and any new config
  keys / env vars with their default and effect.
- **Reference pages updated** — which `wiki/` reference page(s) you brought up to date in the
  same change (and if none were needed, why).
- **Tests** — the test file(s) and the exact command (`uv run python tests/test_<foo>.py`), with
  the actual pass/fail output pasted in — the real numbers, not "it passed".
- **Follow-ups & risks** — known gaps, deferred work, things to watch. "None" is a fine answer.

Head the note with its **date, author, status** (in progress / done / superseded by …), and the
**commit or PR** it landed in.
