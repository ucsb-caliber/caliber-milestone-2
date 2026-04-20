# variant_gen — how it works

Keep this file roughly in sync when you change routing, prompts, validation, or the OpenRouter client. Tone goal: engineer notes, not a polished wiki.

Updated 2026-04-19

---

## What it’s for

Ingested questions (PDF text or a layout `questions.json`) go through `generator.py`: classify the stem → ask an LLM for a JSON variant → deterministic checks → second LLM to verify. Output is meant to be *usable practice material*, not a guarantee of pedagogical correctness.

---

## Where questions come from

Same JSON shape either way: top-level `ingestions` list.

Default: `exam_tests_questions` walks `exam_tests/*.pdf`, pdfminer text, heuristic split on headers like “Problem 2”, “3. What…”. We normalize `\r`, then replace form-feed `\f` with newline so `(?m)^` splitters see real line starts — without that, College Board–style PDFs glued many items into one blob and everything downstream (format, language, MCQ) lied.

`VARIANT_GEN_QUESTIONS_JSON` or `--db` loads layout output instead. Re-ingesting after splitter changes **changes** `question_id` hashes; `variants.json` dedupes by `original_id`, so stale rows don’t auto-heal.

Tail trim: boilerplate headings like “answer key” / “solutions to…” get cut so solution keys don’t dominate the student prompt.

Diagrams: default PDF ingest sets `image_crops` to `[]` — **no raster**. Vision only runs when upstream JSON fills `image_crops` and `should_use_vision` agrees (diagram-ish words in text, not a pure coding-write stem, vision env on). Generation may attach **one** image; verify is always text-only.

---

## Routing (`question_contract`, `question_inputs`, `question_router`)

Everything the rest of the pipeline needs lives on **`QuestionContract`**: `language`, `mode`, `allow_thematic_reskin`, `question_format` (MCQ / FREE_RESPONSE / TRUE_FALSE), **`expected_mcq_options`** (how strictly we enforce MCQ option count — see below), `routing_source`.

**`route_stem(text)`** is the only entry from `generator.py`. Rules path = `build_question_contract`. Optional **`QUESTION_ROUTER=llm`**: one small JSON call sets `question_format` + `language` only; mode and reskin stay rule-derived so we don’t accidentally enable parking-lot reskins on trace questions. Bad router output → rules fallback (one console line).

**Format (`detect_format`)** order matters: true/false stubs early; “write a function / pseudocode” forces FR before loose MCQ regexes; numbered multi-part worksheets can look like MCQ lines — we bail to FR; `(a)`–`(e)` plus “which of the following” catches AP-style lettered answers that `A.` regexes miss.

**Language** tries Java/AP signals before the loose C++ `int x = …;` heuristic so snippets without `import` don’t land as Python. Stanford-style `VectorNew` / `VectorSplit` + no `std::` → **`generic`** (C-ish answers validate with `{`/`;`). **`_mentions_cpp_as_required_language`** exists because “no C++ whatsoever” still contains the substring `c++` — we only treat C++ as the target language when the mention isn’t negated. Default fallthrough is still **python** for ambiguous prose (Scheme etc.: use `QUESTION_LANG` or the LLM router).

**Mode**: `class_design` vs `conceptual` (explain/compare markers) vs `algorithmic`. Thematic reskin (random domains) only for algorithmic stems that are **not** structural traces and **not** “write a function **named** …” specs — those reskins kept breaking verify. When reskin is off we reuse the same CS-only scenario block as conceptual.

**`expected_mcq_options`**: computed with **`expected_mcq_options_for_stem`** from the same stem text + `question_format` + `language`. `count_options` (line-start `A.` / `1.` patterns) lives in **`question_inputs`** next to `detect_format`. If the stem isn’t MCQ → 0. If MCQ but fewer than two detected lines → assume **4**. If language is **cpp** → **0** (numbered code lines masquerade as options; we skip strict count enforcement). That value is what `is_invalid_variant` uses — no second guessing in `generator`.

---

## Environment (one place)

| Variable | Role |
|----------|------|
| `OPENROUTER_API_KEY` | Required for any LLM call |
| `OPENROUTER_MODEL` | Generate + verify slug default |
| `OPENROUTER_TIMEOUT` | Generate read timeout (s), default 90 |
| `OPENROUTER_TIMEOUT_VERIFY` | Verify read timeout (s), default 55 |
| `OPENROUTER_HTTP_RETRIES` | TLS/connection retries, default 3 |
| `OPENROUTER_VISION` / `OPENROUTER_SEND_IMAGES` | Disable with `0`/`false` to force text-only |
| `GENERATION_SOURCE_MAX_CHARS` | Clip stem for generation prompt |
| `VERIFY_VARIANT_TEXT_MAX_CHARS` | Clip `variant_text` embedded in verify |
| `QUESTION_LANG` | Force `python` / `cpp` / `java` / `generic` for a whole run |
| `QUESTION_ROUTER` | `rules` (default) or `llm` |
| `QUESTION_ROUTER_MODEL` | Optional slug for router (defaults to `OPENROUTER_MODEL`) |
| `QUESTION_ROUTER_TIMEOUT` | Router call timeout (s), default 25 |
| `VARIANT_GEN_TELEMETRY` | `1` / `true` → `[variant_gen:telemetry]` JSON lines |
| `VARIANT_GEN_QUESTIONS_JSON` | Path to questions JSON when not using `--db` |

---

## Prompts (`prompts.py`)

Generation is one long instruction; verify is separate. MCQ verify lists **actual** option keys from the JSON (hardcoding A–E broke real exams). Verify text is head+tail truncated with a middle omission note when huge.

Extras (`prompt_extras`): list/dict/set/vector API nudges when the stem smells like method MCQs; conceptual items must not become coding tasks unless the source asked for code; trace / no-reskin FR stays in CS vocabulary; named-function stems must keep names/types; class_design Python warns that ingest may paste an answer key — keep student text clean, code only in `correct_answer`. Output-only verify hint fires when **variant** text looks like “what is the output / when executed” (reskin that drops the phrase won’t get the hint — known gap).

---

## Validation (`variant_validation.py`)

Placeholder fragments in model output → reject. Similarity gate uses `SequenceMatcher`; explanation-style originals get a **looser** threshold so small rewrites still pass.

FR **`correct_answer`**: empty or rubric phrases (`_META_ANSWER_SNIPPETS`) → reject. **Code-shaped** answer required only when **`_variant_asks_for_code_submission`** (variant blob asks for implementation — name is language-agnostic) **and** **`original_asks_for_code_submission`** (source phrasing) — stops reskins alone from triggering the code gate. Python answers optionally `ast`-checked; mutable default `[]`/`{}` rejected unless the original showed that pattern. Conceptual mode can run **`free_response_cs_vocabulary_lost`** so explain/compare items don’t drift to unrelated domains.

MCQ: need an `options` dict; garbage “all 0.1234” numeric distractors → reject. **`original_asks_for_code_submission`** is also used from `prompts` for conceptual extras.

---

## Python policy (`policies/python_intro.py`)

Linked-list homework (`reverseList`, “linked list”) is excluded from “list method MCQ” heuristics so we don’t confuse nodes with `list.append`. Autofix can pin `correct_answer` when exactly one real (or fake, for “which is NOT”) list method option exists.

---

## OpenRouter (`llm_client.py`)

JSON mode, fence stripping on parse. Verify timeout lower than generate so a stuck verify doesn’t block a long batch. Retries on flaky TLS/connect noise.

---

## Batch runner

Appends to `exam_tests/variants.json`, skips `question_id` already present, atomic save per question, short sleep between calls, passes preloaded `questions_db` so PDFs aren’t re-read every index.

---

## Tests

From `server/`: `python -m unittest variant_gen.tests.test_stem_routing -v` (rules routing; no API key).

---

## Changelog (high level)

Recent work: unified `route_stem` + `question_format` on contract, `expected_mcq_options` on contract, `count_options` moved to `question_inputs`, `_variant_asks_for_code_submission` rename, optional LLM router + telemetry, PDF `\f` fix, Java / negated-C++ / Stanford-C routing, `(a)` MCQ detection, trace + named-function reskin off, verify output hint, OpenRouter retries. Older detail: git history.
