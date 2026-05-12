# variant_gen — how it works

This file is meant for whoever ships or audits the pipeline later (including a public write-up). Update it when you touch routing, prompts, validation, or the OpenRouter client. It should read like notes from the people who built it, not like marketing copy.

Updated 2026-04-19

---

## What it’s for

Ingested questions—either raw text from **`exam_tests/*.pdf`** or a **`questions.json`-shaped** export—get turned into **variants**: same skill being tested, sometimes with a different skin on the story, always as structured JSON the rest of the app can render.

We’re not building an autograder for arbitrary coursework. The bar is **usable practice material**: good enough to study from, cheap enough to batch overnight, honest about where heuristics lie.

---

## Rough philosophy (why it looks like this)

**Two model calls, not one.** The generator writes `variant_text`, options, and a `correct_answer`. A second pass **re-solves** the problem and checks agreement (MCQ/TF labels) or marks whether the claimed free-response answer is actually right. One-shot “trust my JSON” failed too often—models happily output rubric text as the answer, flip MCQ directions, or agree with themselves on nonsense. A second pass is extra latency and tokens, but it’s the difference between “looks like JSON” and “often right.”

**Rules first, LLM second (mostly).** Stem classification is mostly deterministic: format, language, whether we allow a silly theme, how many MCQ options we enforce. That’s intentional. LLMs are good at paraphrase; they’re mediocre at **consistently** respecting “this PDF says pure C” or “this is a trace, don’t reskin it” across thousands of chunks. We use a small optional router model only for **format + language** when you opt in; mode and reskin stay in code so safety doesn’t depend on router mood.

**Cheap checks before expensive ones.** `generator.py` runs deterministic validation and similarity before it pays for verify. There’s no deep reason beyond cost and annoyance: if the JSON is missing fields or full of placeholder phrases, there’s nothing interesting for the verifier to do.

**One default chat model.** Generate and verify share **`OPENROUTER_MODEL`** unless you override the router slug separately. The old “different model per stage” setup was harder to reproduce and harder to explain in an incident postmortem. You can still split models later if you have evidence one model is systematically better at verify—nothing in the code forbids it.

**Retries bump temperature slightly.** If verify disagrees or the JSON is invalid, we retry a few times with a bit more sampling noise. Not sophisticated; it just unsticks local minima without hand-tuning per exam.

---

## Where questions come from

Same JSON shape either way: top-level `ingestions` list.

Default: `exam_tests_questions` walks `exam_tests/*.pdf`, pdfminer text, heuristic split on headers like “Problem 2”, “3. What…”. We normalize `\r`, then replace form-feed `\f` with newline so `(?m)^` splitters see real line starts. That one’s worth spelling out: pdfminer was giving us page breaks as `\f`, and regexes anchored at line start never fired—so a whole exam page could end up as **one** “question,” and every classifier downstream guessed wrong.

`VARIANT_GEN_QUESTIONS_JSON` or `--db` loads layout output instead. Re-ingesting after splitter changes **changes** `question_id` hashes; `variants.json` dedupes by `original_id`, so stale rows don’t auto-heal.

Tail trim: boilerplate headings like “answer key” / “solutions to…” get cut so solution keys don’t dominate the student prompt.

**Diagrams.** Default PDF ingest sets `image_crops` to `[]`—there is no “figure” in the pipeline unless something else fills that field. If layout JSON attaches crops **and** the stem looks diagram-ish (and isn’t obviously “write this function”), generation can send **one** image to OpenRouter when vision is enabled. Verify stays **text-only** on purpose: we didn’t want to double multimodal cost on every question, and we’re implicitly betting that a good variant **restates** anything essential from the figure in words. That bet is wrong for some exams; it’s a known trade.

---

## Routing (`question_contract`, `question_inputs`, `question_router`)

Everything downstream reads a **`QuestionContract`**: `language`, `mode`, `allow_thematic_reskin`, **`question_format`** (`MCQ` | `FREE_RESPONSE` | `CODING` | `TRUE_FALSE`), **`expected_mcq_options`**, **`autograde_kind`**, `routing_source`.

**Formats:** **`FREE_RESPONSE`** is prose, traces, short answers, and **fill-in-the-blank / one-hole code** (still code-shaped answers sometimes, but not the “full implementation” autograder path). **`CODING`** is routed only when ``stem_routes_as_coding_format`` says so: same broad “asks for code” phrases as validators, **minus** obvious blank/fill-in stems (see ``question_inputs.py``). **`autograde_kind`** matches **`question_format`**.

**`route_stem(text)`** is the only entry from `generator.py`. Rules path = `build_question_contract`. Optional **`QUESTION_ROUTER=llm`**: one small JSON call sets `question_format` + `language` only; mode and reskin stay rule-derived so we don’t accidentally enable parking-lot reskins on trace questions. Bad router output → rules fallback (one console line).

**Format (`detect_format`)** order matters: true/false stubs early; **`stem_routes_as_coding_format`** (substantial implementation, not fill-in-blank) → **`CODING`** before MCQ heuristics; numbered multi-part worksheets can look like MCQ lines—bail to **FREE_RESPONSE**; `(a)`–`(e)` plus “which of the following” catches AP-style lettered answers that `A.` regexes miss.

**Language** tries Java/AP signals before the loose C++ `int x = …;` heuristic so snippets without `import` don’t land as Python. Stanford-style `VectorNew` / `VectorSplit` + no `std::` → **`generic`** so we don’t demand C++-shaped answers for C. **`_mentions_cpp_as_required_language`** exists because “no C++ whatsoever” still contains the substring `c++`. Default fallthrough is still **python** for ambiguous prose (Scheme, etc.): that’s lazy but fixable with `QUESTION_LANG` or the router.

**Mode**: `class_design` vs `conceptual` vs `algorithmic`. Thematic reskin only for algorithmic stems that aren’t structural traces and aren’t “write a function **named** …”—we burned a lot of time on verify failures before we turned that off. When reskin is off we reuse the same CS-only scenario block as conceptual.

**`expected_mcq_options`**: from **`expected_mcq_options_for_stem`** using the same stem + format + language. `count_options` lives in **`question_inputs`**. Not MCQ → 0. MCQ but fewer than two detected lines → assume **4**. Language **cpp** → **0** because numbered code lines masquerade as options. Same number feeds **`is_invalid_variant`** so we don’t compute it twice in different places.

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

Generation is one long instruction; verify is separate. MCQ verify lists **actual** option keys from the JSON—hardcoding A–E broke real exams. Verify text is head+tail truncated with a middle omission note when huge; otherwise prompts blow past context limits and you get truncated JSON back.

Extras (`prompt_extras`): nudges for list/dict/set/vector method MCQs; conceptual items must not become coding tasks unless the source asked for code; trace / no-reskin FR stays in CS vocabulary; named-function stems must keep names/types; class_design Python warns that ingest may paste an answer key—keep student text clean, code only in `correct_answer`. The output-only verify hint keys off **variant** text (“what is the output…”). If a reskin drops that wording, the hint doesn’t fire—that’s logged as a gap, not ignored.

---

## Validation (`variant_validation.py`)

Placeholder fragments in model output → reject. Similarity gate uses `SequenceMatcher`; explanation-style originals get a looser threshold so a light rewrite doesn’t get thrown out.

FR **`correct_answer`**: empty or rubric phrases → reject. **Code-shaped** answer required only when **`_variant_asks_for_code_submission`** *and* **`original_asks_for_code_submission`**—we got burned when reskins added “implement a function” and suddenly the validator demanded code for a trace question. Python answers optionally `ast`-checked; mutable default `[]`/`{}` rejected unless the original showed that pattern. Conceptual mode can run **`free_response_cs_vocabulary_lost`** so explain/compare items don’t drift into unrelated domains.

MCQ: need an `options` dict; garbage “all 0.1234” numeric distractors → reject.

---

## Python policy (`policies/python_intro.py`)

Linked-list homework (`reverseList`, “linked list”) is excluded from “list method MCQ” heuristics so we don’t confuse nodes with `list.append`. Autofix can pin `correct_answer` when exactly one real (or fake, for “which is NOT”) list method option exists.

---

## OpenRouter (`llm_client.py`)

JSON mode, fence stripping on parse. Verify timeout is lower than generate so one bad prompt doesn’t hold the whole batch hostage. Retries on flaky TLS/connect noise—that was a practical fix for laptops on bad Wi‑Fi, not a theoretical nicety.

---

## Batch runner

Appends to `exam_tests/variants.json`, skips `question_id` already present, atomic save per question, short sleep between calls, passes preloaded `questions_db` so PDFs aren’t re-read every index.

---

## Tests

From `server/`: `python -m unittest variant_gen.tests.test_stem_routing -v` (rules routing; no API key).

---

## What we’re explicitly not doing (yet)

Perfect PDF understanding without layout help. Guaranteed semantic correctness on every variant. Full multimodal verify. Automatic repair of every kind of OCR garbage. If you need those, you’ll extend this layer—or replace the stem router with something heavier—and this doc should get a paragraph when you do.
