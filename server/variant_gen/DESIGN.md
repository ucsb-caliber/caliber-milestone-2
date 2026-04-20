# variant_gen — design reference

If you change routing, prompts, validation, or the OpenRouter client, update the section you touched so the next person is not reverse‑engineering from git alone. Keep this file in the same voice as now: plain notes, not polished wiki copy or stacked template headings.

Updated 2026-04-19

---

## Purpose and scope

**Decision:** Treat the pipeline as “produce usable exam‑ish variants,” not perfect pedagogy or guaranteed assessment quality.

**Rationale:** Two LLM passes (generate + verify) plus heuristics trade cost and flakiness for throughput. Deterministic checks catch structural failures cheaply; semantic correctness still depends on models.

**Non‑goal:** No guarantee that every PDF layout or every course’s notation is handled without adding rules or turning on `QUESTION_ROUTER=llm`.

---

## End‑to‑end flow (`generator.py`)

**Decision:** One question → classify → build prompts → generate JSON variant → deterministic validation → similarity gate → extra FR answer checks → verify LLM → compare answers (MCQ/TF by normalized label; FR by `claimed_answer_is_correct`) → up to `MAX_RETRIES` attempts with rising temperature.

**Rationale:** Retries (`BASE_TEMPERATURE` + 0.15 per attempt) recover from sampling noise. Validate before verify to avoid paying for verify on obviously bad JSON. Check similarity after `is_invalid_variant` so we do not burn retries on variants that would fail cheaper checks first. MCQ/TF reuse `normalize_answer` so “B” vs “(B)” vs solver prose still lines up when possible.

---

## Question sources (`exam_tests_questions.py`)

**Decision:** Default: walk `exam_tests/*.pdf`, pdfminer extract, heuristic split into question strings, synthesize `ingestions` / `questions` records with stable `question_id` from stem + index + text hash.

**Rationale:** Keeps variant_gen runnable without the layout pipeline. Hash in the id means **re‑ingestion after splitter or extract fixes changes ids** — downstream `variants.json` dedupe is by `original_id`, so old rows do not auto‑update.

**Decision:** After extract, normalize `\r`, replace form feed `\f` with newline, collapse runs of spaces before newlines and excessive blank lines.

**Rationale:** pdfminer emits `\f` between pages. Splitting uses `(?m)^…` patterns; `^` does not match after `\f`, so many numbered items used to merge into one blob (wrong format, wrong language, MCQ chaos). Treating `\f` as a line break restores predictable chunk boundaries.

**Decision:** `_PROBLEM_HEADER` regex prefers “Problem N”, “Question N”, `N)`, and `N.` only when followed by a question‑like opener (long allowlist) and **not** when `N.` looks like a Python list method line (`append`, `union`, …).

**Rationale:** Otherwise code listings and “1. append” lines become false question starts.

**Decision:** Trim trailing boilerplate from “answer key”, “solutions to …”, etc., via `_BOILERPLATE_TAIL`.

**Rationale:** Keeps solution keys out of the student prompt where the heading is unambiguous; avoids model copying full answers into `variant_text`.

**Decision:** `VARIANT_GEN_QUESTIONS_JSON` or an explicit db path loads a fixed JSON with the same top‑level `ingestions` shape (layout export).

**Rationale:** Same generator and batch runner for PDF‑ingested vs layout‑accurate text.

---

## What gets skipped (`question_inputs.should_skip_question`)

**Decision:** Drop stems that look like honor code, integrity policies, “by selecting you agree”, docstring instructions, etc.

**Rationale:** Not gradable items; generating variants would be noise. Batch may still print a slot as failed if the runner does not pre‑filter — acceptable cosmetic.

---

## Vision (`question_inputs.should_use_vision`)

**Decision:** Only when vision env is on, the question has `image_crops`, the stem is not clearly a coding/written‑answer task, and either diagram keywords appear **or** the stem is short and not already MCQ/TF shaped.

**Rationale:** Avoid sending code exercises as fake “diagram” tasks; prefer text. When enabled, `generator` passes **one** crop (first) if the text model supports images.

---

## Coarse algorithm tag (`question_inputs.extract_algorithm`)

**Decision:** Keyword scoring over `_ALGORITHM_PATTERNS` in `config.py`; highest score wins, else `"general problem solving"`.

**Rationale:** Feeds the generation prompt as a soft hint (“preserve this algorithmic structure”). It is not used for validators — wrong tag hurts theme wording more than correctness.

---

## Unified stem routing (`question_contract` + `question_router`)

**Decision:** `QuestionContract` holds `language`, `mode`, `allow_thematic_reskin`, `question_format` (`MCQ` | `FREE_RESPONSE` | `TRUE_FALSE`), and `routing_source`. The generator calls **`route_stem(text)`** only (not separate `detect_format` + contract).

**Rationale:** One object is what prompts and validators read; avoids two classifiers disagreeing on the same stem.

**Decision:** `QUESTION_ROUTER=rules` (default): all fields from heuristics. `QUESTION_ROUTER=llm`: one JSON OpenRouter call fills **`question_format` and `language` only**; `mode` and `allow_thematic_reskin` always come from rules. Bad JSON or invalid enums → full rules fallback with a single console line.

**Rationale:** LLM helps long‑tail PDF phrasing; mode/reskin stay deterministic so trace‑heavy and named‑function stems do not get thematic reskin because the router hallucinated “algorithmic”.

**Env:** `QUESTION_ROUTER_MODEL` (default: same as generate), `QUESTION_ROUTER_TIMEOUT` (default 25s).

---

## Format detection (`question_inputs.detect_format`)

**Decision:** Fixed order: early TF (short stems with true/false); then **coding phrases** (`write a function`, `write pseudocode`, …) → `FREE_RESPONSE` before broad MCQ patterns; then “has MCQ‑shaped lines” combined with “numbered written subproblems” → FR; then generic `A.` / `1.` + word MCQ; then scanned `o A` options; then `(a)`–`(e)` paren style **plus** a “which of the following / choose …” phrase → MCQ; else FR.

**Rationale:** Coding worksheets often have many numbered lines and letters; MCQ regex alone would mislabel them. AP‑style Roman clauses with `(a)`–`(e)` answers need paren counting because `[A-E]\.` does not match `(a)`.

---

## Language inference (`question_contract.infer_programming_language`)

**Decision:** Optional `QUESTION_LANG` env forces `python` | `cpp` | `java` | `generic` for whole‑PDF uniformity.

**Rationale:** Some exams are one language end‑to‑end; heuristics still misfire on mixed PDFs.

**Decision:** Check **Java / AP CS A** signals before loose C++ heuristics (`public class` / `interface`, `implements`, generics `List<`, `instanceof`, `@Override`, `boolean` locals, `private int x;` fields, `String` identifiers, etc., plus classic `import java` / `main`).

**Rationale:** Snippets often lack `import`; a broad `int x = …;` C++ pattern used to steal Java chunks.

**Decision:** Stanford‑style C (`VectorNew`, `VectorSplit`, …) with no `std::` → **`generic`**, not `cpp`.

**Rationale:** Prompts say “language implied by original”; FR validators accept `{`/`;` style C answers. Labeling those stems `cpp` forced C++‑shaped `correct_answer` checks.

**Decision:** Treat substring `c++` as C++ **only** if not immediately preceded by negations (`no`, `not`, `without`, …) in a short window — `_mentions_cpp_as_required_language`.

**Rationale:** Phrases like “no C++ component whatsoever” are pure C; naive `'c++' in text` misrouted to `cpp`.

**Decision:** Python from obvious syntax (`def`, `self.`, `__init__`, …) or explicit “python” / “list method” in text; default fallthrough **python** when nothing else matched (historical intro‑Python bias).

**Rationale:** Last resort for prose‑only stems; may be wrong for Scheme/Racket — use `QUESTION_LANG` or LLM router when that matters.

---

## Mode and thematic reskin (`question_contract`)

**Decision:** `class_design` if markers like `write a class`, `inherits`, `__init__`, …; `conceptual` if explain/compare/define‑style markers; else `algorithmic`.

**Decision:** Thematic reskin (random real‑world themes from `THEMATIC_RESKINS`) only when `algorithmic` **and** not `looks_like_structural_trace_task` **and** not `looks_like_named_function_write_task`.

**Rationale:** Traces and pinned function names break when nouns swap; verifier mismatch spikes. Course names are not baked into detectors — phrases are generic.

**Decision:** When reskin is off but mode is still `algorithmic`, `scenario_from_contract` uses the **same CS‑only scenario dict** as conceptual (no parking lots / recipes).

**Rationale:** Previously every algorithmic item got a random theme; trace‑like items still need a stable CS surface.

---

## Generation prompt (`prompts.build_generation_prompt` + `prompt_extras`)

**Decision:** SWE / conceptual / reskin context blocks differ; always pass `algorithm` tag and `TARGET LANGUAGE` from `language_display`.

**Rationale:** Keeps difficulty and algorithm family explicit; reduces language drift in code.

**Decision:** MCQ block uses `count_options(original)` or defaults to 4; insists same direction (IS vs NOT) and real API names for list/dict/set/vector method questions when detected.

**Rationale:** Flipped “which is NOT” destroys the answer key; fake method names fail list‑method policy.

**Decision:** `prompt_extras` adds conditional blocks: conceptual FR must not become coding unless the source asked for code; multi‑step / no‑reskin FR must keep CS vocabulary and intermediate steps; **named function** stems must keep names/types and put full code in `correct_answer`; class_design Python warns that ingest may include answer keys below the prompt — **student text vs `correct_answer`** split with fenced stubs only; generic code‑in‑stem asks for fenced blocks; OOP Python warns against mutable default args unless the original shows them.

**Rationale:** Each block addresses a observed failure mode (verifier rejects, rubric in answer field, pasted solution in variant text, bad Python idioms).

---

## Verification prompt (`prompts.build_verify_prompt`)

**Decision:** MCQ/TF: solver returns short JSON with `final_answer`; labels for MCQ are taken from **actual option keys** when they look like short tokens (not hardcoded A–E).

**Rationale:** Some exams use extra letters or numeric labels; hardcoding broke verify instructions.

**Decision:** Truncate `variant_text` for verify with head + tail and a middle omission note; separate cap `VERIFY_VARIANT_TEXT_MAX_CHARS`.

**Rationale:** Huge stems blow token limits and JSON escapes; middle omission is rarer loss than head‑only truncation for code + follow‑up text.

**Decision:** Conceptual FR: explicit note that prose‑only claimed answers can be valid.

**Rationale:** Verifiers otherwise penalize explanations for “not being code.”

**Decision:** Non‑conceptual FR: extra bullet that code questions need real code in the claimed answer.

**Rationale:** Pushes model to reject rubric‑only “answers.”

**Decision:** `_verify_output_only_hint`: if **variant** text matches “what is the output / when executed / …” phrases, inject OUTPUT‑ONLY instructions (trim whitespace, brief output OK).

**Rationale:** Verifier was marking correct trace output false for being “too short”; hint targets that false negative. (Variant‑only text means reskins that drop the phrase may miss the hint — known limitation.)

**Decision:** Ask for bounded `reasoning` length in the prompt text.

**Rationale:** Reduces truncated or invalid JSON from runaway chains‑of‑thought.

---

## Deterministic validation (`variant_validation.py`)

**Decision:** Reject placeholder fragments in `variant_text` / `storyline` / `task` (`_BAD_PLACEHOLDER_FRAGMENTS`).

**Rationale:** Models sometimes echo meta‑instructions instead of exam prose.

**Decision:** `is_too_similar`: `SequenceMatcher` ratio over full lowercased strings; threshold `SIMILARITY_THRESHOLD` (0.6) or `SIMILARITY_THRESHOLD_EXPLANATION` (0.72) when the original looks explanation‑heavy.

**Rationale:** Variants that are near‑copies of the source are not useful practice; explanation questions get a looser bar so small rewrites still pass.

**Decision:** FR `correct_answer`: reject empty, reject `_META_ANSWER_SNIPPETS` rubric prose. **Coding shape** required only when both `_variant_asks_for_python_code` (variant/task/constraints blob) **and** `original_asks_for_code_submission` (source phrasing like “write a function”, …) — so reskins cannot alone trigger code requirement.

**Rationale:** Reskins often add “implement …”; without the gate, validators fought trace questions.

**Decision:** `_answer_looks_like_code` is per‑language shallow heuristics (Python `def`/`class`, C++ keywords + `{`/`;`, Java `class` / `public` methods, `generic` allows `{`+`;` or pythonic).

**Rationale:** Fast filter; misses edge cases but stops empty prose “answers” for coding stems.

**Decision:** Conceptual mode only: `free_response_cs_vocabulary_lost` compares CS keyword sets from original vs variant fields; fail if conceptual vocabulary is dropped for novelty themes.

**Rationale:** Extra guard because verifier alone often allowed “parking lot” drift on define/explain items.

**Decision:** `_variant_code_blob_heuristic`: if `variant_text` looks like long unbroken code lines without fences, reject with message to use ``` fences.

**Rationale:** Downstream verify and human readability break on one‑line megablobs.

**Decision:** Python FR: `ast` parse when `def`/`class` present; reject mutable default `[]`/`{}` in answers unless original shows that pattern.

**Rationale:** Bad teaching idiom unless the source uses it.

**Decision:** MCQ: require `options` dict; if `expected_mcq_options >= 2`, enforce exact count; if count > 10 from `count_options`, expect 0 (disabled). Reject options that are all `0.dddd` decimals with no letters (“garbage probabilities”).

**Rationale:** Tree / trace PDFs produced dozens of fake “options”; numeric garbage was a generator failure mode.

**Decision:** `normalize_answer`: TF synonyms; MCQ strip to a letter or digit with mappings 1–5 ↔ A–E; also scan for a lone `\b[A-Z]\b` for noisy solver strings.

**Rationale:** Solver and generator often disagree on surface form but same option.

---

## Python list‑method policy (`policies/python_intro.py`)

**Decision:** `original_suggests_list_method_mcq` requires MCQ‑shaped lines and list‑method wording but **excludes** linked‑list homework (`linked list`, `reverseList` compact form, etc.).

**Rationale:** Those problems are about nodes, not `list.append`.

**Decision:** `list_method_mcq_autofix` can set `correct_answer` when exactly one real (or fake, for “which is NOT”) list method option exists.

**Rationale:** Recovers common generator mistakes before verify.

**Decision:** `validate_list_method_mcq` enforces counts and “NOT a method” direction vs `correct_answer`.

**Rationale:** Catches inconsistent MCQ after autofix edge cases.

---

## OpenRouter client (`llm_client.py`)

**Decision:** JSON in/out via `response_format: json_object`; parse strips optional ``` fences.

**Rationale:** Structured downstream handling; models still occasionally invalid JSON — callers retry or skip.

**Decision:** Separate connect vs read timeouts; verify uses `openrouter_timeout_verify` (lower than generate).

**Rationale:** Verify prompts should fail fast; hung verify should not block batch for minutes.

**Decision:** Retries with exponential backoff on TLS/connection errors (`OPENROUTER_HTTP_RETRIES`).

**Rationale:** Transient `BAD_RECORD_MAC`‑style client noise should not fail a whole batch on first blip.

---

## Batch runner (`generation_batch_runner.py`)

**Decision:** Append to `exam_tests/variants.json`; skip questions whose `question_id` is already in the file; `save_atomic` after each question; optional `time.sleep(0.5)` between calls.

**Rationale:** Resume‑safe progress; atomic write avoids corrupt JSON on crash; sleep reduces rate‑limit risk on OpenRouter.

**Decision:** Pass pre‑loaded `questions_db` into `generate_variant` so PDF/JSON is not re‑parsed per index.

**Rationale:** Performance on multi‑question batches.

---

## Telemetry and operator env (`config.py`)

**Decision:** `VARIANT_GEN_TELEMETRY=1` emits `[variant_gen:telemetry]` JSON lines: stem routing, `verified`, `failed_all_retries`.

**Rationale:** Greppable logs for CI or large batches without parsing human prose.

**Other env (quick index):** `OPENROUTER_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_TIMEOUT`, `OPENROUTER_TIMEOUT_VERIFY`, `OPENROUTER_VISION` / `OPENROUTER_SEND_IMAGES`, `GENERATION_SOURCE_MAX_CHARS`, `VERIFY_VARIANT_TEXT_MAX_CHARS`, `QUESTION_LANG`, `QUESTION_ROUTER`, `QUESTION_ROUTER_MODEL`, `QUESTION_ROUTER_TIMEOUT`, `VARIANT_GEN_TELEMETRY`, `VARIANT_GEN_QUESTIONS_JSON`.

---

## Tests

From `server/`: `python -m unittest variant_gen.tests.test_stem_routing -v` — rules‑only golden stems (no API key).

**Rationale:** Locks in regressions for AP‑style `(a)` MCQ, CS107 C vs “no C++”, and “write a function” vs numbered noise.

---

## Recent changes (rough log)

Details live in git; this list is only orientation.

- 2026-04-19: Unified `route_stem`, `question_format` on contract, optional LLM router, telemetry, stem routing unit tests; prior same‑day work: PDF `\f` → newline, Java heuristics, negated C++, Stanford `Vector*` → `generic`, `(a)` MCQ detection, trace/named‑function reskin off, verify output‑only hint, OpenRouter retries, `DESIGN.md` introduced.
