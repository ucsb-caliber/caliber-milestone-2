# variant_gen — notes on how this works

If you change routing, prompts, validation, or the OpenRouter client, skim this file and update the bit you touched so the next person isn’t reverse‑engineering from git alone. When you edit this file itself, keep it in the same voice as now—plain notes, not polished wiki copy or stacked template headings.

Updated 2026-04-19

---

## What this package is for

Ingested questions (PDF text or a layout `questions.json`) go through `generator.py`: classify → generate JSON variant → sanity checks → second LLM pass to verify answers. The goal is usable exam-ish output, not perfect pedagogy.

---

## Where questions come from (`exam_tests_questions.py`)

Same top-level shape either way: a dict with an `ingestions` list.

By default we walk `exam_tests/*.pdf`, run pdfminer, split on problem-header-ish regexes, and synthesize question records (empty crops/pages are normal). `ap.pdf`-style College Board extracts used to glue many numbered questions into one record because page breaks were `\f` and the splitter’s `^` patterns never fired; we turn `\f` into `\n` right after extract so `2.` / `3.` / `7.` each start their own chunk.

`VARIANT_GEN_QUESTIONS_JSON` or an explicit db path swaps in real layout output. The batch runner doesn’t care which path you used.

---

## `QuestionContract` (`question_contract.py`)

One object carries `language`, `mode`, `allow_thematic_reskin`, and `routing_source` so prompts and validators don’t each re-parse the stem.

Modes are coarse: `class_design` (OOP write-a-class stuff), `conceptual` (explain/compare), `algorithmic` (everything else that isn’t those).

`infer_programming_language` used to miss a lot of AP CS A–style Java (only import/main/java.util). We also look for `public class` / `interface`, `implements`, generics like `List<`, `boolean` locals, `private int x;`-style fields, `String` identifiers, etc., **before** the broad C++ `int x = …;` heuristic so chunks don’t fall through to Python or C++ by accident. Whole-PDF override: `QUESTION_LANG=java` in `.env` if a file is uniformly one language.

Thematic reskin (random “finance / sports / …” themes) is only on for algorithmic items that pass two cheap filters: not a multi-step structure trace, and not a pinned “write a function named …” spec. `looks_like_structural_trace_task()` watches for phrases like “after each”, “step by step”, “trace the”, “starting from” plus insert/delete/operation, etc. `looks_like_named_function_write_task()` catches “write a function named/called …” (regex-normalized spaces) so we don’t reskin specs that fall apart in verify. Still no course names baked in.

`scenario_from_contract` used to ignore `allow_thematic_reskin` for algorithmic questions (everything got a random theme anyway). Now if reskin is off we reuse the same CS-only scenario block as conceptual mode.

---

## Format (`question_inputs.py`)

`detect_format` runs in a fixed order so we don’t label a coding question as MCQ just because the PDF has `1.` lines or lettered bullets later. Coding phrases (`write a function`, pseudocode, …) and multi-part “for each / worst-case runtime” worksheets bail out to free response before the generic MCQ regex wins.

---

## Prompts (`prompts.py`)

Generation builds one big instruction; verification is a separate prompt. Verify prompts truncate long `variant_text` (env `VERIFY_VARIANT_TEXT_MAX_CHARS`) and ask for short reasoning so we don’t get 20k‑char JSON blowups.

MCQ verify tells the model which option labels exist (from the `options` dict keys). Hardcoding A–E broke when stems used more letters.

If the **variant text** looks like “what is the output / when executed” style, we inject an extra paragraph telling the verifier to treat brief literal output as OK and only fail on substantive mismatch — cuts down bogus `claimed_answer_is_correct: false` on trace questions.

---

## OpenRouter (`llm_client.py`, `config.py`)

Generate and verify use different read timeouts (`OPENROUTER_TIMEOUT`, `OPENROUTER_TIMEOUT_VERIFY`). Transient TLS / connection failures retry a few times with backoff (`OPENROUTER_HTTP_RETRIES`, default 3). That’s for `BAD_RECORD_MAC`‑style noise on flaky networks, not for fixing bad prompts.

---

## Validation (`variant_validation.py`)

`count_options` only looks for option-looking lines at line starts. If it counts more than ten it returns zero so we skip “must have exactly N options” — traversal dumps used to look like 40+ fake options.

`normalize_answer` allows A–Z for MCQ labels.

---

## Python intro policy (`policies/python_intro.py`)

“List method MCQ” detection ignores linked-list wording so `reverseList` / “linked list” doesn’t trip the Python `list.append` rules.

---

## Generator (`generator.py`)

Long PDF blobs get clipped before the generation prompt (`GENERATION_SOURCE_MAX_CHARS`). MCQ + C++ sometimes forces `expected_mcq_options = 0` because numbered code lines broke option counting.

---

## Odds and ends

`should_skip_question` drops honor-code blobs on purpose; the batch script may still print “Failed” for that slot — cosmetic only.

---

## Recent changes (rough log)

- 2026-04-19 (later): `exam_tests_questions` replaces PDF form-feed (`\\f`) with newline after extract so numbered AP-style questions don’t merge into one blob.
- 2026-04-19 (later): Broader Java / AP-style detection in `infer_programming_language` so snippets without `import` don’t get labeled Python or C++ by mistake.
- 2026-04-19 (later): “write a function named …” → no thematic reskin; verify prompt extra for output-only questions; small generation extra for named-function FR.
- 2026-04-19: Structural trace → no thematic reskin; scenario selection respects `allow_thematic_reskin`; extra FR prompt line for multi-step stems; OpenRouter retries on TLS/connection errors; this file added.
- Same day (earlier commits): dynamic MCQ verify labels, option counting / answer normalization, verify truncation and timeouts, question_inputs and C++ / list-method tweaks — details in git if you need them.
