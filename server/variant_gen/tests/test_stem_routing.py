"""Golden checks for rule-based stem routing (no OpenRouter calls)."""

import unittest


class TestStemRoutingRules(unittest.TestCase):
    def test_ap_style_paren_mcq(self):
        from variant_gen.question_router import route_stem

        stem = """Which of the following satisfies the interface?
public class Circle implements Shape { }
  I.  public int foo(Shape x)
 II.  public int foo(Circle x)
(a)  I only
(b)  II only
(c)  I and II
(d)  None
(e)  All
"""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "MCQ")
        self.assertEqual(c.routing_source, "rules")
        self.assertEqual(c.language, "java")
        self.assertGreaterEqual(c.expected_mcq_options, 2)

    def test_cs107_negated_cpp_is_generic(self):
        from variant_gen.question_router import route_stem

        stem = """Problem 1: Matchmaking
Write a function generateAllCouples that returns a new vector.
No C++ component whatsoever. It's all C.
VectorNew(&boys, sizeof(char *), StringFree, 0);
vector generateAllCouples(vector *boys, vector *girls) {
  vector couples;
"""
        c = route_stem(stem)
        self.assertEqual(c.language, "generic")
        self.assertEqual(c.routing_source, "rules")

    def test_write_function_defaults_fr_not_mcq_numbered_noise(self):
        from variant_gen.question_router import route_stem

        stem = """Write a function longestPrefix that takes two lists of strings.
1. First consider empty lists.
2. Then scan characters left to right.
"""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "CODING")
        self.assertEqual(c.autograde_kind, "CODING")

    def test_autograde_kind_prose_free_response(self):
        from variant_gen.question_router import route_stem

        stem = """Explain why immutable strings can be used safely as dict keys.
Give at least two sentences comparing tuples and lists."""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "FREE_RESPONSE")
        self.assertEqual(c.autograde_kind, "FREE_RESPONSE")

    def test_autograde_kind_mcq(self):
        from variant_gen.question_router import route_stem

        stem = """Which of the following is true?
(a) foo (b) bar (c) baz (d) qux (e) none"""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "MCQ")
        self.assertEqual(c.autograde_kind, "MCQ")

    def test_trace_output_is_free_response_not_coding(self):
        from variant_gen.question_router import route_stem

        stem = """What is the output when the following code is executed?
public class Main {
  public static void main(String[] args) {
    System.out.println(3 + 4);
  }
}"""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "FREE_RESPONSE")
        self.assertEqual(c.autograde_kind, "FREE_RESPONSE")

    def test_fill_in_blank_stays_free_response(self):
        from variant_gen.question_router import route_stem

        stem = """Fill in the missing expression. Complete the following code so the method returns the sum.

    public int f(int[] a) {
      return _______;
    }"""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "FREE_RESPONSE")
        self.assertEqual(c.autograde_kind, "FREE_RESPONSE")

    def test_complete_method_without_blank_is_coding(self):
        from variant_gen.question_router import route_stem

        stem = """Complete the following code. Implement the body of binarySearch so it returns the index
or -1. You may not change the signature."""
        c = route_stem(stem)
        self.assertEqual(c.question_format, "CODING")

    def test_contract_has_question_format(self):
        from variant_gen.question_contract import build_question_contract

        c = build_question_contract("Which is true? (a) foo (b) bar")
        self.assertTrue(hasattr(c, "question_format"))
        self.assertEqual(c.routing_source, "rules")


if __name__ == "__main__":
    unittest.main()
