"""English coding suite: problems with hidden unit tests, ordered by difficulty."""

PROMPT_TEMPLATE = (
    "Write a Python 3 function with the signature `{signature}`.\n\n{description}\n\n"
    "Return only the complete function (and any imports it needs) in a single ```python code block, "
    "without explanations or example usage."
)

PROBLEMS = [
    {
        "id": "C1_palindrome",
        "func": "is_palindrome",
        "signature": "is_palindrome(s: str) -> bool",
        "description": "Return True if the string reads the same forwards and backwards, considering only "
                       "alphanumeric characters and ignoring case.",
        "cases": [
            [["A man, a plan, a canal: Panama"], True],
            [["race a car"], False],
            [[""], True],
            [[" "], True],
            [["0P"], False],
            [["No 'x' in Nixon"], True],
            [["ab_a"], True],
        ],
    },
    {
        "id": "C2_merge_intervals",
        "func": "merge_intervals",
        "signature": "merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        "description": "Merge all overlapping intervals and return the result sorted by start. Intervals that "
                       "touch (end of one equals start of another) are considered overlapping. The input may be "
                       "unsorted or empty.",
        "cases": [
            [[[[1, 3], [2, 6], [8, 10], [15, 18]]], [[1, 6], [8, 10], [15, 18]]],
            [[[[1, 4], [4, 5]]], [[1, 5]]],
            [[[]], []],
            [[[[5, 7], [1, 2]]], [[1, 2], [5, 7]]],
            [[[[1, 10], [2, 3], [4, 5]]], [[1, 10]]],
            [[[[2, 3], [1, 2]]], [[1, 3]]],
        ],
    },
    {
        "id": "C3_int_to_roman",
        "func": "int_to_roman",
        "signature": "int_to_roman(n: int) -> str",
        "description": "Convert an integer from 1 to 3999 to a Roman numeral using standard subtractive notation "
                       "(IV, IX, XL, XC, CD, CM).",
        "cases": [
            [[3], "III"], [[4], "IV"], [[9], "IX"], [[58], "LVIII"],
            [[444], "CDXLIV"], [[1994], "MCMXCIV"], [[3999], "MMMCMXCIX"],
        ],
    },
    {
        "id": "C4_top_k_words",
        "func": "top_k_frequent_words",
        "signature": "top_k_frequent_words(words: list[str], k: int) -> list[str]",
        "description": "Return the k most frequent words sorted by frequency from highest to lowest. Words with "
                       "the same frequency are sorted alphabetically.",
        "cases": [
            [[["i", "love", "leetcode", "i", "love", "coding"], 2], ["i", "love"]],
            [[["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], 4],
             ["the", "is", "sunny", "day"]],
            [[["b", "a", "c"], 2], ["a", "b"]],
            [[["a"], 1], ["a"]],
            [[["x", "y", "x", "y", "z"], 3], ["x", "y", "z"]],
        ],
    },
    {
        "id": "C5_decode_string",
        "func": "decode_string",
        "signature": "decode_string(s: str) -> str",
        "description": "Decode a string encoded with the rule k[encoded], where the encoded part inside the "
                       "brackets is repeated exactly k times (k is a positive integer, possibly with several "
                       "digits). Encodings can be nested. Characters outside brackets are kept as is.",
        "cases": [
            [["3[a]2[bc]"], "aaabcbc"],
            [["3[a2[c]]"], "accaccacc"],
            [["2[abc]3[cd]ef"], "abcabccdcdcdef"],
            [["abc"], "abc"],
            [["10[a]"], "aaaaaaaaaa"],
            [["2[b3[a]]c"], "baaabaaac"],
        ],
    },
    {
        "id": "C6_longest_valid_parens",
        "func": "longest_valid_parentheses",
        "signature": "longest_valid_parentheses(s: str) -> int",
        "description": "Given a string containing only '(' and ')', return the length of the longest "
                       "well-formed (valid) parentheses substring.",
        "cases": [
            [["(()"], 2], [[")()())"], 4], [[""], 0], [["()(()"], 2],
            [["()(())"], 6], [["(()())"], 6], [["))(("], 0], [["(()))())("], 4],
        ],
    },
    {
        "id": "C7_min_window",
        "func": "min_window",
        "signature": "min_window(s: str, t: str) -> str",
        "description": "Return the shortest substring of s that contains every character of t, including "
                       "duplicates. If there are several shortest substrings, return the leftmost one. If no "
                       "such substring exists, return an empty string.",
        "cases": [
            [["ADOBECODEBANC", "ABC"], "BANC"],
            [["a", "a"], "a"],
            [["a", "aa"], ""],
            [["ab", "b"], "b"],
            [["bba", "ab"], "ba"],
            [["acbbaca", "aba"], "baca"],
            [["abcab", "ab"], "ab"],
        ],
    },
    {
        "id": "C8_calculator",
        "func": "calculate",
        "signature": "calculate(expression: str) -> int",
        "description": "Evaluate an arithmetic expression with non-negative integer literals, the binary "
                       "operators +, -, *, /, unary minus, parentheses and arbitrary spaces. Use the usual "
                       "precedence (* and / before + and -, left to right). Division is integer division that "
                       "truncates toward zero and is applied at each step, e.g. 7 / -2 = -3 and (7 / 2) * 2 = 6. "
                       "Do not use eval or exec.",
        "cases": [
            [["1 + 1"], 2],
            [["3+2*2"], 7],
            [[" 14-3/2 "], 13],
            [["(1+(4+5+2)-3)+(6+8)"], 23],
            [["-(2+3)*4"], -20],
            [["10 - 2 * -3"], 16],
            [["7/-2"], -3],
            [["-7/2"], -3],
            [["(7/2)*2"], 6],
            [["2*(5+5*2)/3+(6/2+8)"], 21],
            [["- - 3"], 3],
        ],
    },
]


def prompt(problem):
    return PROMPT_TEMPLATE.format(signature=problem["signature"], description=problem["description"])
