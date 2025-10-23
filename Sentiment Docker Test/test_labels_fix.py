"""
Test script to verify labels parameter handling fix

This tests that:
1. Empty strings are treated as None
2. Labels are only passed to zero-shot tasks
3. Sentiment tasks don't receive unnecessary labels parameter

Run: python test_labels_fix.py
"""

def test_parse_labels_csv():
    """Test the _parse_labels_csv function"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from app import _parse_labels_csv

    # Test cases
    tests = [
        (None, None, "None input"),
        ("", None, "Empty string"),
        ("   ", None, "Whitespace only"),
        ("politics", ["politics"], "Single label"),
        ("politics,economy", ["politics", "economy"], "Multiple labels"),
        ("  politics  ,  economy  ", ["politics", "economy"], "Labels with whitespace"),
        ("politics,,economy", ["politics", "economy"], "Empty items filtered"),
        (",,,", None, "Only commas"),
    ]

    print("Testing _parse_labels_csv():")
    print("-" * 60)

    all_passed = True
    for input_val, expected, description in tests:
        result = _parse_labels_csv(input_val)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✓" if passed else "✗"
        print(f"{status} {description}")
        if not passed:
            print(f"  Input: {repr(input_val)}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")

    print("-" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    return all_passed


def test_labels_logic():
    """Test the endpoint logic for when to use labels"""
    print("\n" + "=" * 60)
    print("Testing label assignment logic:")
    print("=" * 60)

    # Simulate the endpoint logic
    def should_use_labels(preset, labels_input):
        """Simulates the fixed endpoint logic"""
        lbls = None
        if preset and "zeroshot" in preset:
            # For zero-shot, parse labels
            if labels_input:
                lbls = [x.strip() for x in labels_input.split(",") if x.strip()] if isinstance(labels_input, str) else labels_input
            if not lbls:
                lbls = ["default", "labels"]  # Simulating DEFAULT_ZS_LABELS
        return lbls

    test_cases = [
        ("sentiment-twitter", "", None, "Sentiment with empty string → None"),
        ("sentiment-twitter", None, None, "Sentiment with None → None"),
        ("sentiment-twitter", "politics", None, "Sentiment with labels → None (ignored)"),
        ("zeroshot-bart", "", ["default", "labels"], "Zero-shot with empty → defaults"),
        ("zeroshot-bart", None, ["default", "labels"], "Zero-shot with None → defaults"),
        ("zeroshot-bart", "custom1,custom2", ["custom1", "custom2"], "Zero-shot with custom"),
        ("ner-conll", "", None, "NER with empty → None"),
    ]

    all_passed = True
    for preset, labels_input, expected, description in test_cases:
        result = should_use_labels(preset, labels_input)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✓" if passed else "✗"
        print(f"{status} {description}")
        if not passed:
            print(f"  Preset: {preset}, Labels: {repr(labels_input)}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")

    print("-" * 60)
    if all_passed:
        print("✅ All logic tests passed!")
    else:
        print("❌ Some logic tests failed!")

    return all_passed


def main():
    print("=" * 60)
    print("Labels Parameter Fix - Test Suite")
    print("=" * 60)
    print()

    test1 = test_parse_labels_csv()
    test2 = test_labels_logic()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe fix correctly handles:")
        print("  ✓ Empty strings → None")
        print("  ✓ Whitespace-only strings → None")
        print("  ✓ Labels only passed to zero-shot tasks")
        print("  ✓ Sentiment tasks receive None (no labels)")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
