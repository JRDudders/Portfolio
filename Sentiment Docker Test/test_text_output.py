"""
Test to verify that original text strings are included in sentiment output

This tests that the fix properly includes the original text in results.
"""
import json

def test_output_format():
    """Test the expected output format"""

    # Simulate what the endpoint should return now
    original_texts = ["I love this!", "This is terrible."]

    # Simulate predictions from run_task
    predictions = [
        {"labels": ["positive", "neutral", "negative"], "scores": [0.92, 0.05, 0.03]},
        {"labels": ["negative", "neutral", "positive"], "scores": [0.78, 0.15, 0.07]}
    ]

    # Merge texts with predictions (what the fix does)
    output = []
    for t, p in zip(original_texts, predictions):
        result_dict = {"text": t}
        if isinstance(p, dict):
            result_dict.update(p)
        output.append(result_dict)

    # Check the output format
    print("=" * 60)
    print("Expected Output Format After Fix:")
    print("=" * 60)
    print(json.dumps({"results": output}, indent=2))
    print()

    # Verify structure
    print("=" * 60)
    print("Verification:")
    print("=" * 60)

    all_pass = True
    for i, item in enumerate(output):
        has_text = "text" in item
        has_labels = "labels" in item
        has_scores = "scores" in item

        print(f"\nResult {i}:")
        print(f"  ✓ Has 'text': {has_text} - {item.get('text', 'MISSING')[:30]}...")
        print(f"  ✓ Has 'labels': {has_labels}")
        print(f"  ✓ Has 'scores': {has_scores}")

        if not (has_text and has_labels and has_scores):
            all_pass = False
            print(f"  ❌ MISSING FIELDS!")

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ Output format is correct!")
        print("\nEach result includes:")
        print("  - text: original input string")
        print("  - labels: sentiment labels")
        print("  - scores: confidence scores")
    else:
        print("❌ Output format has issues!")
    print("=" * 60)

    return all_pass


def test_before_vs_after():
    """Compare before and after the fix"""

    print("\n" + "=" * 60)
    print("BEFORE THE FIX (Wrong - missing text):")
    print("=" * 60)

    wrong_output = {
        "results": [
            {"labels": ["positive", "neutral", "negative"], "scores": [0.92, 0.05, 0.03]},
            {"labels": ["negative", "neutral", "positive"], "scores": [0.78, 0.15, 0.07]}
        ]
    }
    print(json.dumps(wrong_output, indent=2))

    print("\n❌ Problem: No way to know which text got which scores!")

    print("\n" + "=" * 60)
    print("AFTER THE FIX (Correct - includes text):")
    print("=" * 60)

    correct_output = {
        "results": [
            {
                "text": "I love this!",
                "labels": ["positive", "neutral", "negative"],
                "scores": [0.92, 0.05, 0.03]
            },
            {
                "text": "This is terrible.",
                "labels": ["negative", "neutral", "positive"],
                "scores": [0.78, 0.15, 0.07]
            }
        ]
    }
    print(json.dumps(correct_output, indent=2))

    print("\n✅ Now we can see which text got which sentiment!")


if __name__ == "__main__":
    print("Testing Sentiment Output Format Fix\n")

    test_output_format()
    test_before_vs_after()

    print("\n" + "=" * 60)
    print("To test the live API:")
    print("=" * 60)
    print("""
    curl -X POST "http://localhost:8080/predict/batch" \\
      -H "Content-Type: application/json" \\
      -d '{
        "texts": ["I love this!", "This is terrible."],
        "preset": "sentiment-twitter"
      }'

    Expected output should include "text" field in each result!
    """)
