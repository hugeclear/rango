"""
Unit Tests for Strict Output Format Utilities
=============================================

Tests for the strict output format validation and repair functionality.
"""

import sys
import unittest
from pathlib import Path

# Add utils path for imports
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "verification" / "utils"))

from strict_output import (
    extract_strict_answer, 
    format_repair_prompt, 
    json_default, 
    StrictOutputValidator,
    DEFAULT_ANSWER_PATTERN
)


class TestStrictOutputExtraction(unittest.TestCase):
    """Test strict answer extraction functionality."""
    
    def setUp(self):
        self.pattern = DEFAULT_ANSWER_PATTERN
        self.allowed_labels = ['action', 'comedy', 'drama', 'sci-fi', 'horror']
    
    def test_extract_valid_answer(self):
        """Test extraction of valid Answer: format."""
        # Test case: "Answer: Foo_Bar" should be extracted correctly
        text = "Answer: action"
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertTrue(success)
        self.assertEqual(answer, "action")
    
    def test_extract_with_underscores_hyphens(self):
        """Test extraction with underscores and hyphens."""
        text = "Answer: sci-fi"
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertTrue(success)
        self.assertEqual(answer, "sci-fi")
    
    def test_extract_with_surrounding_text(self):
        """Test extraction from text with surrounding content."""
        # Test case: "Here is the result.\nAnswer: X\nThanks" → extraction OK
        text = "Here is the result.\nAnswer: comedy\nThanks for your question."
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertTrue(success)
        self.assertEqual(answer, "comedy")
    
    def test_extract_multiline_context(self):
        """Test extraction from multiline context."""
        text = """Based on the movie description provided, I need to classify this film.
        
Answer: drama

This classification is based on the emotional themes present."""
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertTrue(success)
        self.assertEqual(answer, "drama")
    
    def test_extract_no_match(self):
        """Test extraction failure when no pattern matches."""
        # Test case: "Some text" → False
        text = "Some random text without the required format"
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertFalse(success)
        self.assertEqual(answer, "")
    
    def test_extract_wrong_format(self):
        """Test extraction failure with wrong format."""
        text = "The answer is comedy"  # Wrong format
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertFalse(success)
        self.assertEqual(answer, "")
    
    def test_extract_invalid_label(self):
        """Test extraction failure with invalid label."""
        text = "Answer: invalid_genre"  # Not in allowed list
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertFalse(success)
        self.assertEqual(answer, "")
    
    def test_extract_no_allowed_labels(self):
        """Test extraction without label validation."""
        text = "Answer: any_label"
        answer, success = extract_strict_answer(text, self.pattern, None)  # No validation
        
        self.assertTrue(success)
        self.assertEqual(answer, "any_label")
    
    def test_extract_case_sensitive(self):
        """Test case-sensitive label matching."""
        text = "Answer: ACTION"  # Wrong case
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertFalse(success)  # Should fail due to case mismatch
        self.assertEqual(answer, "")
    
    def test_extract_whitespace_handling(self):
        """Test whitespace handling in extraction."""
        text = "Answer:   comedy   "  # Extra whitespace
        answer, success = extract_strict_answer(text, self.pattern, self.allowed_labels)
        
        self.assertTrue(success)
        self.assertEqual(answer, "comedy")


class TestFormatRepairPrompt(unittest.TestCase):
    """Test format repair prompt generation."""
    
    def test_repair_prompt_basic(self):
        """Test basic repair prompt generation."""
        base_prompt = "Classify this movie"
        enhanced_prompt = format_repair_prompt(base_prompt)
        
        self.assertIn("絶対に次の形式のみで返答してください", enhanced_prompt)
        self.assertIn("`Answer: <label>`", enhanced_prompt)
        self.assertIn("前後に一切の説明文を付けないこと", enhanced_prompt)
    
    def test_repair_prompt_with_allowed_labels(self):
        """Test repair prompt with allowed labels list."""
        base_prompt = "Classify this movie"
        allowed_labels = ['action', 'comedy', 'drama']
        enhanced_prompt = format_repair_prompt(base_prompt, allowed_labels)
        
        self.assertIn("ラベルは", enhanced_prompt)
        self.assertIn("'action'", enhanced_prompt)
        self.assertIn("'comedy'", enhanced_prompt)
        self.assertIn("'drama'", enhanced_prompt)
    
    def test_repair_prompt_preserves_base(self):
        """Test that repair prompt preserves base prompt content."""
        base_prompt = "Original classification instruction"
        enhanced_prompt = format_repair_prompt(base_prompt)
        
        self.assertIn("Original classification instruction", enhanced_prompt)


class TestJSONDefault(unittest.TestCase):
    """Test JSON serialization helper."""
    
    def test_numpy_bool_conversion(self):
        """Test numpy boolean conversion."""
        import numpy as np
        
        numpy_bool = np.bool_(True)
        result = json_default(numpy_bool)
        
        self.assertIsInstance(result, bool)
        self.assertEqual(result, True)
    
    def test_numpy_int_conversion(self):
        """Test numpy integer conversion."""
        import numpy as np
        
        numpy_int = np.int64(42)
        result = json_default(numpy_int)
        
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)
    
    def test_numpy_float_conversion(self):
        """Test numpy float conversion."""
        import numpy as np
        
        numpy_float = np.float64(3.14)
        result = json_default(numpy_float)
        
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=2)
    
    def test_numpy_array_conversion(self):
        """Test numpy array conversion."""
        import numpy as np
        
        numpy_array = np.array([1, 2, 3])
        result = json_default(numpy_array)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])
    
    def test_dict_object_conversion(self):
        """Test object with __dict__ conversion."""
        class TestObject:
            def __init__(self):
                self.value = 42
                self.name = "test"
        
        obj = TestObject()
        result = json_default(obj)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['value'], 42)
        self.assertEqual(result['name'], "test")
    
    def test_string_fallback(self):
        """Test string fallback for unknown types."""
        class UnknownType:
            def __str__(self):
                return "unknown_object"
        
        obj = UnknownType()
        result = json_default(obj)
        
        # The function returns __dict__ if available, then string fallback
        expected = obj.__dict__ if hasattr(obj, '__dict__') else "unknown_object"
        self.assertEqual(result, expected)


class TestStrictOutputValidator(unittest.TestCase):
    """Test StrictOutputValidator class."""
    
    def setUp(self):
        self.pattern = DEFAULT_ANSWER_PATTERN
        self.allowed_labels = ['action', 'comedy', 'drama']
        self.validator = StrictOutputValidator(self.pattern, self.allowed_labels)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.pattern, self.pattern)
        self.assertEqual(self.validator.allowed_labels, self.allowed_labels)
        self.assertEqual(self.validator.total_attempts, 0)
        self.assertEqual(self.validator.successful_extractions, 0)
    
    def test_validation_success(self):
        """Test successful validation."""
        text = "Answer: action"
        answer, is_valid = self.validator.validate(text)
        
        self.assertTrue(is_valid)
        self.assertEqual(answer, "action")
        self.assertEqual(self.validator.total_attempts, 1)
        self.assertEqual(self.validator.successful_extractions, 1)
    
    def test_validation_failure(self):
        """Test validation failure."""
        text = "Wrong format"
        answer, is_valid = self.validator.validate(text)
        
        self.assertFalse(is_valid)
        self.assertEqual(answer, "")
        self.assertEqual(self.validator.total_attempts, 1)
        self.assertEqual(self.validator.successful_extractions, 0)
    
    def test_compliance_rate_calculation(self):
        """Test compliance rate calculation."""
        # Start with empty validator
        self.assertEqual(self.validator.get_compliance_rate(), 0.0)
        
        # Add successful validation
        self.validator.validate("Answer: action")
        self.assertEqual(self.validator.get_compliance_rate(), 1.0)
        
        # Add failed validation
        self.validator.validate("Wrong format")
        self.assertEqual(self.validator.get_compliance_rate(), 0.5)
        
        # Add another success
        self.validator.validate("Answer: comedy")
        self.assertAlmostEqual(self.validator.get_compliance_rate(), 2/3, places=3)
    
    def test_statistics_reporting(self):
        """Test statistics reporting."""
        # Validate some texts
        self.validator.validate("Answer: action")
        self.validator.validate("Wrong format")
        self.validator.validate("Answer: comedy")
        
        stats = self.validator.get_stats()
        
        self.assertEqual(stats['format_compliance_total'], 3)
        self.assertEqual(stats['format_compliance_ok'], 2)
        self.assertAlmostEqual(stats['format_compliance_rate'], 2/3, places=3)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios with realistic use cases."""
    
    def test_movie_classification_scenario(self):
        """Test realistic movie classification scenario."""
        pattern = DEFAULT_ANSWER_PATTERN
        allowed_labels = ['action', 'adventure', 'animation', 'comedy', 'crime', 
                         'drama', 'family', 'fantasy', 'horror', 'mystery', 
                         'romance', 'sci-fi', 'thriller', 'western']
        
        validator = StrictOutputValidator(pattern, allowed_labels)
        
        # Test various realistic model outputs
        test_cases = [
            ("Answer: sci-fi", True, "sci-fi"),  # Perfect format
            ("Answer: action", True, "action"),  # Perfect format
            ("The movie is a sci-fi film.", False, ""),  # Wrong format
            ("Answer: science-fiction", False, ""),  # Wrong label
            ("Answer: sci-fi\nBased on the futuristic setting.", True, "sci-fi"),  # Multiline - should extract first match
            ("I think the answer is:\nAnswer: comedy", True, "comedy"),  # Mixed format but extractable
        ]
        
        for text, expected_valid, expected_answer in test_cases:
            answer, is_valid = validator.validate(text)
            self.assertEqual(is_valid, expected_valid, f"Failed for text: '{text}'")
            self.assertEqual(answer, expected_answer, f"Wrong answer for text: '{text}'")
    
    def test_format_repair_effectiveness(self):
        """Test format repair prompt effectiveness simulation."""
        base_prompt = "Classify this movie into one of the provided genres"
        allowed_labels = ['action', 'comedy', 'drama']
        
        # Generate repair prompt
        repair_prompt = format_repair_prompt(base_prompt, allowed_labels)
        
        # Verify repair prompt contains all necessary elements
        self.assertIn(base_prompt, repair_prompt)
        self.assertIn("絶対に次の形式のみで返答してください", repair_prompt)
        self.assertIn("Answer: <label>", repair_prompt)
        self.assertIn("action", repair_prompt)
        self.assertIn("comedy", repair_prompt)
        self.assertIn("drama", repair_prompt)


if __name__ == '__main__':
    # Set up test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStrictOutputExtraction,
        TestFormatRepairPrompt,
        TestJSONDefault,
        TestStrictOutputValidator,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, trace in result.failures:
            print(f"  {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, trace in result.errors:
            print(f"  {test}: {trace.split('Error:')[-1].strip()}")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    print(f"\n{'✅ ALL TESTS PASSED' if exit_code == 0 else '❌ SOME TESTS FAILED'}")
    sys.exit(exit_code)