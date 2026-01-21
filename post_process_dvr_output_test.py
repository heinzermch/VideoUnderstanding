#!/usr/bin/env python3
"""
Test suite for post_process_dvr_output.py

Run with: python3 post_process_dvr_output_test.py
Or with unittest: python3 -m unittest post_process_dvr_output_test
"""

import unittest
import sys
import os

# Add the current directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from post_process_dvr_output import (
    clean_current_ampere,
    clean_home_point_distance,
    validate_home_point_distance_sequence
)


class TestCleanCurrentAmpere(unittest.TestCase):
    """Test cases for clean_current_ampere function"""
    
    def test_valid_cases(self):
        """Test valid input cases"""
        test_cases = [
            ("1.52", "1.52", "Single dot, no 'A' unit"),
            ("1.40 A", "1.40", "Single dot with 'A' unit"),
            ("3\n4. 27\n5. 35\n6. 15.75", "15.75", "Has '6.' followed by decimal number"),
            ("6. 3.52 A", "3.52", "Has '6.' prefix with decimal and 'A'"),
            ("6. 3.52", "3.52", "Has '6.' prefix with decimal, no 'A'"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_current_ampere(input_val)
                self.assertEqual(result, expected, 
                               f"Failed for {description}: expected {expected}, got {result}")
    
    def test_invalid_cases(self):
        """Test invalid input cases"""
        test_cases = [
            ("33", None, "No dot, no 'A' unit"),
            ("9\n4. 26\n5. 607\n6. 3", None, "Has '6.' but value after is just '3' (no decimal)"),
            ("", None, "Empty string"),
            (None, None, "None value"),
            ("LAT 46.97, LON 8.58", None, "Coordinates (should be skipped)"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_current_ampere(input_val)
                self.assertEqual(result, expected,
                               f"Failed for {description}: expected {expected}, got {result}")


class TestCleanHomePointDistance(unittest.TestCase):
    """Test cases for clean_home_point_distance function"""
    
    def test_valid_km_conversions(self):
        """Test valid KM values converted to meters"""
        test_cases = [
            ("1.70 KM", "1700.0", "Valid: decimal value with KM -> meters"),
            ("1.70 km", "1700.0", "Valid: lowercase km -> meters"),
            ("0.5 KM", "500.0", "Valid: decimal less than 1 -> meters"),
            ("999.99 KM", "999990.0", "Valid: large decimal value -> meters"),
            ("9999.99 KM", "9999990.0", "Valid: large value at limit -> meters"),
            ("1.42KM", "1420.0", "Valid: no space between number and KM -> meters"),
            ("1.70KM", "1700.0", "Valid: no space, decimal value -> meters"),
            ("0.5KM", "500.0", "Valid: no space, decimal less than 1 -> meters"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_home_point_distance(input_val)
                self.assertEqual(result, expected,
                               f"Failed for {description}: expected {expected}, got {result}")
    
    def test_valid_m_values(self):
        """Test valid M values (already in meters)"""
        test_cases = [
            ("37 M", "37.0", "Valid: integer value with M -> meters"),
            ("123 M", "123.0", "Valid: three digit integer with M -> meters"),
            ("37 m", "37.0", "Valid: lowercase m -> meters"),
            ("1234 M", "1234.0", "Valid: more than 3 digits -> meters"),
            ("37M", "37.0", "Valid: no space between number and M -> meters"),
            ("123M", "123.0", "Valid: no space, three digits -> meters"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_home_point_distance(input_val)
                self.assertEqual(result, expected,
                               f"Failed for {description}: expected {expected}, got {result}")
    
    def test_invalid_cases(self):
        """Test invalid input cases"""
        test_cases = [
            ("1.70", None, "Invalid: no unit"),
            ("KM", None, "Invalid: no value"),
            ("M", None, "Invalid: no value"),
            ("10000 KM", None, "Invalid: too large (> 9999.99)"),
            ("10000KM", None, "Invalid: too large, no space (> 9999.99)"),
            ("1.70 MI", None, "Invalid: wrong unit (MI instead of M/KM)"),
            ("1.70MI", None, "Invalid: wrong unit, no space (MI instead of M/KM)"),
            ("1.70 K", None, "Invalid: wrong unit (K instead of KM)"),
            ("1.70K", None, "Invalid: wrong unit, no space (K instead of KM)"),
            ("", None, "Empty string"),
            (None, None, "None value"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_home_point_distance(input_val)
                self.assertEqual(result, expected,
                               f"Failed for {description}: expected {expected}, got {result}")
    
    def test_multiline_and_text_extraction(self):
        """Test multiline and text extraction cases"""
        test_cases = [
            ("1.70 KM\nSome other text", "1700.0", "Multiline: should use first line -> meters"),
            ("Some text 1.70 KM more text", "1700.0", "Text before and after: should extract -> meters"),
            ("1.70 KM and 2.50 M", "1700.0", "Multiple matches: should extract first -> meters"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_home_point_distance(input_val)
                self.assertEqual(result, expected,
                               f"Failed for {description}: expected {expected}, got {result}")
    
    def test_question_number_prefix_format(self):
        """Test the '9. ' prefix format from actual CSV data"""
        # Based on actual sample from dvr_ocr_results CSV
        test_cases = [
            ("9. 0 M", "0.0", "Question 9 format: 0 M -> meters"),
            ("9. 1.70 KM", "1700.0", "Question 9 format: 1.70 KM -> meters"),
            ("9. 37 M", "37.0", "Question 9 format: 37 M -> meters"),
            ("9. 111M", "111.0", "Question 9 format: 111M (no space) -> meters"),
            ("9. 1.42KM", "1420.0", "Question 9 format: 1.42KM (no space) -> meters"),
            ("9. 0M", "0.0", "Question 9 format: 0M (no space) -> meters"),
            # Actual multiline sample from CSV
            ("2874412\n2. 19\n3. 0.1\n4. 0\n5. 9\n6. 1.52\n7. 4.08 V\n8. 28°C\n9. 0 M", 
             "0.0", "Actual CSV multiline format with 9. 0 M"),
            # Another multiline example
            ("Some text\n9. 1.5 KM\nMore text", 
             "1500.0", "Multiline with 9. prefix -> meters"),
            # Multiline with no space
            ("Some text\n9. 111M\nMore text", 
             "111.0", "Multiline with 9. prefix, no space -> meters"),
        ]
        
        for input_val, expected, description in test_cases:
            with self.subTest(input=input_val, description=description):
                result = clean_home_point_distance(input_val)
                self.assertEqual(result, expected,
                               f"Failed for {description}: expected {expected}, got {result}")


class TestValidateHomePointDistanceSequence(unittest.TestCase):
    """Test cases for validate_home_point_distance_sequence function"""
    
    def test_values_below_25_meters_kept_as_is(self):
        """Test that values < 25 meters are kept as-is unless >50% drop from >= 25m"""
        # Test case: All values < 25, should all be kept
        series = pd.Series([0.0, 10.0, 20.0, 24.9])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([0.0, 10.0, 20.0, 24.9])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 22 -> 11 (both < 25m, should be kept)
        series = pd.Series([22.0, 11.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([22.0, 11.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 25 -> 10 (60% drop from >= 25m to < 25m, should be filtered)
        series = pd.Series([25.0, 10.0, 15.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([25.0, None, 15.0])  # 10 is >50% drop from 25
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 1000 -> 20 (98% drop from >= 25m to < 25m, should be filtered)
        series = pd.Series([1000.0, 20.0, 15.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([1000.0, None, 15.0])  # 20 is >50% drop from 1000
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> 60 (40% drop from >= 25m to < 25m, should be kept)
        series = pd.Series([100.0, 60.0, 50.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, 60.0, 50.0])  # 60 is only 40% drop, should be kept
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 25 -> 12.5 (exactly 50% drop, should be kept - threshold is >50%)
        series = pd.Series([25.0, 12.5, 10.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([25.0, 12.5, 10.0])  # 12.5 is exactly 50% drop, should be kept
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_percentage_change_validation(self):
        """Test that values >= 25m with >50% jump (bigger or smaller) are set to None"""
        # Test case: 100 -> 160 (60% increase, should be None)
        series = pd.Series([100.0, 160.0, 150.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, 150.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> 149 (49% increase, should be valid)
        series = pd.Series([100.0, 149.0, 150.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, 149.0, 150.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> 50 (50% decrease, should be valid - exactly 50%)
        series = pd.Series([100.0, 50.0, 60.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, 50.0, 60.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> 49 (51% decrease, should be None)
        series = pd.Series([100.0, 49.0, 60.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, 60.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> 40 (60% decrease, should be None)
        series = pd.Series([100.0, 40.0, 60.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, 60.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_handles_none_values(self):
        """Test that None values don't reset the last valid value"""
        # Test case: 100 -> None -> 160 (should compare 160 to 100, not None)
        series = pd.Series([100.0, None, 160.0, 150.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, None, 150.0])  # 160 is >50% jump from 100
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> None -> 149 (should compare 149 to 100, should be valid)
        series = pd.Series([100.0, None, 149.0, 150.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, 149.0, 150.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_more_than_three_consecutive_none_allows_change(self):
        """Test that after >3 consecutive None values, accept next value even if >50% change"""
        # Test case: 100 -> None -> None -> None -> None -> 200 (4 consecutive None, should accept 200)
        series = pd.Series([100.0, None, None, None, None, 200.0, 210.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, None, None, None, 200.0, 210.0])  # 200 accepted after 4 None
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> None -> None -> None -> 200 (exactly 3 None, should reject 200)
        series = pd.Series([100.0, None, None, None, 200.0, 210.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, None, None, None, 210.0])  # 200 rejected (only 3 None)
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> None -> None -> None -> None -> None -> 40 (5 consecutive None, should accept 40 even though it's 60% decrease)
        series = pd.Series([100.0, None, None, None, None, None, 40.0, 50.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, None, None, None, None, 40.0, 50.0])  # 40 accepted after 5 None
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> 10 (<25m, kept) -> 15 (<25m, kept) -> 20 (<25m, kept) -> 5 (<25m, kept) -> 200 (values <25m don't count as None, so 200 compared to 100, >50% change, should be None)
        series = pd.Series([100.0, 10.0, 15.0, 20.0, 5.0, 200.0, 210.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, 10.0, 15.0, 20.0, 5.0, None, 210.0])  # 200 rejected (compared to 100, >50% change)
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test case: 100 -> None -> None -> None -> None -> 200 (4 consecutive None, should accept 200)
        series = pd.Series([100.0, None, None, None, None, 200.0, 210.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, None, None, None, 200.0, 210.0])  # 200 accepted after 4 None
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_handles_string_values(self):
        """Test that string numeric values are converted and validated"""
        series = pd.Series(["100.0", "149.0", "160.0"])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, 149.0, None])  # 160 is >50% jump from 149
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_invalid_values_set_to_none(self):
        """Test that invalid values (non-numeric) are set to None"""
        series = pd.Series([100.0, "invalid", 150.0])
        result = validate_home_point_distance_sequence(series)
        expected = pd.Series([100.0, None, 150.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)


def run_tests():
    """Run all tests and return success status"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCleanCurrentAmpere))
    suite.addTests(loader.loadTestsFromTestCase(TestCleanHomePointDistance))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateHomePointDistanceSequence))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running test suite for post_process_dvr_output.py")
    print("=" * 60)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 60)
    if success:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)
