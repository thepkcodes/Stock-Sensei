"""
Unit tests for app_utils.py
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app_utils import (
    format_large_number,
    validate_ticker,
    format_timeframe,
    create_metric_card
)

class TestAppUtils(unittest.TestCase):
    
    def test_format_large_number(self):
        """Test large number formatting"""
        self.assertEqual(format_large_number(999), "$999.00")
        self.assertEqual(format_large_number(1500), "$1.50K")
        self.assertEqual(format_large_number(1500000), "$1.50M")
        self.assertEqual(format_large_number(2500000000), "$2.50B")
    
    def test_validate_ticker(self):
        """Test ticker validation"""
        # Valid tickers
        self.assertEqual(validate_ticker("aapl"), "AAPL")
        self.assertEqual(validate_ticker("MSFT"), "MSFT")
        self.assertEqual(validate_ticker(" TSLA "), "TSLA")
        
        # Invalid tickers
        with self.assertRaises(ValueError):
            validate_ticker("")
        
        with self.assertRaises(ValueError):
            validate_ticker("123")
        
        with self.assertRaises(ValueError):
            validate_ticker("TOOLONG")
        
        with self.assertRaises(ValueError):
            validate_ticker("AA-PL")
    
    def test_format_timeframe(self):
        """Test timeframe formatting"""
        self.assertEqual(format_timeframe(1), "1 day")
        self.assertEqual(format_timeframe(3), "3 days")
        self.assertEqual(format_timeframe(7), "1 week")
        self.assertEqual(format_timeframe(14), "2 weeks")
        self.assertEqual(format_timeframe(30), "1 month")
        self.assertEqual(format_timeframe(90), "3 months")
    
    def test_create_metric_card(self):
        """Test metric card creation"""
        # Basic metric
        card = create_metric_card("Price", "$150.00")
        self.assertIn("Price", card)
        self.assertIn("$150.00", card)
        
        # With positive delta
        card = create_metric_card("Price", "$150.00", delta=5.5)
        self.assertIn("↑", card)
        self.assertIn("5.50%", card)
        self.assertIn("green", card)
        
        # With negative delta
        card = create_metric_card("Price", "$150.00", delta=-3.2)
        self.assertIn("↓", card)
        self.assertIn("3.20%", card)
        self.assertIn("red", card)
        
        # With zero delta
        card = create_metric_card("Price", "$150.00", delta=0)
        self.assertIn("→", card)
        self.assertIn("gray", card)

if __name__ == '__main__':
    unittest.main()
