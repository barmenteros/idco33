"""
MQL5 Prompt Detector
Task D19: Implement is_mql5_prompt(text: str) -> bool
Module: PromptDetector

Detects whether a given text prompt is related to MQL5 programming
using regex patterns and keyword matching.
"""

import re
from typing import Set, List


class MQL5PromptDetector:
    """
    Detects MQL5-related prompts using regex patterns and keyword matching.
    Optimized for fast detection with comprehensive MQL5 coverage.
    """
    
    def __init__(self):
        # Core MQL5 keywords - functions, types, and concepts (MQL5 ONLY)
        self.mql5_keywords: Set[str] = {
            # Core MQL5 identifier (removed mql4, mt4 references)
            'mql5', 'metatrader5', 'mt5',
            
            # Essential MQL5 functions
            'ontick', 'oninit', 'ondeinit', 'onstart', 'oncalculate',
            'ontester', 'onchartproperty', 'onbookproperty',
            
            # Array functions
            'arrayresize', 'arraysize', 'arraycopy', 'arrayinitialize',
            'arrayfill', 'arraysetasseries', 'arraygetasseries',
            'arraysort', 'arraybsearch', 'arrayminimum', 'arraymaximum',
            
            # Trading functions
            'ordersend', 'orderselect', 'orderclose', 'ordermodify',
            'orderdelete', 'ordershistorytotal', 'orderstotal',
            'positionselect', 'positionopen', 'positionclose',
            'positionselectbysymbol', 'positionselectbyticket',
            
            # Market data functions
            'marketinfo', 'symbolinfo', 'symbolselect', 'symbolscount',
            'ibars', 'itime', 'iopen', 'ihigh', 'ilow', 'iclose', 'ivolume',
            'copybuffer', 'copyrates', 'copytime', 'copyopen', 'copyhigh',
            'copylow', 'copyclose', 'copyvolume', 'copytickvolume',
            
            # Technical indicators
            'ima', 'imacd', 'irsi', 'istochastic', 'ibollingerbands',
            'iadx', 'icci', 'idemar', 'ienvelopes', 'iforce',
            'ifractals', 'igator', 'iichimoku', 'ibwmfi', 'imomentum',
            'imfi', 'iosma', 'isar', 'istddev', 'iwpr', 'izigzag',
            
            # String and conversion functions
            'stringfind', 'stringreplace', 'stringsubstr', 'stringlen',
            'stringtoupper', 'stringtolower', 'doubletostring', 'stringtodouble',
            'integertostring', 'stringtointeger', 'timetostring', 'stringtotime',
            
            # File operations
            'fileopen', 'fileclose', 'filewrite', 'fileread', 'fileseek',
            'filetell', 'filesize', 'fileisexist', 'filedelete', 'filecopy',
            
            # Mathematical functions
            'mathabs', 'matharccos', 'matharcsin', 'matharctan', 'mathcos',
            'mathsin', 'mathtan', 'mathexp', 'mathlog', 'mathmax', 'mathmin',
            'mathmod', 'mathpow', 'mathrand', 'mathround', 'mathsqrt',
            
            # Object-oriented programming
            'cinputparameters', 'cexpertbase', 'ctrade', 'cpositioninfo',
            'corderinfo', 'csymbolinfo', 'caccountinfo', 'chistoryinfo',
            
            # Data types
            'mqlrates', 'mqltick', 'mqlbookinfo', 'mqlparam', 'mqltraderesult',
            'mqltraderequestactions', 'mqltraderesult', 'mqldatetime',
            
            # Constants and enums
            'period_current', 'period_m1', 'period_m5', 'period_m15',
            'period_m30', 'period_h1', 'period_h4', 'period_d1',
            'mode_main', 'mode_signal', 'mode_plusdi', 'mode_minusdi',
            'op_buy', 'op_sell', 'op_buylimit', 'op_selllimit',
            'op_buystop', 'op_sellstop',
            
            # Expert Advisor concepts
            'expert', 'advisor', 'ea', 'robot', 'script', 'indicator',
            'library', 'include', 'template',
            
            # Common MQL5 terms
            'tick', 'bar', 'candle', 'timeframe', 'symbol', 'chart',
            'history', 'real', 'demo', 'backtest', 'optimization',
            'strategy', 'signal', 'trailing', 'stop', 'profit'
        }
        
        # Regex patterns for MQL5-specific syntax and concepts
        self.mql5_patterns: List[re.Pattern] = [
            # MQL5 function calls with parentheses
            re.compile(r'\b(?:OnTick|OnInit|OnDeinit|ArrayResize|OrderSend)\s*\(', re.IGNORECASE),
            
            # MQL5 data types and declarations
            re.compile(r'\b(?:int|double|string|bool|datetime|color|input|extern)\s+\w+', re.IGNORECASE),
            
            # MQL5 preprocessor directives
            re.compile(r'#(?:property|include|import|define)\s+', re.IGNORECASE),
            
            # MQL5 object method calls
            re.compile(r'\w+\.(?:Open|Close|Buy|Sell|Modify|Delete)\s*\(', re.IGNORECASE),
            
            # MQL5 array syntax
            re.compile(r'\w+\[\s*\d*\s*\]\s*(?:=|\[)', re.IGNORECASE),
            
            # MQL5 event handler patterns
            re.compile(r'\bOn(?:Tick|Init|Deinit|Calculate|Tester|Timer|Trade|BookEvent|ChartEvent)\b', re.IGNORECASE),
            
            # MQL5 built-in constants
            re.compile(r'\b(?:PERIOD_|MODE_|OP_|SYMBOL_|ACCOUNT_|TRADE_)[A-Z_]+\b'),
            
            # MQL5 file extensions and contexts (removed .mq4, .ex4)
            re.compile(r'\.(?:mq5|mqh|ex5)\b', re.IGNORECASE),
            
            # MetaTrader 5 specific terms (removed MT4 references)
            re.compile(r'\b(?:MetaTrader\s*5|MT5|Terminal|Strategy Tester|Market Watch)\b', re.IGNORECASE),
            
            # Trading specific patterns
            re.compile(r'\b(?:pip|spread|swap|margin|leverage|balance|equity|free margin)\b', re.IGNORECASE)
        ]
        
        # Context phrases that suggest MQL5 programming (MQL5 specific)
        self.mql5_context_patterns: List[re.Pattern] = [
            re.compile(r'\b(?:expert advisor|trading robot|automated trading|algorithmic trading)\b', re.IGNORECASE),
            re.compile(r'\bmql5\s+(?:code|programming|script|function|syntax)\b', re.IGNORECASE),
            re.compile(r'\bmetatrader\s*5\s+(?:programming|development|coding)\b', re.IGNORECASE),
            re.compile(r'\b(?:forex\s+(?:robot|ea|expert|automation))\b', re.IGNORECASE),
            re.compile(r'\b(?:custom\s+indicator|technical\s+indicator\s+mql5?)\b', re.IGNORECASE)
        ]
    
    def is_mql5_prompt(self, text: str) -> bool:
        """
        Determine if a text prompt is related to MQL5 programming.
        
        Args:
            text: Input text to analyze
            
        Returns:
            bool: True if the text appears to be MQL5-related, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        # Normalize text for analysis
        text_lower = text.lower().strip()
        
        if not text_lower:
            return False
        
        # Quick keyword check for common MQL5 terms
        if self._contains_mql5_keywords(text_lower):
            return True
        
        # Pattern matching for MQL5 syntax and concepts
        if self._matches_mql5_patterns(text):
            return True
        
        # Context pattern matching for MQL5-related discussions
        if self._matches_context_patterns(text):
            return True
        
        return False
    
    def _contains_mql5_keywords(self, text_lower: str) -> bool:
        """Check if text contains MQL5 keywords."""
        # Split text into words for exact matching
        words = re.findall(r'\b\w+\b', text_lower)
        word_set = set(words)
        
        # Check for direct keyword matches
        return bool(self.mql5_keywords.intersection(word_set))
    
    def _matches_mql5_patterns(self, text: str) -> bool:
        """Check if text matches MQL5 syntax patterns."""
        return any(pattern.search(text) for pattern in self.mql5_patterns)
    
    def _matches_context_patterns(self, text: str) -> bool:
        """Check if text matches MQL5 context patterns."""
        return any(pattern.search(text) for pattern in self.mql5_context_patterns)
    
    def get_detection_details(self, text: str) -> dict:
        """
        Get detailed information about MQL5 detection results.
        Useful for debugging and understanding detection logic.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Detection details including matched keywords and patterns
        """
        if not text or not isinstance(text, str):
            return {'is_mql5': False, 'reason': 'Invalid input'}
        
        text_lower = text.lower().strip()
        details = {
            'is_mql5': False,
            'matched_keywords': [],
            'matched_patterns': [],
            'matched_contexts': [],
            'confidence': 0.0
        }
        
        # Check keywords
        words = re.findall(r'\b\w+\b', text_lower)
        word_set = set(words)
        matched_keywords = list(self.mql5_keywords.intersection(word_set))
        
        # Check syntax patterns
        matched_patterns = []
        for i, pattern in enumerate(self.mql5_patterns):
            if pattern.search(text):
                matched_patterns.append(f"pattern_{i}")
        
        # Check context patterns
        matched_contexts = []
        for i, pattern in enumerate(self.mql5_context_patterns):
            if pattern.search(text):
                matched_contexts.append(f"context_{i}")
        
        # Calculate confidence
        confidence = 0.0
        if matched_keywords:
            confidence += min(len(matched_keywords) * 0.3, 0.6)
        if matched_patterns:
            confidence += min(len(matched_patterns) * 0.2, 0.4)
        if matched_contexts:
            confidence += min(len(matched_contexts) * 0.25, 0.5)
        
        confidence = min(confidence, 1.0)
        
        details.update({
            'is_mql5': confidence > 0.3,
            'matched_keywords': matched_keywords,
            'matched_patterns': matched_patterns,
            'matched_contexts': matched_contexts,
            'confidence': round(confidence, 2)
        })
        
        return details


# Global detector instance for efficient reuse
_detector_instance = None

def is_mql5_prompt(text: str) -> bool:
    """
    Primary API function to determine if a text prompt is MQL5-related.
    
    This is the main function that will be used by the PromptProxy.
    
    Args:
        text: Input text to analyze
        
    Returns:
        bool: True if the text appears to be MQL5-related, False otherwise
        
    Examples:
        >>> is_mql5_prompt("How to use ArrayResize in MQL5?")
        True
        >>> is_mql5_prompt("What's the weather today?")
        False
        >>> is_mql5_prompt("OnTick function implementation")
        True
        >>> is_mql5_prompt("Create an Expert Advisor for MetaTrader")
        True
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = MQL5PromptDetector()
    
    return _detector_instance.is_mql5_prompt(text)


def get_mql5_detection_details(text: str) -> dict:
    """
    Get detailed MQL5 detection information for debugging purposes.
    
    Args:
        text: Input text to analyze
        
    Returns:
        dict: Detailed detection results
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = MQL5PromptDetector()
    
    return _detector_instance.get_detection_details(text)


# Test function for validation
def test_mql5_detection():
    """Test the MQL5 detection functionality with sample prompts."""
    test_cases = [
        # Positive cases - should return True
        ("How to use ArrayResize in MQL5?", True),
        ("OnTick function implementation", True),
        ("Create an Expert Advisor for MetaTrader 5", True),
        ("MQL5 OrderSend function parameters", True),
        ("What is the difference between OnInit and OnDeinit?", True),
        ("How to implement trailing stop in MQL5?", True),
        ("MetaTrader programming tutorial", True),
        ("Custom indicator development in MT5", True),
        ("Forex robot automation with MQL5", True),
        ("int array[] = {1, 2, 3}; in MQL5", True),
        ("#property copyright in MQL5", True),
        ("trade.Buy() method in Expert Advisor", True),
        
        # Negative cases - should return False
        ("What's the weather today?", False),
        ("Python programming tutorial", False),
        ("How to cook pasta?", False),
        ("JavaScript array methods", False),
        ("General trading discussion", False),
        ("Stock market analysis", False),
        ("Hello world program", False),
        ("Machine learning algorithms", False),
        
        # Edge cases
        ("", False),
        ("mql", False),  # Too short, not specific enough
        ("I love trading", False),  # Trading but not MQL5 specific
    ]
    
    print("=== MQL5 Prompt Detection Test Results ===")
    passed = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        result = is_mql5_prompt(text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} | Expected: {expected}, Got: {result} | '{text}'")
        
        if result == expected:
            passed += 1
        else:
            # Show detection details for failed cases
            details = get_mql5_detection_details(text)
            print(f"      Details: {details}")
    
    print(f"\nTest Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed == total


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_mql5_detection()
