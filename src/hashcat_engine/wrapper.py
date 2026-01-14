import ctypes
import os 
from src.config import Config 

class CppRuleEngine:
    def __init__(self):
        self.lib_path = str(Config.LIB_RULE_ENGINE_PATH)
        self.rules_path = str(Config.RULES_FILE_PATH)
        self.lib = None

        if not os.path.exists(self.lib_path):
            print(f"[!] Warning: C++ Library not found at {self.lib_path}")
            return
        
        try:
            # 1. Load Library
            self.lib = ctypes.CDLL(self.lib_path)
            
            # 2. Setup 'load_rules_from_file'
            self.lib.load_rules_from_file.argtypes = [ctypes.c_char_p]
            self.lib.load_rules_from_file.restype = ctypes.c_int

            # 3. Setup 'generate_bulk'
            # Args: (input_words_str, output_buffer, buffer_size)
            self.lib.generate_bulk.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_char_p, 
                ctypes.c_int
            ]
            self.lib.generate_bulk.restype = ctypes.c_int

            # 4. Initialize Rules Immediately
            self._init_rules()
            
        except Exception as e:
            print(f"[!] Failed to load C++ library: {e}")
            self.lib = None
        
    def _init_rules(self):
        """Loads the rules file into C++ memory once."""
        if not os.path.exists(self.rules_path):
            print(f"[!] Rule file missing: {self.rules_path}")
            return

        b_path = self.rules_path.encode('utf-8')
        count = self.lib.load_rules_from_file(b_path)
        print(f"[System] C++ Engine loaded {count} rules.")

    def expand_bulk(self, passwords_list):
        """
        Sends a list of passwords to C++, gets back ALL permutations.
        """
        if not self.lib or not passwords_list:
            return passwords_list

        # 1. Prepare Input: Join all passwords with newlines
        # "pass1\npass2\npass3"
        bulk_input = "\n".join(passwords_list).encode('utf-8')
        
        # 2. Estimate Buffer Size
        # (Num Passwords * Num Rules * Avg Length) + Margin
        # 50MB is usually plenty for ~100k-500k results
        buffer_size = 50 * 1024 * 1024 
        output_buffer = ctypes.create_string_buffer(buffer_size)

        # 3. Call C++
        count = self.lib.generate_bulk(bulk_input, output_buffer, buffer_size)

        if count <= 0:
            return passwords_list

        # 4. Decode Output
        # C++ returns a massive string separated by newlines
        raw_output = output_buffer.value.decode('utf-8', errors='ignore')
        return raw_output.split('\n')