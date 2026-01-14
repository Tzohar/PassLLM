#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <unordered_set>

static std::vector<std::string> loaded_rules;

extern "C" {

    int load_rules_from_file(const char* rule_file_path) {
        loaded_rules.clear();
        std::ifstream file(rule_file_path);
        if (!file.is_open()) return -1;

        std::string rule_line;
        while (std::getline(file, rule_line)) {
            if (rule_line.empty() || rule_line[0] == '#') continue;
            if (!rule_line.empty() && rule_line.back() == '\r') rule_line.pop_back();
            
            loaded_rules.push_back(rule_line);
        }
        return loaded_rules.size();
    }

    std::string apply_rule(std::string password, const std::string& rule) {
        size_t len = rule.length();
        for (size_t i = 0; i < len; ++i) {
            char op = rule[i];
            switch (op) {
                case ':': break;
                case 'u': std::transform(password.begin(), password.end(), password.begin(), ::toupper); break;
                case 'l': std::transform(password.begin(), password.end(), password.begin(), ::tolower); break;
                case 'c': if (!password.empty()) password[0] = toupper(password[0]); break;
                case 'r': std::reverse(password.begin(), password.end()); break;
                case 'd': password += password; break;
                case '$': 
                    if (i + 1 < len) { password += rule[i+1]; i++; } 
                    break;
                case '^': 
                    if (i + 1 < len) { password.insert(0, 1, rule[i+1]); i++; } 
                    break;
            }
        }
        return password;
    }

    int generate_bulk(const char* input_words, char* output_buffer, int buffer_size) {
        if (loaded_rules.empty()) return -1; 

        std::unordered_set<std::string> unique_results;
        std::stringstream ss(input_words);
        std::string word;

        while (std::getline(ss, word, '\n')) {
            if (word.empty()) continue;
            if (word.back() == '\r') word.pop_back();

            unique_results.insert(word);

            for (const auto& rule : loaded_rules) {
                unique_results.insert(apply_rule(word, rule));
            }
        }

        int current_offset = 0;
        for (const auto& res : unique_results) {
            int len = res.length();
            if (current_offset + len + 1 >= buffer_size) break;

            std::memcpy(output_buffer + current_offset, res.c_str(), len);
            current_offset += len;
            output_buffer[current_offset] = '\n';
            current_offset++;
        }

        if (current_offset > 0) output_buffer[current_offset - 1] = '\0';
        else output_buffer[0] = '\0';

        return unique_results.size();
    }
}