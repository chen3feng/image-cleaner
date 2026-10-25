#include "json_min.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ic {
namespace {

class Parser {
public:
    Parser(const std::string& text) : text_(text) {}

    bool parse(JsonValue& out, std::string& error) {
        skip_ws();
        if (!parse_value(out)) {
            error = error_.empty() ? "invalid JSON" : error_;
            return false;
        }
        skip_ws();
        if (pos_ != text_.size()) {
            error = "trailing characters after JSON value";
            return false;
        }
        return true;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;
    std::string error_;

    bool eof() const { return pos_ >= text_.size(); }
    char peek() const { return text_[pos_]; }

    void skip_ws() {
        while (!eof()) {
            char c = peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool fail(const char* msg) {
        if (error_.empty()) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%s at offset %zu", msg, pos_);
            error_ = buf;
        }
        return false;
    }

    bool consume_literal(const char* lit) {
        size_t n = 0;
        while (lit[n]) ++n;
        if (text_.compare(pos_, n, lit) != 0) return false;
        pos_ += n;
        return true;
    }

    bool parse_value(JsonValue& out) {
        skip_ws();
        if (eof()) return fail("unexpected end of input");
        char c = peek();
        switch (c) {
            case '{': return parse_object(out);
            case '[': return parse_array(out);
            case '"': {
                out.type = JsonValue::String;
                return parse_string_raw(out.str);
            }
            case 't': {
                if (!consume_literal("true")) return fail("bad literal");
                out.type = JsonValue::Bool;
                out.boolean = true;
                return true;
            }
            case 'f': {
                if (!consume_literal("false")) return fail("bad literal");
                out.type = JsonValue::Bool;
                out.boolean = false;
                return true;
            }
            case 'n': {
                if (!consume_literal("null")) return fail("bad literal");
                out.type = JsonValue::Null;
                return true;
            }
            default:
                if (c == '-' || (c >= '0' && c <= '9')) {
                    out.type = JsonValue::Number;
                    return parse_number(out.number);
                }
                return fail("unexpected token");
        }
    }

    bool parse_object(JsonValue& out) {
        out.type = JsonValue::Object;
        ++pos_;  // {
        skip_ws();
        if (!eof() && peek() == '}') { ++pos_; return true; }
        while (true) {
            skip_ws();
            if (eof() || peek() != '"') return fail("expected key string");
            std::string key;
            if (!parse_string_raw(key)) return false;
            skip_ws();
            if (eof() || peek() != ':') return fail("expected ':'");
            ++pos_;
            JsonValue value;
            if (!parse_value(value)) return false;
            out.obj.emplace(std::move(key), std::move(value));
            skip_ws();
            if (eof()) return fail("unterminated object");
            char c = peek();
            ++pos_;
            if (c == '}') return true;
            if (c != ',') return fail("expected ',' or '}'");
        }
    }

    bool parse_array(JsonValue& out) {
        out.type = JsonValue::Array;
        ++pos_;  // [
        skip_ws();
        if (!eof() && peek() == ']') { ++pos_; return true; }
        while (true) {
            JsonValue value;
            if (!parse_value(value)) return false;
            out.arr.push_back(std::move(value));
            skip_ws();
            if (eof()) return fail("unterminated array");
            char c = peek();
            ++pos_;
            if (c == ']') return true;
            if (c != ',') return fail("expected ',' or ']'");
        }
    }

    bool parse_string_raw(std::string& out) {
        ++pos_;  // opening quote
        while (true) {
            if (eof()) return fail("unterminated string");
            char c = text_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (eof()) return fail("unterminated escape");
                char e = text_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) return fail("bad \\u escape");
                        unsigned int code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = text_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                            else return fail("bad hex digit");
                        }
                        // Encode as UTF-8 (surrogate pairs are not expected in model manifests)
                        if (code < 0x80) {
                            out.push_back(static_cast<char>(code));
                        } else if (code < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        return fail("bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
    }

    bool parse_number(double& out) {
        size_t start = pos_;
        if (!eof() && peek() == '-') ++pos_;
        while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        if (!eof() && peek() == '.') {
            ++pos_;
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!eof() && (peek() == '+' || peek() == '-')) ++pos_;
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (pos_ == start || (pos_ == start + 1 && text_[start] == '-'))
            return fail("bad number");
        out = std::strtod(text_.c_str() + start, nullptr);
        return true;
    }
};

}  // namespace

bool json_parse(const std::string& text, JsonValue& out, std::string& error) {
    Parser parser(text);
    return parser.parse(out, error);
}

}  // namespace ic
