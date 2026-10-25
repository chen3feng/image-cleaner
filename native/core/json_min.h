// Minimal recursive-descent JSON parser. Just enough for models.json:
// objects, arrays, strings, numbers, true/false/null. No allocator tricks,
// no comments, no trailing comma support.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ic {

class JsonValue {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;

    bool is(Type t) const { return type == t; }

    const JsonValue* find(const std::string& key) const {
        if (type != Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }

    std::string get_string(const std::string& key,
                           const std::string& fallback = "") const {
        if (const JsonValue* v = find(key)) return v->type == String ? v->str : fallback;
        return fallback;
    }

    double get_number(const std::string& key, double fallback = 0.0) const {
        if (const JsonValue* v = find(key)) return v->type == Number ? v->number : fallback;
        return fallback;
    }

    int get_int(const std::string& key, int fallback = 0) const {
        return static_cast<int>(get_number(key, fallback));
    }
};

// Returns false and fills error on malformed input.
bool json_parse(const std::string& text, JsonValue& out, std::string& error);

}  // namespace ic
