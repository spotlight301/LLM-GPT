#include "justlm.hpp"

#include <string_view>



bool LLM::ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}
