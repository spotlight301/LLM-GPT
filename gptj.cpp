#include "justlm_gptj.hpp"
#include "justlm.hpp"

#include <string>
#include <string_view>
#include <fstream>
#include <cstdint>



extern "C" {
const LM::Implementation *get_justlm_implementation() {
    static LM::Implementation fres{false};
    return &fres;
}

bool magic_match(std::istream& f) {
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return magic == 0x67676d6c;
}

LM::Inference *construct(const std::string &weights_path, std::ifstream& f, const LM::Inference::Params &p) {
    return new LM::GPTJInference(weights_path, f, p);
}
}
