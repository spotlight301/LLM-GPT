#include "justlm_llama.hpp"
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
    // Check magic
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x67676a74) return false;
    // Check version
    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    return version >= 3;
}

LM::Inference *construct(const std::string &weights_path, std::ifstream& f, const LM::Inference::Params &p) {
    f.close();
    return new LM::LLaMAInference(weights_path, p);
}
}
