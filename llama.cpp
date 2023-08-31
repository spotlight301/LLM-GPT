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
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return magic == 0x46554747;
}

LM::Inference *construct(const std::string &weights_path, std::ifstream& f, const LM::Inference::Params &p) {
    f.close();
    return new LM::LLaMAInference(weights_path, p);
}
}


__attribute__((constructor))
static void init() {
    llama_backend_init(true);
}

__attribute__((destructor))
static void deinit() {
    llama_backend_free();
}
