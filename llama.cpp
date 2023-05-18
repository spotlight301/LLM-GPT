#include "justlm_llama.hpp"
#include "justlm.hpp"

#include <string>
#include <string_view>
#include <fstream>
#include <cstdint>



extern "C" {
const LM::Implementation *get_justlm_implementation() {
    static LM::Implementation fres{true};
    return &fres;
}

LM::Inference *construct(const std::string &weights_path, std::ifstream& f, const LM::Inference::Params &p) {
    f.close();
    return new LM::LLaMaInference(weights_path, p);
}
}
