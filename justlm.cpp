#include "justlm.hpp"
#include "justlm_llama.hpp"



LM::Inference *LM::Inference::construct(const std::string &weights_path, const Params &p) {
    return new LLaMaInference(weights_path, p);
}
