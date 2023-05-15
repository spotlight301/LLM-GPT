#include "justlm.hpp"
#include "justlm_llama.hpp"
#include "justlm_gptj.hpp"
#ifdef LM_MPT
#   include "justlm_mpt.hpp"
#endif

#include <fstream>



LM::Inference *LM::Inference::construct(const std::string &weights_path, const Params &p) {
    // Read magic
    std::ifstream f(weights_path, std::ios::binary);
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    // Create inference instance
    if (magic == 0x67676d6c) {
        f.seekg(0);
        return new GPTJInference(weights_path, f, p);
#   ifdef LM_MPT
    } else if (magic == 0x67676d6d) {
        f.seekg(0);
        return new MPTInference(weights_path, f, p);
#   endif
    } else {
        f.close();
        return new LLaMaInference(weights_path, p);
    }
}
